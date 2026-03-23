"""VS Code Copilot Agent adapter for hook communication.

Reference: https://code.visualstudio.com/docs/copilot/agent-hooks

Key differences from Claude Code adapter:
- Tool input uses camelCase (e.g., filePath not file_path)
- Different tool names (runInTerminal, editFiles, create_file, etc.)
- Hook event names are PascalCase (PreToolUse, PostToolUse, etc.)
- VS Code ignores matchers — hooks fire for ALL tool invocations
- Additional events: SubagentStart, SubagentStop, Stop, PreCompact

Input format (PreToolUse):
{
    "timestamp": "2026-02-09T10:30:00.000Z",
    "cwd": "/path/to/workspace",
    "sessionId": "session-identifier",
    "hookEventName": "PreToolUse",
    "transcript_path": "/path/to/transcript.json",
    "tool_name": "runInTerminal",
    "tool_input": {"command": "kubectl get pods"},
    "tool_use_id": "tool-123"
}

Output format:
- Exit code 0: Allow; stdout must be valid JSON.
- Exit code 2: Blocking error; stderr shown to model.
- PreToolUse hookSpecificOutput controls per-tool permission.
"""

import json
import os
import sys
from typing import Any

from ..validator import SecurityValidator
from ..logger import AuditLogger, get_environment_context


# VS Code tool name → operation type mapping
# Shell tools
_SHELL_TOOLS = frozenset({
    "runInTerminal", "run_in_terminal",
    "terminal", "bash", "shell",
    "run_command", "run_shell_command",
})

# File read tools
_READ_TOOLS = frozenset({
    "readFile", "read_file",
    "getFileContents", "get_file_contents",
    "openFile", "open_file",
})

# File write/edit tools
_EDIT_TOOLS = frozenset({
    "editFiles", "edit_files",
    "create_file", "createFile",
    "replace_string_in_file", "replaceStringInFile",
    "insert_edit_into_file", "insertEditIntoFile",
    "writeFile", "write_file",
    "applyPatch", "apply_patch",
})


class VSCodeAdapter:
    """Adapter for VS Code Copilot Agent hook protocol."""

    # Map CLI --event values to internal event type strings
    EVENT_MAP = {
        # PascalCase (as sent by VS Code in hookEventName)
        "PreToolUse": "pre_tool_use",
        "PostToolUse": "post_tool_use",
        "SessionStart": "session_start",
        "Stop": "stop",
        "UserPromptSubmit": "user_prompt_submit",
        "SubagentStart": "subagent_start",
        "SubagentStop": "subagent_stop",
        "PreCompact": "pre_compact",
        # Lowercase aliases for CLI convenience
        "pre": "pre_tool_use",
        "post": "post_tool_use",
        "session-start": "session_start",
        "stop": "stop",
        "prompt": "user_prompt_submit",
        "subagent-start": "subagent_start",
        "subagent-stop": "subagent_stop",
        "pre-compact": "pre_compact",
    }

    def __init__(
        self,
        validator: SecurityValidator,
        logger: AuditLogger,
        debug: bool = False,
    ):
        self.validator = validator
        self.logger = logger
        self.debug = debug

    def _debug(self, message: str) -> None:
        if self.debug:
            print(f"[DEBUG:vscode] {message}", file=sys.stderr)

    def _get_context(self, input_data: dict) -> dict[str, str | None]:
        """Get context from environment and VS Code's common input fields."""
        ctx = get_environment_context()
        # VS Code passes sessionId directly in the JSON input
        if not ctx.get("session_id"):
            ctx["session_id"] = input_data.get("sessionId")
        # VS Code passes cwd in the JSON input
        if not ctx.get("project_dir"):
            ctx["project_dir"] = input_data.get("cwd") or os.environ.get("PWD")
        return ctx

    def handle(self, event: str, input_data: dict[str, Any]) -> int:
        """
        Handle a VS Code Copilot Agent hook event.

        Args:
            event: The event type (e.g., "PreToolUse", "pre").
            input_data: JSON input from VS Code.

        Returns:
            Exit code (0 = allow/success, 2 = block).
        """
        # VS Code also sends hookEventName inside the JSON — prefer that
        # so a hook script doesn't need to know its own event type.
        hook_event_name = input_data.get("hookEventName", event)
        event_type = self.EVENT_MAP.get(hook_event_name) or self.EVENT_MAP.get(event, event)

        self._debug(f"Handling event: {event_type} (raw: {event!r}, hookEventName: {hook_event_name!r})")
        self._debug(f"Input: {json.dumps(input_data)}")

        if event_type == "pre_tool_use":
            return self._handle_pre_tool_use(input_data)
        elif event_type == "post_tool_use":
            return self._handle_post_tool_use(input_data)
        elif event_type == "session_start":
            return self._handle_session_start(input_data)
        elif event_type == "stop":
            return self._handle_stop(input_data)
        elif event_type == "user_prompt_submit":
            return self._handle_user_prompt(input_data)
        elif event_type == "subagent_start":
            return self._handle_subagent_start(input_data)
        elif event_type == "subagent_stop":
            return self._handle_subagent_stop(input_data)
        elif event_type == "pre_compact":
            return self._handle_pre_compact(input_data)
        else:
            # SECURITY: Fail-secure — block unknown event types
            self._debug(f"Unknown event type: {event_type} - blocking for security")
            output = {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"Unknown event type: {event_type}",
                }
            }
            print(json.dumps(output))
            sys.stdout.flush()
            return 2

    def _extract_tool_info(self, input_data: dict) -> tuple[str, str, str, str]:
        """
        Extract (tool_name, command, file_path, operation) from VS Code tool input.

        VS Code uses camelCase for tool_input keys (filePath not file_path),
        and different tool names than Claude Code.
        """
        tool_name = input_data.get("tool_name", "")
        # VS Code uses tool_input (same key as Claude Code)
        tool_input = input_data.get("tool_input") or {}

        command = ""
        file_path = ""
        operation = "shell"

        if tool_name in _SHELL_TOOLS:
            # runInTerminal uses "command" or "input"
            command = tool_input.get("command", tool_input.get("input", ""))
            operation = "shell"
        elif tool_name in _READ_TOOLS:
            # VS Code uses camelCase: filePath
            file_path = tool_input.get("filePath", tool_input.get("file_path", ""))
            operation = "read"
        elif tool_name in _EDIT_TOOLS:
            # editFiles may have a list of files; create_file / replace_string_in_file use filePath
            file_path = (
                tool_input.get("filePath")
                or tool_input.get("file_path")
                or _first_file(tool_input.get("files"))
                or ""
            )
            operation = "edit"
        else:
            # Unknown tool — try common field names without converting whole dict
            command = (
                tool_input.get("command", "")
                or tool_input.get("cmd", "")
                or tool_input.get("script", "")
                or tool_input.get("input", "")
            )
            file_path = tool_input.get("filePath", tool_input.get("file_path", ""))
            if not command and not file_path:
                self._debug(f"Unknown VS Code tool with no command/file field: {tool_name}")

        return tool_name, command, file_path, operation

    def _handle_pre_tool_use(self, input_data: dict) -> int:
        """Handle PreToolUse events — validate and possibly block."""
        tool_name, command, file_path, operation = self._extract_tool_info(input_data)
        tool_use_id = input_data.get("tool_use_id", "")

        self._debug(f"Tool: {tool_name}, Command: {command!r}, File: {file_path!r}, Op: {operation}")

        result = self.validator.validate(
            command=command,
            file_path=file_path,
            operation=operation,
        )

        self._debug(f"Validation result: {result}")

        ctx = self._get_context(input_data)
        self.logger.log_pre_execution(
            platform="vscode",
            event_type="pre_tool_use",
            tool_name=tool_name,
            command=command,
            file_path=file_path,
            decision=result.decision,
            reason=result.reason,
            matched_rules=result.matched_rules,
            severity=result.severity.value if result.severity else None,
            category=result.category,
            tool_use_id=tool_use_id,
            **ctx,
        )

        if result.decision == "block":
            output = {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": result.reason or "Blocked by security policy",
                }
            }
            print(json.dumps(output))
            sys.stdout.flush()
            return 2

        if result.decision == "ask":
            output = {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "ask",
                    "permissionDecisionReason": f"Confirmation required: {result.reason}",
                }
            }
            print(json.dumps(output))
            sys.stdout.flush()
            return 2

        # Allow
        print(json.dumps({}))
        sys.stdout.flush()
        return 0

    def _handle_post_tool_use(self, input_data: dict) -> int:
        """Handle PostToolUse events — logging only."""
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input") or {}
        tool_response = input_data.get("tool_response", "")

        # Extract what we can for logging
        command = tool_input.get("command", tool_input.get("input", ""))
        file_path = (
            tool_input.get("filePath")
            or tool_input.get("file_path")
            or _first_file(tool_input.get("files"))
            or ""
        )

        ctx = self._get_context(input_data)
        self.logger.log_post_execution(
            platform="vscode",
            event_type="post_tool_use",
            tool_name=tool_name,
            command=command,
            file_path=file_path,
            tool_use_id=input_data.get("tool_use_id", ""),
            **ctx,
        )

        print(json.dumps({}))
        sys.stdout.flush()
        return 0

    def _handle_session_start(self, input_data: dict) -> int:
        """Handle SessionStart events."""
        source = input_data.get("source", "new")

        ctx = self._get_context(input_data)
        self.logger.log_pre_execution(
            platform="vscode",
            event_type="session_start",
            decision="allow",
            source=source,
            **ctx,
        )

        print(json.dumps({}))
        sys.stdout.flush()
        return 0

    def _handle_stop(self, input_data: dict) -> int:
        """Handle Stop events — session ended."""
        stop_hook_active = input_data.get("stop_hook_active", False)

        ctx = self._get_context(input_data)
        self.logger.log_post_execution(
            platform="vscode",
            event_type="session_end",
            stop_hook_active=stop_hook_active,
            **ctx,
        )

        print(json.dumps({}))
        sys.stdout.flush()
        return 0

    def _handle_user_prompt(self, input_data: dict) -> int:
        """Handle UserPromptSubmit events — log prompt, always allow."""
        prompt = input_data.get("prompt", "")

        ctx = self._get_context(input_data)
        self.logger.log_pre_execution(
            platform="vscode",
            event_type="user_prompt_submit",
            command=prompt[:500],
            decision="allow",
            **ctx,
        )

        print(json.dumps({}))
        sys.stdout.flush()
        return 0

    def _handle_subagent_start(self, input_data: dict) -> int:
        """Handle SubagentStart events — log only."""
        agent_id = input_data.get("agent_id", "")
        agent_type = input_data.get("agent_type", "")

        ctx = self._get_context(input_data)
        self.logger.log_pre_execution(
            platform="vscode",
            event_type="subagent_start",
            decision="allow",
            agent_id=agent_id,
            agent_type=agent_type,
            **ctx,
        )

        print(json.dumps({}))
        sys.stdout.flush()
        return 0

    def _handle_subagent_stop(self, input_data: dict) -> int:
        """Handle SubagentStop events — log only."""
        agent_id = input_data.get("agent_id", "")
        agent_type = input_data.get("agent_type", "")
        stop_hook_active = input_data.get("stop_hook_active", False)

        ctx = self._get_context(input_data)
        self.logger.log_post_execution(
            platform="vscode",
            event_type="subagent_stop",
            agent_id=agent_id,
            agent_type=agent_type,
            stop_hook_active=stop_hook_active,
            **ctx,
        )

        print(json.dumps({}))
        sys.stdout.flush()
        return 0

    def _handle_pre_compact(self, input_data: dict) -> int:
        """Handle PreCompact events — log only."""
        trigger = input_data.get("trigger", "auto")

        ctx = self._get_context(input_data)
        self.logger.log_pre_execution(
            platform="vscode",
            event_type="pre_compact",
            decision="allow",
            trigger=trigger,
            **ctx,
        )

        print(json.dumps({}))
        sys.stdout.flush()
        return 0


def _first_file(files: Any) -> str:
    """Return the first file path from a list, or empty string."""
    if isinstance(files, list) and files:
        first = files[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return first.get("filePath", first.get("file_path", ""))
    return ""
