"""Platform-specific adapters for AI tool hooks."""

from .claude import ClaudeAdapter
from .cursor import CursorAdapter
from .gemini import GeminiAdapter
from .vscode import VSCodeAdapter

__all__ = [
    "ClaudeAdapter",
    "CursorAdapter",
    "GeminiAdapter",
    "VSCodeAdapter",
]

