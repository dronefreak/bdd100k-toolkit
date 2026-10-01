"""Rich console helper shared across scripts (single instance, lazily created)."""

from __future__ import annotations

from rich.console import Console


class RichConsoleManager:
    """Lazily-instantiated singleton wrapper around ``rich.console.Console``."""

    _console: Console | None = None

    @classmethod
    def get_console(cls) -> Console:
        """Return the shared ``Console`` instance, creating it on first use."""
        if cls._console is None:
            cls._console = Console()
        return cls._console
