"""Source adapters that convert tool-specific session logs into the shared event schema.

Each adapter module exposes the same interface:

- ``SOURCE``: the value written to the ``source`` column of the events table
- ``discover_sessions(path)``: find session files under an input path
- ``convert_thread(path)``: convert one session file into a list of event dicts

To support a new tool (e.g. Cursor), add a module implementing this interface.
"""

from . import claude_code, codex

__all__ = ["claude_code", "codex"]
