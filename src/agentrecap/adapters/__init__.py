"""Source adapters that convert tool-specific session logs into the shared event schema.

Each adapter module exposes the same interface:

- ``SOURCE``: the value written to the ``source`` column of the events table
- ``DISPLAY_NAME``: human-readable tool name for CLI output
- ``DEFAULT_INPUT``: default directory holding the tool's session files
- ``INPUT_HELP``: help text for the tool's ``--<source>-input`` CLI flag
- ``discover_sessions(path)``: find session files under an input path
- ``convert_thread(path)``: convert one session file into a list of event dicts

To support a new tool (e.g. Cursor), add a module implementing this interface
and register it in ``ADAPTERS``.
"""

from . import claude_code, codex

ADAPTERS = {adapter.SOURCE: adapter for adapter in (codex, claude_code)}
