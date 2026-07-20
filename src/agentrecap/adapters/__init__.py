"""Source adapters that convert tool-specific session logs into the standardized event schema.

Adapters own every source-specific decision; nothing downstream of the exported
threads.csv branches on the source. Each adapter module exposes:

- ``SOURCE``: the value written to the ``source`` column of the events table
- ``PROVIDER``: models.dev provider id used for pricing (e.g. ``anthropic``)
- ``DISPLAY_NAME``: human-readable tool name for CLI output
- ``GRAPH_COLOR``: Matplotlib color used for the tool in report charts
- ``DEFAULT_INPUT``: default directory holding the tool's session files
- ``INPUT_HELP``: help text for the tool's ``--<source>-input`` CLI flag
- ``discover_sessions(path)``: find session files under an input path
- ``convert_thread(path)``: convert one session file into standardized event
  dicts. Every event carries ``source``/``provider``/``model``/``speed``, an
  ``event_kind`` from the closed vocabulary (``user_prompt``,
  ``assistant_message``, ``reasoning``, ``tool_call``, ``tool_result``,
  ``run_end``, ``other``), plus ``is_run_start`` and ``run_end_status`` so run
  boundaries need no source-specific analysis. Anything the analysis does not
  consume is exported as ``other`` with its ``raw_event_type`` preserved.
- ``finalize_events(events)``: given every converted event for the source, mark
  canonical model-call usage rows (``usage_canonical``/``usage_dedup_reason``)
  and fill the normalized ``call_*`` usage columns. Runs across all files at
  once because resumed/forked sessions duplicate calls between files.

To support a new tool, add a module implementing this interface and register
it in ``ADAPTERS``.
"""

import argparse
from pathlib import Path

from . import claude_code, codex, opencode

# Cursor and VS Code remain implemented but are intentionally not registered:
# their local histories do not currently provide reliable token accounting.
ADAPTERS = {adapter.SOURCE: adapter for adapter in (codex, claude_code, opencode)}


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    """Register one --<source>-input flag per registered adapter."""
    for source, adapter in ADAPTERS.items():
        parser.add_argument(
            f"--{source}-input",
            type=Path,
            default=adapter.DEFAULT_INPUT,
            help=adapter.INPUT_HELP,
        )


def inputs_from_args(args: argparse.Namespace) -> dict[str, Path]:
    """Collect the per-source input paths parsed by add_input_arguments."""
    return {source: getattr(args, f"{source}_input") for source in ADAPTERS}
