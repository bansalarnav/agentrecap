"""Gather metadata from supported coding-agent session files."""

from pathlib import Path

import pandas as pd

from .adapters import ADAPTERS 


def convert_sessions(inputs: dict[str, Path], output: Path) -> dict:
    all_events = []
    thread_counts = {}
    for source, input_path in inputs.items():
        adapter = ADAPTERS[source]
        paths = adapter.discover_sessions(input_path)
        source_events = []
        for path in paths:
            source_events.extend(adapter.convert_thread(path))
        # Canonical-usage marking needs every session of a source at once:
        # resumed/forked sessions duplicate calls across files.
        source_events = adapter.finalize_events(source_events)
        thread_counts[source] = len({event["thread_id"] for event in source_events})
        all_events.extend(source_events)

    output.parent.mkdir(parents=True, exist_ok=True)
    events_df = pd.DataFrame(all_events)
    if not events_df.empty:
        events_df = events_df.sort_values(
            ["source", "thread_id", "timestamp", "file_id", "file_event_index"],
            na_position="last",
        ).reset_index(drop=True)
        events_df["event_index"] = events_df.groupby(["source", "thread_id"]).cumcount()

    events_df.to_csv(output, index=False)
    return {
        "threads": thread_counts,
        "events": len(all_events),
    }


