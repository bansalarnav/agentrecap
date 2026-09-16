"""Gather metadata from supported coding-agent session files."""

from datetime import datetime
from pathlib import Path

import pandas as pd

from .adapters import ADAPTERS


def discover_thread_ids(inputs: dict[str, Path]) -> set[str]:
    """Return the anonymized ids of every session currently on disk."""
    thread_ids: set[str] = set()
    for source, input_path in inputs.items():
        adapter = ADAPTERS[source]
        for path in adapter.discover_sessions(input_path):
            thread_ids.update(
                event["thread_id"]
                for event in adapter.convert_thread(path)
                if event.get("thread_id")
            )
    return thread_ids


def convert_sessions(
    inputs: dict[str, Path],
    output: Path,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    thread_ids: set[str] | None = None,
) -> dict:
    all_events = []
    for source, input_path in inputs.items():
        adapter = ADAPTERS[source]
        paths = adapter.discover_sessions(input_path)
        source_events = []
        for path in paths:
            source_events.extend(adapter.convert_thread(path))
        # Canonical-usage marking needs every session of a source at once:
        # resumed/forked sessions duplicate calls across files.
        source_events = adapter.finalize_events(source_events)
        if thread_ids is not None:
            source_events = [
                event for event in source_events if event["thread_id"] in thread_ids
            ]
        all_events.extend(source_events)

    output.parent.mkdir(parents=True, exist_ok=True)
    events_df = pd.DataFrame(all_events)
    if not events_df.empty and (start_time or end_time):
        timestamps = pd.to_datetime(events_df["timestamp"], utc=True, errors="coerce")
        included = timestamps.notna()
        if start_time:
            included &= timestamps.ge(start_time)
        if end_time:
            included &= timestamps.lt(end_time)
        events_df = events_df[included].copy()

    if events_df.empty:
        range_suffix = " in the selected date range" if start_time or end_time else ""
        raise ValueError(f"No coding agent events found{range_suffix}")

    thread_counts = {
        source: group["thread_id"].nunique()
        for source, group in events_df.groupby("source")
    }
    if not events_df.empty:
        events_df = events_df.sort_values(
            ["source", "thread_id", "timestamp", "file_id", "file_event_index"],
            na_position="last",
        ).reset_index(drop=True)
        events_df["event_index"] = events_df.groupby(["source", "thread_id"]).cumcount()

    events_df.to_csv(output, index=False)
    return {
        "threads": thread_counts,
        "events": len(events_df),
    }
