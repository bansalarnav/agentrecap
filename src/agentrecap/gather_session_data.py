"""Gather metadata from Codex and Claude session files."""

import argparse
from pathlib import Path

import pandas as pd

from .adapters import ADAPTERS
from .adapters.common import anonymous_id


def convert_sessions(inputs: dict[str, Path], output: Path) -> dict:
    all_events = []
    thread_counts = {}
    for source, input_path in inputs.items():
        adapter = ADAPTERS[source]
        paths = adapter.discover_sessions(input_path)
        thread_counts[source] = len(paths)
        source_events = []
        for path in paths:
            source_events.extend(adapter.convert_thread(path))
        # Canonical-usage marking needs every session of a source at once:
        # resumed/forked sessions duplicate calls across files.
        all_events.extend(adapter.finalize_events(source_events))

    output.parent.mkdir(parents=True, exist_ok=True)
    events_df = pd.DataFrame(all_events)
    if not events_df.empty:
        events_df = events_df.sort_values(
            ["source", "thread_id", "timestamp", "file_id", "file_event_index"],
            na_position="last",
        ).reset_index(drop=True)
        events_df["event_index"] = events_df.groupby(["source", "thread_id"]).cumcount()
        events_df["row_id"] = [
            anonymous_id(f"row:{source}:{file_id}:{file_event_index}")
            for source, file_id, file_event_index in zip(
                events_df["source"],
                events_df["file_id"],
                events_df["file_event_index"],
                strict=False,
            )
        ]
        call_speeds = events_df.loc[
            events_df["usage_kind"].notna() & events_df["total_tokens"].fillna(0).gt(0),
            ["source", "thread_id", "speed"],
        ]
        thread_speeds = {}
        for key, group in call_speeds.groupby(["source", "thread_id"]):
            values = set(group["speed"].fillna("unknown"))
            thread_speeds[key] = next(iter(values)) if len(values) == 1 else "mixed"
        events_df["thread_speed_status"] = [
            thread_speeds.get((source, thread_id), "unknown")
            for source, thread_id in zip(events_df["source"], events_df["thread_id"], strict=False)
        ]

    events_df.to_csv(output, index=False)
    return {
        "threads": thread_counts,
        "events": len(all_events),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert agent sessions to a metadata-only CSV file.")
    for source, adapter in ADAPTERS.items():
        parser.add_argument(
            f"--{source}-input",
            type=Path,
            default=adapter.DEFAULT_INPUT,
            help=adapter.INPUT_HELP,
        )
    parser.add_argument("--output", type=Path, default=Path("sanitized") / "threads.csv")
    args = parser.parse_args()

    inputs = {source: getattr(args, f"{source}_input") for source in ADAPTERS}
    result = convert_sessions(inputs, args.output)
    summary = " and ".join(
        f"{count} {ADAPTERS[source].DISPLAY_NAME} threads"
        for source, count in result["threads"].items()
    )
    print(f"Converted {summary} ({result['events']} events) into {args.output}")


if __name__ == "__main__":
    main()
