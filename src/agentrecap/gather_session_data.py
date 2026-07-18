"""Gather metadata from Codex and Claude session files."""

import argparse
from pathlib import Path

import pandas as pd

from .adapters import claude_code, codex
from .adapters.common import anonymous_id


def convert_sessions(
    codex_input: Path,
    claude_input: Path,
    output: Path,
) -> dict[str, int]:
    all_events = []
    codex_paths = codex.discover_sessions(codex_input)
    claude_paths = claude_code.discover_sessions(claude_input)

    for path in codex_paths:
        all_events.extend(codex.convert_thread(path))
    for path in claude_paths:
        all_events.extend(claude_code.convert_thread(path))

    output.parent.mkdir(parents=True, exist_ok=True)
    events_df = pd.DataFrame(all_events)
    if not events_df.empty:
        events_df = events_df.sort_values(
            ["source", "thread_id", "timestamp", "run_id", "run_event_index"],
            na_position="last",
        ).reset_index(drop=True)
        events_df["event_index"] = events_df.groupby(["source", "thread_id"]).cumcount()
        events_df["row_id"] = [
            anonymous_id(f"row:{source}:{run_id}:{run_event_index}")
            for source, run_id, run_event_index in zip(
                events_df["source"],
                events_df["run_id"],
                events_df["run_event_index"],
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
        "codex_threads": len(codex_paths),
        "claude_threads": len(claude_paths),
        "events": len(all_events),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Codex and Claude Code sessions to a metadata-only CSV file.")
    parser.add_argument(
        "--codex-input",
        type=Path,
        default=Path.home() / ".codex",
        help="Codex home directory, including active and archived sessions",
    )
    parser.add_argument(
        "--claude-input",
        type=Path,
        default=Path.home() / ".claude" / "projects",
        help="Claude Code projects directory",
    )
    parser.add_argument("--output", type=Path, default=Path("sanitized") / "threads.csv")
    args = parser.parse_args()

    result = convert_sessions(args.codex_input, args.claude_input, args.output)
    print(
        f"Converted {result['codex_threads']} Codex threads and {result['claude_threads']} Claude threads "
        f"({result['events']} events) into {args.output}"
    )


if __name__ == "__main__":
    main()
