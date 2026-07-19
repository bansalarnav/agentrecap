"""Helpers shared by all session adapters."""

import hashlib
import json
from pathlib import Path


def anonymous_id(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def anonymous_id_or_none(namespace: str, value: object) -> str | None:
    if value is None:
        return None
    return anonymous_id(f"{namespace}:{value}")


def serialized_length(value: object) -> int | None:
    if value is None:
        return None
    return len(json.dumps(value, ensure_ascii=False))


def speed_status(speed: object, service_tier: object) -> str:
    values = {str(value).lower() for value in (speed, service_tier) if value is not None}
    if values & {"fast", "priority"}:
        return "fast"
    if values & {"standard", "default"}:
        return "standard"
    return "unknown"


# Written by every adapter's finalize_events pass. ``usage_canonical`` marks the
# rows that count as real model calls; duplicates stay in the export with a
# ``usage_dedup_reason`` so dedup decisions can be audited or redone later. The
# ``call_*`` columns hold the normalized per-call usage for canonical rows so
# downstream analysis does not need source-specific token accounting.
USAGE_FINALIZE_FIELDS = {
    "usage_canonical": False,
    "usage_dedup_reason": None,
    "usage_source": None,
    "call_served_input_tokens": None,
    "call_cached_input_tokens": None,
    "call_cache_creation_input_tokens": None,
    "call_cache_creation_5m_input_tokens": None,
    "call_cache_creation_1h_input_tokens": None,
    "call_output_tokens": None,
    "call_reasoning_output_tokens": None,
}


def init_usage_fields(events: list[dict]) -> None:
    for event in events:
        event.update(USAGE_FINALIZE_FIELDS)


def event_sort_key(event: dict) -> tuple:
    """Chronological order within a thread, matching the exported event_index."""
    timestamp = event.get("timestamp")
    return (
        event["thread_id"],
        timestamp is None,
        timestamp or "",
        event["file_id"] or "",
        event["file_event_index"],
    )


def read_jsonl_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
