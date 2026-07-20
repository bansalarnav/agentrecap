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


def base_event(
    source: str,
    provider: str,
    thread_id: str,
    file_id: str,
    file_event_index: int | None,
    **values: object,
) -> dict:
    """One standardized event carrying the full shared column set.

    Every adapter builds its events through this factory so the exported
    schema stays identical across sources; unknown field names are rejected
    to catch typos at conversion time.
    """
    event = {
        "source": source,
        "provider": provider,
        "thread_id": thread_id,
        "stream_id": "main",
        "file_id": file_id,
        "file_event_index": file_event_index,
        "event_index": None,
        "timestamp": None,
        "event_id": None,
        "parent_event_id": None,
        "agent_id": None,
        "is_sidechain": False,
        "parent_thread_id": None,
        "child_thread_id": None,
        "spawned_by_event_id": None,
        "event_kind": "other",
        "raw_event_type": None,
        "is_run_start": False,
        "run_end_status": None,
        "duration_ms": None,
        "time_to_first_token_ms": None,
        "model": None,
        "reasoning_effort": None,
        "speed": "unknown",
        "service_tier": None,
        "inference_geo": None,
        "message_id": None,
        "request_id": None,
        "tool_call_id": None,
        "tool_name": None,
        "tool_success": None,
        "usage_kind": None,
        "input_tokens": None,
        "output_tokens": None,
        "cached_input_tokens": None,
        "cache_creation_input_tokens": None,
        "cache_creation_5m_input_tokens": None,
        "cache_creation_1h_input_tokens": None,
        "reasoning_output_tokens": None,
        "total_tokens": None,
        "cumulative_input_tokens": None,
        "cumulative_output_tokens": None,
        "cumulative_cached_input_tokens": None,
        "cumulative_reasoning_output_tokens": None,
        "reported_cost_usd": None,
        "text_length": None,
        "tool_input_length": None,
        "tool_output_length": None,
    }
    unknown = values.keys() - event.keys()
    if unknown:
        raise ValueError(f"Unknown event fields: {sorted(unknown)}")
    event.update(values)
    return event


def mark_canonical_usage(
    event: dict,
    usage_source: str,
    *,
    served_input_tokens: int,
    cached_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cache_creation_5m_input_tokens: int = 0,
    cache_creation_1h_input_tokens: int = 0,
    output_tokens: int = 0,
    reasoning_output_tokens: int | None = None,
    zero_usage_reason: str = "zero_usage",
) -> None:
    """Mark one event as a canonical model call with normalized call_* usage.

    Events with no usage at all are flagged with ``zero_usage_reason`` instead
    so they stay auditable without counting as model calls.
    """
    if not any(
        value > 0
        for value in (
            served_input_tokens,
            cached_input_tokens,
            cache_creation_input_tokens,
            output_tokens,
        )
    ):
        event["usage_dedup_reason"] = zero_usage_reason
        return
    event["usage_canonical"] = True
    event["usage_source"] = usage_source
    event["call_served_input_tokens"] = served_input_tokens
    event["call_cached_input_tokens"] = cached_input_tokens
    event["call_cache_creation_input_tokens"] = cache_creation_input_tokens
    event["call_cache_creation_5m_input_tokens"] = cache_creation_5m_input_tokens
    event["call_cache_creation_1h_input_tokens"] = cache_creation_1h_input_tokens
    event["call_output_tokens"] = output_tokens
    event["call_reasoning_output_tokens"] = reasoning_output_tokens


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
