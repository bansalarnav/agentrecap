"""Adapter for Codex CLI session logs."""

from pathlib import Path

from .common import (
    anonymous_id,
    anonymous_id_or_none,
    base_event,
    event_sort_key,
    init_usage_fields,
    mark_canonical_usage,
    read_jsonl_records,
    serialized_length,
    speed_status,
)

SOURCE = "codex"
PROVIDER = "openai"
DISPLAY_NAME = "Codex"
GRAPH_COLOR = "tab:blue"
DEFAULT_INPUT = Path.home() / ".codex"
INPUT_HELP = "Codex home directory, including active and archived sessions"

TOOL_CALL_TYPES = {
    "function_call",
    "custom_tool_call",
    "web_search_call",
    "image_generation_call",
}
TOOL_RESULT_TYPES = {"function_call_output", "custom_tool_call_output"}
# Built-in calls whose payloads carry no tool name of their own.
DEFAULT_TOOL_NAMES = {
    "web_search_call": "web_search",
    "image_generation_call": "image_generation",
}
RUN_END_STATUSES = {"task_complete": "completed", "turn_aborted": "aborted"}

CUMULATIVE_KEYS = (
    "cumulative_input_tokens",
    "cumulative_output_tokens",
    "cumulative_cached_input_tokens",
    "cumulative_reasoning_output_tokens",
)


def discover_sessions(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    sessions_dir = path / "sessions"
    if not sessions_dir.is_dir():
        return sorted(path.rglob("*.jsonl"))

    paths = list(sessions_dir.rglob("*.jsonl"))
    archived_dir = path / "archived_sessions"
    if archived_dir.is_dir():
        paths.extend(archived_dir.rglob("*.jsonl"))
    return sorted(paths)


def _event_kind(record_type: str, payload_type: str | None, role: str | None) -> str:
    if record_type == "response_item":
        if payload_type == "message":
            if role == "user":
                return "user_prompt"
            if role == "assistant":
                return "assistant_message"
        elif payload_type == "reasoning":
            return "reasoning"
        elif payload_type in TOOL_CALL_TYPES:
            return "tool_call"
        elif payload_type in TOOL_RESULT_TYPES:
            return "tool_result"
    elif record_type == "event_msg" and payload_type in RUN_END_STATUSES:
        return "run_end"
    return "other"


def convert_thread(path: Path) -> list[dict]:
    records = read_jsonl_records(path)
    if not records:
        return []

    meta = next((r.get("payload", {}) for r in records if r.get("type") == "session_meta"), {})
    raw_thread_id = str(meta.get("id") or meta.get("session_id") or path.name)
    thread_id = anonymous_id(f"codex:{raw_thread_id}")
    file_id = anonymous_id(f"codex-file:{path}")
    thread_source = meta.get("thread_source")
    source = meta.get("source") or {}
    subagent_source = source.get("subagent") if isinstance(source, dict) else None
    thread_spawn = (
        subagent_source.get("thread_spawn")
        if isinstance(subagent_source, dict)
        else None
    )
    raw_parent_thread_id = (
        meta.get("parent_thread_id")
        or meta.get("forked_from_id")
        or (
            thread_spawn.get("parent_thread_id")
            if isinstance(thread_spawn, dict)
            else None
        )
    )
    parent_thread_id = anonymous_id_or_none("codex", raw_parent_thread_id)
    is_sidechain = bool(subagent_source) or thread_source == "subagent"
    model = None
    reasoning_effort = None
    service_tier = None
    tool_names_by_call_id = {}
    events = []

    for record in records:
        record_type = record.get("type", "unknown")
        payload = record.get("payload") or {}
        payload_type = payload.get("type")
        info = payload.get("info") or {}

        if record_type == "turn_context":
            model = payload.get("model")
            reasoning_effort = payload.get("effort")

        record_service_tier = (
            payload.get("service_tier")
            or payload.get("serviceTier")
            or (payload.get("thread_settings") or {}).get("service_tier")
            or info.get("service_tier")
            or info.get("serviceTier")
        )
        if record_service_tier is not None:
            service_tier = record_service_tier

        event_id = anonymous_id_or_none("codex-event", payload.get("id") or payload.get("turn_id"))
        event_parent_id = None
        if payload.get("parent_thread_id") or payload.get("forked_from_id"):
            event_parent_id = anonymous_id_or_none(
                "codex",
                payload.get("parent_thread_id") or payload.get("forked_from_id"),
            )
        if payload.get("sender_thread_id"):
            event_parent_id = anonymous_id_or_none("codex", payload.get("sender_thread_id"))

        child_thread_id = None
        if payload_type == "collab_agent_spawn_end":
            child_thread_id = anonymous_id_or_none("codex", payload.get("new_thread_id"))

        role = payload.get("role") if payload_type == "message" else None
        event_kind = _event_kind(record_type, payload_type, role)

        tool_name = None
        call_id = payload.get("call_id")
        if event_kind == "tool_call":
            tool_name = payload.get("name") or DEFAULT_TOOL_NAMES.get(payload_type)
            if call_id and tool_name:
                tool_names_by_call_id[call_id] = tool_name
        elif call_id:
            tool_name = tool_names_by_call_id.get(call_id)

        usage = info.get("last_token_usage") or {}
        total_usage = info.get("total_token_usage") or {}

        tool_success = None
        if isinstance(payload.get("success"), bool):
            tool_success = payload["success"]
        elif payload_type in TOOL_RESULT_TYPES:
            tool_success = True

        content = payload.get("content")
        if content is None:
            content = payload.get("message")

        tool_input = None
        tool_output = None
        if payload_type == "function_call":
            tool_input = payload.get("arguments")
        elif payload_type == "custom_tool_call":
            tool_input = payload.get("input")
        elif payload_type in TOOL_RESULT_TYPES:
            tool_output = payload.get("output")

        events.append(
            base_event(
                SOURCE,
                PROVIDER,
                thread_id,
                file_id,
                len(events),
                timestamp=record.get("timestamp"),
                event_id=event_id,
                agent_id=thread_id if is_sidechain else None,
                is_sidechain=is_sidechain,
                parent_thread_id=event_parent_id or parent_thread_id,
                child_thread_id=child_thread_id,
                event_kind=event_kind,
                raw_event_type=f"{record_type}.{payload_type}" if payload_type else record_type,
                is_run_start=record_type == "event_msg" and payload_type == "user_message",
                run_end_status=RUN_END_STATUSES.get(payload_type) if event_kind == "run_end" else None,
                duration_ms=payload.get("duration_ms"),
                time_to_first_token_ms=payload.get("time_to_first_token_ms"),
                model=payload.get("model") or model,
                reasoning_effort=payload.get("reasoning_effort") or reasoning_effort,
                speed=speed_status(None, service_tier),
                service_tier=service_tier,
                tool_call_id=anonymous_id_or_none("codex-tool", call_id),
                tool_name=tool_name,
                tool_success=tool_success,
                usage_kind="model_call" if usage or total_usage else None,
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                cached_input_tokens=usage.get("cached_input_tokens"),
                reasoning_output_tokens=usage.get("reasoning_output_tokens"),
                total_tokens=usage.get("total_tokens"),
                cumulative_input_tokens=total_usage.get("input_tokens"),
                cumulative_output_tokens=total_usage.get("output_tokens"),
                cumulative_cached_input_tokens=total_usage.get("cached_input_tokens"),
                cumulative_reasoning_output_tokens=total_usage.get("reasoning_output_tokens"),
                text_length=serialized_length(content),
                tool_input_length=serialized_length(tool_input),
                tool_output_length=serialized_length(tool_output),
            )
        )

    return events


def finalize_events(events: list[dict]) -> list[dict]:
    """Mark canonical model-call usage rows across all converted sessions.

    Codex reports usage as a cumulative per-thread ledger, so canonical calls
    are the deltas between successive distinct ledger snapshots. Session files
    without a ledger fall back to the last per-call snapshot per file. All
    duplicate or superseded rows are kept, marked with a usage_dedup_reason.
    """
    init_usage_fields(events)

    ledger_events = [
        event
        for event in events
        if event.get("cumulative_input_tokens") is not None
        or event.get("cumulative_output_tokens") is not None
    ]
    ledger_files = {event["file_id"] for event in ledger_events}

    by_thread: dict[str, list[dict]] = {}
    for event in ledger_events:
        by_thread.setdefault(event["thread_id"], []).append(event)

    snapshots_by_thread = {
        thread_id: {
            tuple(event.get(key) for key in CUMULATIVE_KEYS)
            for event in thread_events
        }
        for thread_id, thread_events in by_thread.items()
    }

    for thread_events in by_thread.values():
        thread_events.sort(key=event_sort_key)
        parent_thread_ids = {
            event["parent_thread_id"]
            for event in thread_events
            if event.get("parent_thread_id")
        }
        parent_snapshots = set().union(
            *(snapshots_by_thread.get(thread_id, set()) for thread_id in parent_thread_ids)
        )
        in_replay_prefix = bool(parent_snapshots)
        seen_observations: set[tuple] = set()
        previous = None
        for event in thread_events:
            cumulative_snapshot = tuple(event.get(key) for key in CUMULATIVE_KEYS)

            # Forked Codex rollouts can begin with the parent's full cumulative
            # ledger. Preserve the last replayed value as the child's baseline,
            # but do not count any of that copied prefix as new API traffic.
            if in_replay_prefix and cumulative_snapshot in parent_snapshots:
                event["usage_dedup_reason"] = "fork_replay_baseline"
                previous = event
                continue
            in_replay_prefix = False

            observation = (
                event.get("timestamp"),
                event.get("model"),
                *cumulative_snapshot,
            )
            if observation in seen_observations:
                event["usage_dedup_reason"] = "duplicate_cumulative_snapshot"
                continue
            seen_observations.add(observation)
            deltas = {}
            for key in CUMULATIVE_KEYS:
                raw = event.get(key)
                filled = raw if raw is not None else 0
                previous_raw = previous.get(key) if previous is not None else None
                if raw is None or previous_raw is None:
                    delta = filled
                else:
                    delta = raw - previous_raw
                    if delta < 0:
                        delta = filled
                deltas[key] = delta
            previous = event
            # Repeated event_msg.token_count snapshots have zero cumulative
            # delta and are not additional model calls.
            mark_canonical_usage(
                event,
                "cumulative_ledger_delta",
                served_input_tokens=deltas["cumulative_input_tokens"],
                cached_input_tokens=deltas["cumulative_cached_input_tokens"],
                output_tokens=deltas["cumulative_output_tokens"],
                reasoning_output_tokens=deltas["cumulative_reasoning_output_tokens"],
                zero_usage_reason="zero_delta_snapshot",
            )

    snapshot_events = sorted(
        (
            event
            for event in events
            if event["file_id"] not in ledger_files
            and (
                event.get("input_tokens") is not None
                or event.get("output_tokens") is not None
            )
        ),
        key=event_sort_key,
    )
    last_by_snapshot: dict[tuple, dict] = {}
    for event in snapshot_events:
        snapshot = (
            event["thread_id"],
            event["file_id"],
            event.get("model"),
            event.get("input_tokens"),
            event.get("output_tokens"),
            event.get("cached_input_tokens"),
            event.get("reasoning_output_tokens"),
        )
        if snapshot in last_by_snapshot:
            last_by_snapshot[snapshot]["usage_dedup_reason"] = "duplicate_snapshot"
        last_by_snapshot[snapshot] = event
    for event in last_by_snapshot.values():
        mark_canonical_usage(
            event,
            "deduplicated_last_snapshot",
            served_input_tokens=event.get("input_tokens") or 0,
            cached_input_tokens=event.get("cached_input_tokens") or 0,
            output_tokens=event.get("output_tokens") or 0,
            reasoning_output_tokens=event.get("reasoning_output_tokens"),
        )

    for event in events:
        if (
            not event["usage_canonical"]
            and event["usage_dedup_reason"] is None
            and event["file_id"] in ledger_files
            and event.get("cumulative_input_tokens") is None
            and event.get("cumulative_output_tokens") is None
            and (
                event.get("input_tokens") is not None
                or event.get("output_tokens") is not None
            )
        ):
            event["usage_dedup_reason"] = "superseded_by_ledger"
    return events
