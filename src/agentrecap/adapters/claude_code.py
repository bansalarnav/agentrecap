"""Adapter for Claude Code session logs."""

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

SOURCE = "claude"
PROVIDER = "anthropic"
DISPLAY_NAME = "Claude Code"
DEFAULT_INPUT = Path.home() / ".claude" / "projects"
INPUT_HELP = "Claude Code projects directory"

# (record type, content-block type) -> standardized event kind. Anything not
# listed here (attachments, queue operations, titles, config records, ...) is
# exported as "other" so the archive stays complete without growing the
# analysis vocabulary.
BLOCK_KINDS = {
    ("user", "text"): "user_prompt",
    ("assistant", "text"): "assistant_message",
    ("assistant", "thinking"): "reasoning",
    ("assistant", "tool_use"): "tool_call",
    ("user", "tool_result"): "tool_result",
}

USAGE_TOKEN_KEYS = (
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
)


def discover_sessions(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    return sorted(path.rglob("*.jsonl"))


def convert_thread(path: Path) -> list[dict]:
    records = read_jsonl_records(path)
    if not records:
        return []

    raw_thread_id = str(next((r.get("sessionId") for r in records if r.get("sessionId")), path.name))
    thread_id = anonymous_id(f"claude:{raw_thread_id}")
    file_id = anonymous_id(f"claude-file:{path}")
    tool_names_by_id = {}
    current_model = None
    events = []

    for record in records:
        usage_record = record
        nested = (record.get("data") or {}).get("message") or {}
        if (nested.get("message") or {}).get("usage"):
            usage_record = nested

        def field(key: str):
            """Prefer the nested usage record's value, falling back to the raw record."""
            return usage_record.get(key) or record.get(key)

        record_type = usage_record.get("type", record.get("type", "unknown"))
        message = usage_record.get("message") or {}
        role = message.get("role") if record_type in {"user", "assistant"} else None
        record_model = message.get("model")
        if record_model:
            current_model = record_model
        usage = message.get("usage") or {}
        blocks = message.get("content")
        raw_message_id = message.get("id")
        event_id = anonymous_id_or_none("claude-event", field("uuid") or raw_message_id)
        parent_event_id = anonymous_id_or_none(
            "claude-event",
            record.get("parentUuid") or record.get("logicalParentUuid"),
        )
        agent_id = anonymous_id_or_none(
            "claude-agent", field("agentId") or record.get("attributionAgent")
        )
        spawned_by_event_id = anonymous_id_or_none("claude-event", record.get("sourceToolAssistantUUID"))
        request_id = anonymous_id_or_none("claude-request", field("requestId"))
        is_sidechain = usage_record.get("isSidechain", record.get("isSidechain"))
        timestamp = field("timestamp")
        if not isinstance(blocks, list) or not blocks:
            blocks = [None]

        for block_index, block in enumerate(blocks):
            block = block if isinstance(block, dict) else {}
            block_type = block.get("type")
            tool_name = None
            tool_success = None
            tool_input = None
            tool_output = None
            text = None

            if block_type == "tool_use":
                tool_name = block.get("name")
                tool_input = block.get("input")
                if block.get("id") and tool_name:
                    tool_names_by_id[block["id"]] = tool_name
            elif block_type == "tool_result":
                tool_name = tool_names_by_id.get(block.get("tool_use_id"))
                tool_output = block.get("content")
                tool_success = not block.get("is_error", False)
            elif block_type == "text":
                text = block.get("text")
            elif block_type == "thinking":
                text = block.get("thinking")

            raw_event_type = f"{record_type}.{block_type}" if block_type else record_type
            event_kind = BLOCK_KINDS.get((record_type, block_type), "other")
            # Blockless user-role records mark run boundaries too (e.g. resumed
            # prompts), even though they are not canonical user prompts.
            is_run_start = event_kind == "user_prompt" or (
                raw_event_type == "user" and role == "user"
            )

            include_usage = block_index == 0 and bool(usage)
            input_tokens = usage.get("input_tokens") if include_usage else None
            output_tokens = usage.get("output_tokens") if include_usage else None
            cached_tokens = usage.get("cache_read_input_tokens") if include_usage else None
            cache_creation_tokens = usage.get("cache_creation_input_tokens") if include_usage else None
            cache_creation = usage.get("cache_creation") or {}
            cache_creation_5m_tokens = (
                cache_creation.get("ephemeral_5m_input_tokens") if include_usage else None
            )
            cache_creation_1h_tokens = (
                cache_creation.get("ephemeral_1h_input_tokens") if include_usage else None
            )
            if include_usage and cache_creation_tokens is None and cache_creation:
                cache_creation_tokens = (cache_creation_5m_tokens or 0) + (
                    cache_creation_1h_tokens or 0
                )
            total_tokens = None
            if include_usage:
                total_tokens = sum(
                    value or 0
                    for value in (input_tokens, output_tokens, cached_tokens, cache_creation_tokens)
                )

            events.append(
                base_event(
                    SOURCE,
                    PROVIDER,
                    thread_id,
                    file_id,
                    len(events),
                    stream_id=f"agent:{agent_id}" if agent_id else "main",
                    timestamp=timestamp,
                    event_id=event_id,
                    parent_event_id=parent_event_id,
                    agent_id=agent_id,
                    is_sidechain=is_sidechain,
                    spawned_by_event_id=spawned_by_event_id,
                    event_kind=event_kind,
                    raw_event_type=raw_event_type,
                    is_run_start=is_run_start,
                    model=record_model or current_model,
                    speed=speed_status(usage.get("speed"), usage.get("service_tier")),
                    service_tier=usage.get("service_tier"),
                    inference_geo=usage.get("inference_geo"),
                    message_id=anonymous_id_or_none("claude-message", raw_message_id),
                    request_id=request_id,
                    tool_call_id=anonymous_id_or_none(
                        "claude-tool", block.get("id") or block.get("tool_use_id")
                    ),
                    tool_name=tool_name,
                    tool_success=tool_success,
                    usage_kind="model_call" if include_usage else None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_input_tokens=cached_tokens,
                    cache_creation_input_tokens=cache_creation_tokens,
                    cache_creation_5m_input_tokens=cache_creation_5m_tokens,
                    cache_creation_1h_input_tokens=cache_creation_1h_tokens,
                    total_tokens=total_tokens,
                    reported_cost_usd=(
                        usage_record.get("costUSD", record.get("costUSD"))
                        if include_usage
                        else None
                    ),
                    text_length=serialized_length(text),
                    tool_input_length=serialized_length(tool_input),
                    tool_output_length=serialized_length(tool_output),
                )
            )

        for iteration_index, iteration in enumerate(usage.get("iterations") or []):
            if iteration.get("type") != "advisor_message":
                continue
            advisor_model = iteration.get("model") or record.get("advisorModel")
            advisor_cache = iteration.get("cache_creation") or {}
            advisor_5m = advisor_cache.get("ephemeral_5m_input_tokens")
            advisor_1h = advisor_cache.get("ephemeral_1h_input_tokens")
            advisor_creation = iteration.get("cache_creation_input_tokens")
            if advisor_creation is None and advisor_cache:
                advisor_creation = (advisor_5m or 0) + (advisor_1h or 0)
            advisor_input = iteration.get("input_tokens")
            advisor_output = iteration.get("output_tokens")
            advisor_cached = iteration.get("cache_read_input_tokens")
            advisor_identity = (
                raw_message_id
                or usage_record.get("uuid")
                or record.get("uuid")
                or f"event-{len(events)}"
            )
            events.append(
                base_event(
                    SOURCE,
                    PROVIDER,
                    thread_id,
                    file_id,
                    len(events),
                    stream_id=f"agent:{agent_id}" if agent_id else "main",
                    timestamp=timestamp,
                    event_id=anonymous_id_or_none(
                        "claude-event", f"{advisor_identity}:advisor:{iteration_index}"
                    ),
                    parent_event_id=event_id,
                    agent_id=agent_id,
                    is_sidechain=is_sidechain,
                    raw_event_type="assistant.advisor_usage",
                    model=advisor_model,
                    speed=speed_status(
                        iteration.get("speed"), iteration.get("service_tier")
                    ),
                    service_tier=iteration.get("service_tier"),
                    inference_geo=iteration.get("inference_geo"),
                    message_id=anonymous_id_or_none(
                        "claude-message", f"{advisor_identity}:advisor:{iteration_index}"
                    ),
                    request_id=request_id,
                    usage_kind="advisor_call",
                    input_tokens=advisor_input,
                    output_tokens=advisor_output,
                    cached_input_tokens=advisor_cached,
                    cache_creation_input_tokens=advisor_creation,
                    cache_creation_5m_input_tokens=advisor_5m,
                    cache_creation_1h_input_tokens=advisor_1h,
                    total_tokens=sum(
                        value or 0
                        for value in (
                            advisor_input,
                            advisor_output,
                            advisor_cached,
                            advisor_creation,
                        )
                    ),
                )
            )

    return events


def _usage_score(event: dict) -> tuple:
    return (
        event["is_sidechain"] is not True,
        sum(event.get(key) or 0 for key in USAGE_TOKEN_KEYS),
        event.get("speed") != "unknown",
        sum(event.get(key) is not None for key in USAGE_TOKEN_KEYS),
    )


def finalize_events(events: list[dict]) -> list[dict]:
    """Mark canonical model-call usage rows across all converted sessions.

    Resumed and forked sessions replay earlier records into new files, and
    sidechain streams can repeat the main stream's usage, so the same API call
    can appear several times. Duplicates are kept but marked with a
    usage_dedup_reason; the best-scoring copy of each call becomes canonical.
    """
    init_usage_fields(events)
    usage_events = sorted(
        (
            event
            for event in events
            if any(event.get(key) is not None for key in USAGE_TOKEN_KEYS)
        ),
        key=event_sort_key,
    )

    survivors: list[dict] = []
    exact_identity: dict[tuple, int] = {}
    message_positions: dict[str, list[int]] = {}
    request_identity: dict[str, int] = {}
    fallback_identity: dict[tuple, int] = {}
    for event in usage_events:
        message_id = event.get("message_id")
        request_id = event.get("request_id")
        position = None
        match_reason = None
        fingerprint = None
        if message_id:
            position = exact_identity.get((message_id, request_id))
            if position is None:
                # A repeated message_id with a different request_id is only the
                # same call when one copy is a sidechain replay of the other.
                for candidate_position in message_positions.get(message_id, []):
                    candidate = survivors[candidate_position]
                    if event["is_sidechain"] is True or candidate["is_sidechain"] is True:
                        position = candidate_position
                        break
            if position is not None:
                match_reason = "duplicate_message_id"
        elif request_id:
            position = request_identity.get(request_id)
            if position is not None:
                match_reason = "duplicate_request_id"
        else:
            fingerprint = (
                event["thread_id"],
                event.get("timestamp"),
                event.get("model"),
                event.get("usage_kind"),
                *(event.get(key) for key in USAGE_TOKEN_KEYS),
            )
            position = fallback_identity.get(fingerprint)
            if position is not None:
                match_reason = "duplicate_fingerprint"

        if position is None:
            position = len(survivors)
            survivors.append(event)
            if message_id:
                message_positions.setdefault(message_id, []).append(position)
            elif request_id:
                request_identity[request_id] = position
            else:
                fallback_identity[fingerprint] = position
        else:
            existing = survivors[position]
            if _usage_score(event) > _usage_score(existing):
                existing["usage_dedup_reason"] = match_reason
                survivors[position] = event
            else:
                event["usage_dedup_reason"] = match_reason
        if message_id:
            exact_identity[(message_id, request_id)] = position

    for event in survivors:
        uncached = event.get("input_tokens") or 0
        cached = event.get("cached_input_tokens") or 0
        creation = event.get("cache_creation_input_tokens") or 0
        mark_canonical_usage(
            event,
            "request_usage",
            served_input_tokens=uncached + cached + creation,
            cached_input_tokens=cached,
            cache_creation_input_tokens=creation,
            cache_creation_5m_input_tokens=event.get("cache_creation_5m_input_tokens") or 0,
            cache_creation_1h_input_tokens=event.get("cache_creation_1h_input_tokens") or 0,
            output_tokens=event.get("output_tokens") or 0,
        )
    return events
