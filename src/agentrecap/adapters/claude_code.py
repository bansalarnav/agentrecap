"""Adapter for Claude Code session logs."""

from pathlib import Path

from .common import anonymous_id, anonymous_id_or_none, read_jsonl_records, serialized_length, speed_status

SOURCE = "claude"
DISPLAY_NAME = "Claude Code"
DEFAULT_INPUT = Path.home() / ".claude" / "projects"
INPUT_HELP = "Claude Code projects directory"


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
    run_id = anonymous_id(f"claude-run:{path}")
    tool_names_by_id = {}
    events = []

    for record_index, record in enumerate(records):
        usage_record = record
        nested = (record.get("data") or {}).get("message") or {}
        if (nested.get("message") or {}).get("usage"):
            usage_record = nested

        record_type = usage_record.get("type", record.get("type", "unknown"))
        message = usage_record.get("message") or {}
        role = message.get("role") if record_type in {"user", "assistant"} else None
        model = message.get("model")
        usage = message.get("usage") or {}
        blocks = message.get("content")
        raw_message_id = message.get("id")
        event_id = anonymous_id_or_none(
            "claude-event", usage_record.get("uuid") or record.get("uuid") or raw_message_id
        )
        parent_event_id = anonymous_id_or_none(
            "claude-event",
            record.get("parentUuid") or record.get("logicalParentUuid"),
        )
        agent_id = anonymous_id_or_none(
            "claude-agent",
            usage_record.get("agentId")
            or record.get("agentId")
            or record.get("attributionAgent"),
        )
        source_tool_assistant_id = anonymous_id_or_none("claude-event", record.get("sourceToolAssistantUUID"))
        prompt_id = anonymous_id_or_none("claude-prompt", record.get("promptId"))
        request_id = anonymous_id_or_none(
            "claude-request", usage_record.get("requestId") or record.get("requestId")
        )
        if not isinstance(blocks, list) or not blocks:
            blocks = [None]

        for block_index, block in enumerate(blocks):
            block = block if isinstance(block, dict) else {}
            block_type = block.get("type")
            tool_name = None
            success = None
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
                success = not block.get("is_error", False)
            elif block_type == "text":
                text = block.get("text")
            elif block_type == "thinking":
                text = block.get("thinking")

            event_type = f"{record_type}.{block_type}" if block_type else record_type
            is_user_prompt = event_type == "user.text"
            is_assistant_message = event_type == "assistant.text"
            is_tool_call = event_type == "assistant.tool_use"
            is_tool_result = event_type == "user.tool_result"
            tool_event_stage = None
            if is_tool_call:
                tool_event_stage = "call"
            elif is_tool_result:
                tool_event_stage = "result"

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
                {
                    "thread_id": thread_id,
                    "source": SOURCE,
                    "run_id": run_id,
                    "run_event_index": len(events),
                    "record_index": record_index,
                    "block_index": block_index,
                    "event_index": None,
                    "timestamp": usage_record.get("timestamp") or record.get("timestamp"),
                    "event_id": event_id,
                    "parent_event_id": parent_event_id,
                    "parent_thread_id": None,
                    "child_thread_id": None,
                    "agent_id": agent_id,
                    "agent_role": None,
                    "is_sidechain": usage_record.get(
                        "isSidechain", record.get("isSidechain")
                    ),
                    "source_tool_assistant_id": source_tool_assistant_id,
                    "prompt_id": prompt_id,
                    "request_id": request_id,
                    "message_id": anonymous_id_or_none("claude-message", raw_message_id),
                    "tool_call_id": anonymous_id_or_none("claude-tool", block.get("id") or block.get("tool_use_id")),
                    "role": role,
                    "event_type": event_type,
                    "canonical_event_type": (
                        "user_prompt"
                        if is_user_prompt
                        else "assistant_message"
                        if is_assistant_message
                        else "tool_call"
                        if is_tool_call
                        else "tool_result"
                        if is_tool_result
                        else "other"
                    ),
                    "is_user_prompt": is_user_prompt,
                    "is_assistant_message": is_assistant_message,
                    "is_tool_call": is_tool_call,
                    "is_tool_result": is_tool_result,
                    "tool_event_stage": tool_event_stage,
                    "model": model,
                    "reasoning_effort": None,
                    "speed": speed_status(usage.get("speed"), usage.get("service_tier")),
                    "service_tier": usage.get("service_tier"),
                    "inference_geo": usage.get("inference_geo"),
                    "usage_kind": "model_call" if include_usage else None,
                    "tool_name": tool_name,
                    "success": success,
                    "status": None,
                    "duration_ms": None,
                    "time_to_first_token_ms": None,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cached_input_tokens": cached_tokens,
                    "cache_creation_input_tokens": cache_creation_tokens,
                    "cache_creation_5m_input_tokens": cache_creation_5m_tokens,
                    "cache_creation_1h_input_tokens": cache_creation_1h_tokens,
                    "reasoning_output_tokens": None,
                    "total_tokens": total_tokens,
                    "cumulative_input_tokens": None,
                    "cumulative_output_tokens": None,
                    "cumulative_cached_input_tokens": None,
                    "cumulative_reasoning_output_tokens": None,
                    "cumulative_total_tokens": None,
                    "reported_cost_usd": (
                        usage_record.get("costUSD", record.get("costUSD"))
                        if include_usage
                        else None
                    ),
                    "model_context_window": None,
                    "text_length": serialized_length(text),
                    "tool_input_length": serialized_length(tool_input),
                    "tool_output_length": serialized_length(tool_output),
                }
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
                or f"record-{record_index}"
            )
            events.append(
                {
                    **{key: None for key in events[-1]},
                    "thread_id": thread_id,
                    "source": SOURCE,
                    "run_id": run_id,
                    "run_event_index": len(events),
                    "record_index": record_index,
                    "block_index": None,
                    "event_index": None,
                    "timestamp": usage_record.get("timestamp") or record.get("timestamp"),
                    "event_id": anonymous_id_or_none(
                        "claude-event", f"{advisor_identity}:advisor:{iteration_index}"
                    ),
                    "parent_event_id": event_id,
                    "agent_id": agent_id,
                    "is_sidechain": usage_record.get(
                        "isSidechain", record.get("isSidechain")
                    ),
                    "request_id": request_id,
                    "message_id": anonymous_id_or_none(
                        "claude-message", f"{advisor_identity}:advisor:{iteration_index}"
                    ),
                    "event_type": "assistant.advisor_usage",
                    "canonical_event_type": "other",
                    "model": advisor_model,
                    "speed": speed_status(
                        iteration.get("speed"), iteration.get("service_tier")
                    ),
                    "service_tier": iteration.get("service_tier"),
                    "inference_geo": iteration.get("inference_geo"),
                    "usage_kind": "advisor_call",
                    "input_tokens": advisor_input,
                    "output_tokens": advisor_output,
                    "cached_input_tokens": advisor_cached,
                    "cache_creation_input_tokens": advisor_creation,
                    "cache_creation_5m_input_tokens": advisor_5m,
                    "cache_creation_1h_input_tokens": advisor_1h,
                    "total_tokens": sum(
                        value or 0
                        for value in (
                            advisor_input,
                            advisor_output,
                            advisor_cached,
                            advisor_creation,
                        )
                    ),
                    "reported_cost_usd": None,
                }
            )

    return events
