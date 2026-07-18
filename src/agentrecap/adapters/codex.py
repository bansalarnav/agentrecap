"""Adapter for Codex CLI session logs."""

from pathlib import Path

from .common import anonymous_id, anonymous_id_or_none, read_jsonl_records, serialized_length, speed_status

SOURCE = "codex"
DISPLAY_NAME = "Codex"
DEFAULT_INPUT = Path.home() / ".codex"
INPUT_HELP = "Codex home directory, including active and archived sessions"


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


def convert_thread(path: Path) -> list[dict]:
    records = read_jsonl_records(path)
    if not records:
        return []

    meta = next((r.get("payload", {}) for r in records if r.get("type") == "session_meta"), {})
    raw_thread_id = str(meta.get("id") or meta.get("session_id") or path.name)
    thread_id = anonymous_id(f"codex:{raw_thread_id}")
    run_id = anonymous_id(f"codex-run:{path}")
    parent_thread_id = anonymous_id_or_none("codex", meta.get("parent_thread_id") or meta.get("forked_from_id"))
    thread_source = meta.get("thread_source")
    source = meta.get("source") or {}
    is_sidechain = (isinstance(source, dict) and bool(source.get("subagent"))) or thread_source == "subagent"
    model = None
    reasoning_effort = None
    service_tier = None
    tool_names_by_call_id = {}
    events = []

    for event_index, record in enumerate(records):
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

        tool_name = None
        call_id = payload.get("call_id")
        if record_type == "response_item" and payload_type in {"function_call", "custom_tool_call"}:
            tool_name = payload.get("name")
            if call_id and tool_name:
                tool_names_by_call_id[call_id] = tool_name
        elif call_id:
            tool_name = tool_names_by_call_id.get(call_id)

        role = payload.get("role") if payload_type == "message" else None
        usage = info.get("last_token_usage") or {}
        total_usage = info.get("total_token_usage") or {}

        success = None
        if isinstance(payload.get("success"), bool):
            success = payload["success"]
        elif payload_type in {"function_call_output", "custom_tool_call_output"}:
            success = True

        status = None
        if payload_type == "task_complete":
            status = "completed"
        elif payload_type == "turn_aborted":
            status = "aborted"

        content = payload.get("content")
        if content is None:
            content = payload.get("message")

        tool_input = None
        tool_output = None
        if payload_type == "function_call":
            tool_input = payload.get("arguments")
        elif payload_type == "custom_tool_call":
            tool_input = payload.get("input")
        elif payload_type in {"function_call_output", "custom_tool_call_output"}:
            tool_output = payload.get("output")

        event_type = f"{record_type}.{payload_type}" if payload_type else record_type
        is_user_prompt = event_type == "response_item.message" and role == "user"
        is_assistant_message = event_type == "response_item.message" and role == "assistant"
        is_tool_call = event_type in {
            "response_item.function_call",
            "response_item.custom_tool_call",
            "response_item.web_search_call",
            "response_item.image_generation_call",
        }
        is_tool_result = event_type in {
            "response_item.function_call_output",
            "response_item.custom_tool_call_output",
        }
        tool_event_stage = None
        if is_tool_call:
            tool_event_stage = "call"
        elif is_tool_result:
            tool_event_stage = "result"
        elif event_type.startswith("event_msg.") and event_type.endswith("_end"):
            tool_event_stage = "runtime_end"

        events.append(
            {
                "thread_id": thread_id,
                "source": SOURCE,
                "run_id": run_id,
                "run_event_index": event_index,
                "event_index": None,
                "timestamp": record.get("timestamp"),
                "event_id": event_id,
                "parent_event_id": None,
                "parent_thread_id": event_parent_id or parent_thread_id,
                "child_thread_id": child_thread_id,
                "agent_id": thread_id if is_sidechain else None,
                "agent_role": payload.get("new_agent_role") or payload.get("agent_role"),
                "is_sidechain": is_sidechain,
                "source_tool_assistant_id": None,
                "prompt_id": None,
                "request_id": None,
                "message_id": None,
                "tool_call_id": anonymous_id_or_none("codex-tool", call_id),
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
                "model": payload.get("model") or model,
                "reasoning_effort": payload.get("reasoning_effort") or reasoning_effort,
                "speed": speed_status(None, service_tier),
                "service_tier": service_tier,
                "inference_geo": None,
                "usage_kind": "model_call" if usage or total_usage else None,
                "tool_name": tool_name,
                "success": success,
                "status": status,
                "duration_ms": payload.get("duration_ms"),
                "time_to_first_token_ms": payload.get("time_to_first_token_ms"),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "cached_input_tokens": usage.get("cached_input_tokens"),
                "cache_creation_input_tokens": None,
                "cache_creation_5m_input_tokens": None,
                "cache_creation_1h_input_tokens": None,
                "reasoning_output_tokens": usage.get("reasoning_output_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "cumulative_input_tokens": total_usage.get("input_tokens"),
                "cumulative_output_tokens": total_usage.get("output_tokens"),
                "cumulative_cached_input_tokens": total_usage.get("cached_input_tokens"),
                "cumulative_reasoning_output_tokens": total_usage.get("reasoning_output_tokens"),
                "cumulative_total_tokens": total_usage.get("total_tokens"),
                "reported_cost_usd": None,
                "model_context_window": info.get("model_context_window") or payload.get("model_context_window"),
                "text_length": serialized_length(content),
                "tool_input_length": serialized_length(tool_input),
                "tool_output_length": serialized_length(tool_output),
            }
        )

    return events
