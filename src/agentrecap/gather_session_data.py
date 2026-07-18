"""Gather metadata from Codex and Claude session files."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


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


def discover_codex_jsonl(path: Path) -> list[Path]:
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


def convert_codex_thread(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

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
    tool_names_by_call_id = {}
    events = []

    for event_index, record in enumerate(records):
        record_type = record.get("type", "unknown")
        payload = record.get("payload") or {}
        payload_type = payload.get("type")

        if record_type == "turn_context":
            model = payload.get("model")
            reasoning_effort = payload.get("effort")

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
        info = payload.get("info") or {}
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
                "source": "codex",
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
                "model_context_window": info.get("model_context_window") or payload.get("model_context_window"),
                "text_length": serialized_length(content),
                "tool_input_length": serialized_length(tool_input),
                "tool_output_length": serialized_length(tool_output),
            }
        )

    return events


def convert_claude_thread(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        return []

    raw_thread_id = str(next((r.get("sessionId") for r in records if r.get("sessionId")), path.name))
    thread_id = anonymous_id(f"claude:{raw_thread_id}")
    run_id = anonymous_id(f"claude-run:{path}")
    tool_names_by_id = {}
    events = []

    for record_index, record in enumerate(records):
        record_type = record.get("type", "unknown")
        message = record.get("message") or {}
        role = message.get("role") if record_type in {"user", "assistant"} else None
        model = message.get("model")
        usage = message.get("usage") or {}
        blocks = message.get("content")
        event_id = anonymous_id_or_none("claude-event", record.get("uuid") or message.get("id"))
        parent_event_id = anonymous_id_or_none(
            "claude-event",
            record.get("parentUuid") or record.get("logicalParentUuid"),
        )
        agent_id = anonymous_id_or_none("claude-agent", record.get("agentId") or record.get("attributionAgent"))
        source_tool_assistant_id = anonymous_id_or_none("claude-event", record.get("sourceToolAssistantUUID"))
        prompt_id = anonymous_id_or_none("claude-prompt", record.get("promptId"))
        request_id = anonymous_id_or_none("claude-request", record.get("requestId"))
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
            total_tokens = None
            if include_usage:
                total_tokens = sum(
                    value or 0
                    for value in (input_tokens, output_tokens, cached_tokens, cache_creation_tokens)
                )

            events.append(
                {
                    "thread_id": thread_id,
                    "source": "claude",
                    "run_id": run_id,
                    "run_event_index": len(events),
                    "record_index": record_index,
                    "block_index": block_index,
                    "event_index": None,
                    "timestamp": record.get("timestamp"),
                    "event_id": event_id,
                    "parent_event_id": parent_event_id,
                    "parent_thread_id": None,
                    "child_thread_id": None,
                    "agent_id": agent_id,
                    "agent_role": None,
                    "is_sidechain": record.get("isSidechain"),
                    "source_tool_assistant_id": source_tool_assistant_id,
                    "prompt_id": prompt_id,
                    "request_id": request_id,
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
                    "model_context_window": None,
                    "text_length": serialized_length(text),
                    "tool_input_length": serialized_length(tool_input),
                    "tool_output_length": serialized_length(tool_output),
                }
            )

    return events


def convert_sessions(
    codex_input: Path,
    claude_input: Path,
    output: Path,
) -> dict[str, int]:
    all_events = []
    codex_paths = discover_codex_jsonl(codex_input)
    claude_paths = [claude_input] if claude_input.is_file() else sorted(claude_input.rglob("*.jsonl"))

    for path in codex_paths:
        all_events.extend(convert_codex_thread(path))
    for path in claude_paths:
        all_events.extend(convert_claude_thread(path))

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
