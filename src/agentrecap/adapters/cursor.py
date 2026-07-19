"""Adapter for Cursor (AI editor) chat/composer sessions.

Cursor persists conversations in SQLite key-value stores:

- ``<User>/globalStorage/state.vscdb``, table ``cursorDiskKV``:
  ``composerData:<composerId>`` records hold per-conversation metadata
  (``createdAt``, ``unifiedMode``, ``fullConversationHeadersOnly``) and, in
  older schema versions, an inline ``conversation`` array or
  ``conversationMap``. Newer versions store each message separately under
  ``bubbleId:<composerId>:<bubbleId>``.
- ``<User>/workspaceStorage/<hash>/state.vscdb``, table ``ItemTable``:
  ``composer.composerData`` lists the workspace's composers (older versions
  embed full conversations here) and legacy chat lives under
  ``workbench.panel.aichat.view.aichat.chatdata`` as tabs of bubbles.

Bubbles use ``type`` 1 for user and 2 for assistant messages; assistant
bubbles may carry ``thinking`` text, ``toolFormerData`` (a tool invocation
with args/result/status), a ``tokenCount`` with per-message input/output
tokens, and a ``modelType`` naming the serving model. Cursor is
multi-provider, so the provider is inferred per event from the model name.

Databases are always opened read-only (``mode=ro``) so the user's live Cursor
state is never touched.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from .common import (
    anonymous_id,
    anonymous_id_or_none,
    event_sort_key,
    init_usage_fields,
    serialized_length,
    speed_status,
)

SOURCE = "cursor"
# Fallback only: Cursor serves many providers and each event's provider is
# inferred from the model name recorded on the bubble. Cursor's default agent
# models have been Anthropic's Claude family, so unknown models price there.
PROVIDER = "anthropic"
DISPLAY_NAME = "Cursor"
DEFAULT_INPUT = Path.home() / "Library" / "Application Support" / "Cursor" / "User"
INPUT_HELP = "Cursor user-data directory containing globalStorage and workspaceStorage state.vscdb databases"

# Substring -> models.dev provider id, checked in order against the
# lowercased model name. OpenAI's o-series needs prefix checks to avoid
# matching inside unrelated names.
_PROVIDER_SUBSTRINGS = (
    ("claude", "anthropic"),
    ("sonnet", "anthropic"),
    ("opus", "anthropic"),
    ("haiku", "anthropic"),
    ("gemini", "google"),
    ("deepseek", "deepseek"),
    ("grok", "xai"),
    ("kimi", "moonshotai"),
    ("gpt", "openai"),
    ("codex", "openai"),
)
_OPENAI_PREFIXES = ("o1", "o3", "o4")

_TOOL_SUCCESS_STATUSES = {"completed", "success", "succeeded", "done", "finished"}
_TOOL_FAILURE_STATUSES = {"error", "errored", "failed", "failure", "cancelled", "canceled", "aborted", "rejected"}


def discover_sessions(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    paths = []
    global_db = path / "globalStorage" / "state.vscdb"
    if global_db.is_file():
        paths.append(global_db)
    workspace_dir = path / "workspaceStorage"
    if workspace_dir.is_dir():
        paths.extend(sorted(workspace_dir.glob("*/state.vscdb")))
    if not paths:
        paths = sorted(path.rglob("state.vscdb"))
    return paths


def _read_json_kv(path: Path, table: str, like_patterns: tuple[str, ...]) -> dict:
    """Read matching key/value rows from a Cursor SQLite store, read-only."""
    uri = f"file:{quote(str(path), safe='/:')}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error:
        return {}
    try:
        where = " OR ".join("key LIKE ?" for _ in like_patterns)
        rows = connection.execute(
            f"SELECT key, value FROM {table} WHERE {where}", like_patterns
        ).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        connection.close()

    records = {}
    for key, value in rows:
        if value is None:
            continue
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        try:
            records[key] = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            continue
    return records


def _provider_for_model(model: str | None) -> str:
    if not model:
        return PROVIDER
    name = str(model).lower()
    for substring, provider in _PROVIDER_SUBSTRINGS:
        if substring in name:
            return provider
    if name in _OPENAI_PREFIXES or name.startswith(tuple(f"{p}-" for p in _OPENAI_PREFIXES)):
        return "openai"
    return PROVIDER


def _iso_timestamp(value: object) -> str | None:
    """Convert Cursor timestamps to ISO-8601 UTC."""
    if isinstance(value, str):
        try:
            moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    seconds = value / 1000 if value > 1e11 else value
    try:
        moment = datetime.fromtimestamp(seconds, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    return moment.isoformat().replace("+00:00", "Z")


def _model_name(bubble: dict) -> str | None:
    raw = (
        bubble.get("modelType")
        or bubble.get("model")
        or bubble.get("modelName")
        or bubble.get("modelId")
    )
    if isinstance(raw, dict):
        raw = raw.get("modelName") or raw.get("name")
    return str(raw) if raw else None


def _tool_success(status: object) -> bool | None:
    if status is None:
        return None
    name = str(status).lower()
    if name in _TOOL_SUCCESS_STATUSES:
        return True
    if name in _TOOL_FAILURE_STATUSES:
        return False
    return None


def _base_event(thread_id: str, file_id: str, file_event_index: int) -> dict:
    """One standardized event with the full shared column set."""
    return {
        "source": SOURCE,
        "provider": PROVIDER,
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
        "speed": speed_status(None, None),
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
        "cumulative_total_tokens": None,
        "reported_cost_usd": None,
        "model_context_window": None,
        "text_length": None,
        "tool_input_length": None,
        "tool_output_length": None,
    }


def _composer_bubble_events(
    events: list[dict],
    bubble: dict,
    thread_id: str,
    file_id: str,
    fallback_timestamp: str | None,
    current_model: str | None,
) -> str | None:
    """Emit standardized events for one composer bubble; returns the model."""
    model = _model_name(bubble) or current_model
    bubble_id = bubble.get("bubbleId")
    server_bubble_id = bubble.get("serverBubbleId")
    bubble_type = bubble.get("type")
    role = {1: "user", 2: "assistant"}.get(bubble_type)

    timing = bubble.get("timingInfo")
    if not isinstance(timing, dict):
        timing = {}
    timestamp = (
        _iso_timestamp(timing.get("clientStartTime") or timing.get("clientRpcSendTime"))
        or _iso_timestamp(bubble.get("createdAt"))
        or fallback_timestamp
    )
    duration_ms = None
    if isinstance(timing.get("clientStartTime"), (int, float)) and isinstance(
        timing.get("clientEndTime"), (int, float)
    ):
        duration_ms = timing["clientEndTime"] - timing["clientStartTime"]
        if duration_ms < 0:
            duration_ms = None

    text = bubble.get("text") or None
    thinking = bubble.get("thinking")
    if thinking is None:
        thinking = bubble.get("allThinkingBlocks")
    thinking_text = None
    if isinstance(thinking, dict):
        thinking_text = thinking.get("text")
    elif isinstance(thinking, list):
        thinking_text = thinking or None
    elif isinstance(thinking, str):
        thinking_text = thinking
    # Some versions flag thought bubbles instead of nesting the text.
    if role == "assistant" and thinking_text is None and bubble.get("isThought"):
        thinking_text, text = text, None
    tool = bubble.get("toolFormerData") if isinstance(bubble.get("toolFormerData"), dict) else None

    # (event_kind, raw suffix, text, tool fields) per standardized sub-event.
    subevents = []
    if bubble.get("_missing"):
        subevents.append(("other", "missing", None, None, None, None))
    elif role == "user":
        subevents.append(("user_prompt", "user", text, None, None, None))
    elif role == "assistant":
        if thinking_text is not None:
            subevents.append(("reasoning", "ai.thinking", thinking_text, None, None, None))
        if tool is not None:
            tool_name = tool.get("tool") or tool.get("name") or tool.get("toolName") or None
            tool_input = tool.get("rawArgs")
            if tool_input is None:
                tool_input = tool.get("params")
            subevents.append(("tool_call", "ai.tool_call", None, tool_name, tool_input, None))
            if tool.get("result") is not None:
                subevents.append(("tool_result", "ai.tool_result", None, tool_name, None, tool.get("result")))
        if text is not None:
            subevents.append(("assistant_message", "ai.text", text, None, None, None))
        if not subevents:
            subevents.append(("other", "ai", None, None, None, None))
    else:
        subevents.append(("other", f"type_{bubble_type}", None, None, None, None))

    token_count = bubble.get("tokenCount")
    if not isinstance(token_count, dict):
        token_count = {}
    input_tokens = token_count.get("inputTokens")
    output_tokens = token_count.get("outputTokens")
    has_usage = input_tokens is not None or output_tokens is not None

    tool_success = _tool_success(tool.get("status")) if tool is not None else None
    tool_call_identity = (tool.get("toolCallId") if tool is not None else None) or bubble_id

    for sub_index, (kind, suffix, sub_text, tool_name, tool_input, tool_output) in enumerate(subevents):
        event = _base_event(thread_id, file_id, len(events))
        first = sub_index == 0
        event.update(
            {
                "timestamp": timestamp,
                "event_id": anonymous_id_or_none(
                    "cursor-event", f"{bubble_id}:{sub_index}" if bubble_id else None
                ),
                "event_kind": kind,
                "raw_event_type": f"composer.bubble.{suffix}",
                "is_run_start": kind == "user_prompt",
                "duration_ms": duration_ms if first else None,
                "model": model,
                "provider": _provider_for_model(model),
                "message_id": anonymous_id_or_none("cursor-message", server_bubble_id or bubble_id),
                "request_id": anonymous_id_or_none("cursor-request", bubble.get("usageUuid")),
                "text_length": serialized_length(sub_text),
            }
        )
        if kind in {"tool_call", "tool_result"}:
            event["tool_call_id"] = anonymous_id_or_none("cursor-tool", tool_call_identity)
            event["tool_name"] = tool_name
            event["tool_input_length"] = serialized_length(tool_input)
            event["tool_output_length"] = serialized_length(tool_output)
            if kind == "tool_result":
                event["tool_success"] = tool_success
        if first and has_usage:
            event["usage_kind"] = "model_call"
            event["input_tokens"] = input_tokens
            event["output_tokens"] = output_tokens
            event["total_tokens"] = (input_tokens or 0) + (output_tokens or 0)
        events.append(event)
    return model


def _convert_composer(events: list[dict], composer: dict, kv: dict, path: Path, file_id: str) -> None:
    composer_id = composer.get("composerId")
    raw_thread_id = str(composer_id or f"{path}:{composer.get('createdAt')}")
    thread_id = anonymous_id(f"cursor:{raw_thread_id}")
    created_timestamp = _iso_timestamp(composer.get("createdAt"))

    conversation = composer.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        conversation = []
        conversation_map = composer.get("conversationMap")
        if not isinstance(conversation_map, dict):
            conversation_map = {}
        for header in composer.get("fullConversationHeadersOnly") or []:
            header = header if isinstance(header, dict) else {}
            bubble_id = header.get("bubbleId")
            bubble = kv.get(f"bubbleId:{composer_id}:{bubble_id}") if bubble_id else None
            if not isinstance(bubble, dict) and bubble_id:
                bubble = conversation_map.get(bubble_id)
            if isinstance(bubble, dict):
                conversation.append({**bubble, "bubbleId": bubble.get("bubbleId") or bubble_id})
            else:
                # Header without a stored bubble: keep the row as "other".
                conversation.append(
                    {"bubbleId": bubble_id, "type": header.get("type"), "_missing": True}
                )
        if not conversation and conversation_map:
            conversation = [
                {**bubble, "bubbleId": bubble.get("bubbleId") or bubble_id}
                for bubble_id, bubble in conversation_map.items()
                if isinstance(bubble, dict)
            ]

    model_config = composer.get("modelConfig")
    current_model = _model_name(model_config) if isinstance(model_config, dict) else None
    current_model = current_model or _model_name(composer)
    for bubble in conversation:
        if not isinstance(bubble, dict):
            bubble = {"_missing": True}
        current_model = _composer_bubble_events(
            events, bubble, thread_id, file_id, created_timestamp, current_model
        )


def _convert_legacy_chat(events: list[dict], chat_data: dict, path: Path, file_id: str) -> None:
    for tab_index, tab in enumerate(chat_data.get("tabs") or []):
        if not isinstance(tab, dict):
            continue
        raw_tab_id = tab.get("tabId") or f"{path}:aichat-tab:{tab_index}"
        thread_id = anonymous_id(f"cursor:{raw_tab_id}")
        fallback_timestamp = _iso_timestamp(tab.get("lastSendTime"))
        current_model = None
        for bubble_index, bubble in enumerate(tab.get("bubbles") or []):
            if not isinstance(bubble, dict):
                bubble = {}
            bubble_type = bubble.get("type")
            kind = {"user": "user_prompt", "ai": "assistant_message"}.get(bubble_type, "other")
            model = _model_name(bubble) or current_model
            current_model = model
            bubble_id = bubble.get("id") or bubble.get("bubbleId") or f"{raw_tab_id}:{bubble_index}"
            event = _base_event(thread_id, file_id, len(events))
            event.update(
                {
                    "timestamp": fallback_timestamp,
                    "event_id": anonymous_id_or_none("cursor-event", bubble_id),
                    "event_kind": kind,
                    "raw_event_type": f"aichat.bubble.{bubble_type}",
                    "is_run_start": kind == "user_prompt",
                    "model": model,
                    "provider": _provider_for_model(model),
                    "message_id": anonymous_id_or_none("cursor-message", bubble_id),
                    "text_length": serialized_length(bubble.get("text") or None),
                }
            )
            events.append(event)


def convert_thread(path: Path) -> list[dict]:
    kv = _read_json_kv(path, "cursorDiskKV", ("composerData:%", "bubbleId:%"))
    items = _read_json_kv(path, "ItemTable", ("composer.composerData", "%aichat%chatdata%"))
    if not kv and not items:
        return []

    file_id = anonymous_id(f"cursor-file:{path}")
    events: list[dict] = []

    for key in sorted(kv):
        if key.startswith("composerData:") and isinstance(kv[key], dict):
            _convert_composer(events, kv[key], kv, path, file_id)

    for key in sorted(items):
        record = items[key]
        if not isinstance(record, dict):
            continue
        if key == "composer.composerData":
            # Newer versions store only conversation-less heads here (full data
            # lives in globalStorage); older versions embedded conversations.
            for composer in record.get("allComposers") or []:
                if isinstance(composer, dict) and (
                    composer.get("conversation") or composer.get("conversationMap")
                ):
                    _convert_composer(events, composer, kv, path, file_id)
        elif "aichat" in key and "chatdata" in key:
            _convert_legacy_chat(events, record, path, file_id)

    return events


def _mark_canonical(event: dict) -> None:
    served = event.get("input_tokens") or 0
    output = event.get("output_tokens") or 0
    if not (served > 0 or output > 0):
        event["usage_dedup_reason"] = "zero_usage"
        return
    event["usage_canonical"] = True
    event["usage_source"] = "bubble_token_count"
    event["call_served_input_tokens"] = served
    # Cursor reports a single per-bubble input total with no cache breakdown.
    event["call_uncached_input_tokens"] = served
    event["call_cached_input_tokens"] = 0
    event["call_cache_creation_input_tokens"] = 0
    event["call_cache_creation_5m_input_tokens"] = 0
    event["call_cache_creation_1h_input_tokens"] = 0
    event["call_output_tokens"] = output
    event["call_reasoning_output_tokens"] = None
    event["call_reasoning_tokens_available"] = False


def finalize_events(events: list[dict]) -> list[dict]:
    """Mark canonical model-call usage rows across all converted sessions.

    Cursor stores one tokenCount per bubble under a stable bubbleId, so the
    same call only repeats when a conversation is present in more than one
    store (e.g. workspace migration copies). Deduping by hashed bubble id
    keeps one canonical copy per bubble; duplicates stay marked.
    """
    init_usage_fields(events)
    usage_events = sorted(
        (
            event
            for event in events
            if event.get("input_tokens") is not None or event.get("output_tokens") is not None
        ),
        key=event_sort_key,
    )

    grouped: dict[str, list[dict]] = {}
    ungrouped = []
    for event in usage_events:
        identity = event.get("message_id")
        if identity:
            grouped.setdefault(identity, []).append(event)
        else:
            ungrouped.append(event)

    for group in grouped.values():
        best = max(
            group,
            key=lambda event: (event.get("input_tokens") or 0) + (event.get("output_tokens") or 0),
        )
        for event in group:
            if event is not best:
                event["usage_dedup_reason"] = "duplicate_bubble_id"
        _mark_canonical(best)
    for event in ungrouped:
        _mark_canonical(event)
    return events
