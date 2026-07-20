"""Adapter for OpenCode session storage."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from .common import (
    anonymous_id,
    anonymous_id_or_none,
    base_event,
    event_sort_key,
    init_usage_fields,
    mark_canonical_usage,
    serialized_length,
)

SOURCE = "opencode"
PROVIDER = "opencode"
DISPLAY_NAME = "OpenCode"
GRAPH_COLOR = "black"
DEFAULT_INPUT = Path.home() / ".local" / "share" / "opencode"
INPUT_HELP = (
    "OpenCode data directory containing opencode.db or the legacy storage directory"
)

# OpenCode provider IDs normally match models.dev. These aliases cover older
# and commonly configured names that do not.
PROVIDER_ALIASES = {
    "bedrock": "amazon-bedrock",
    "github": "github-models",
    "vertex": "google-vertex",
    "vertexai": "google-vertex",
}


def discover_sessions(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []

    database = path / "opencode.db"
    if database.is_file():
        return [database]

    candidates = (path / "storage" / "session", path / "session", path)
    session_dir = next((candidate for candidate in candidates if candidate.is_dir()), None)
    if session_dir is None:
        return []
    return sorted((*session_dir.glob("*.json"), *session_dir.glob("*/*.json")))


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _storage_dir(session_path: Path) -> Path | None:
    for parent in session_path.parents:
        if parent.name == "session" and parent.parent.name == "storage":
            return parent.parent
    return None


def _timestamp(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        seconds = value / 1000 if value > 10_000_000_000 else value
        try:
            return (
                datetime.fromtimestamp(seconds, tz=timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z")
            )
        except (OSError, OverflowError, ValueError):
            return None
    return str(value)


def _provider(value: object) -> str:
    provider = str(value) if value else PROVIDER
    return PROVIDER_ALIASES.get(provider, provider)


def _convert_json_thread(path: Path) -> list[dict]:
    session = _read_json(path)
    storage_dir = _storage_dir(path)
    if session is None or storage_dir is None:
        return []

    raw_thread_id = str(session.get("id") or path.stem)
    message_dir = storage_dir / "message" / raw_thread_id
    messages = []
    if message_dir.is_dir():
        for message_path in message_dir.glob("*.json"):
            message = _read_json(message_path)
            if message is not None:
                messages.append(message)

    parts_by_message = {}
    for message in messages:
        raw_message_id = message.get("id")
        part_dir = storage_dir / "part" / str(raw_message_id)
        parts = []
        if part_dir.is_dir():
            for part_path in part_dir.glob("*.json"):
                part = _read_json(part_path)
                if part is not None:
                    parts.append(part)
        parts_by_message[str(raw_message_id)] = parts

    return _convert_session(session, messages, parts_by_message, str(path))


def _convert_database(path: Path) -> list[dict]:
    uri = f"file:{quote(str(path), safe='/:')}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error:
        return []
    try:
        session_rows = connection.execute(
            "SELECT id, parent_id, time_created, time_updated FROM session"
        ).fetchall()
        message_rows = connection.execute(
            "SELECT id, session_id, time_created, data FROM message"
        ).fetchall()
        part_rows = connection.execute(
            "SELECT id, message_id, session_id, time_created, data FROM part"
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        connection.close()

    sessions = {
        session_id: {
            "id": session_id,
            "parentID": parent_id,
            "time": {"created": created, "updated": updated},
        }
        for session_id, parent_id, created, updated in session_rows
    }
    messages_by_session: dict[str, list[dict]] = {}
    for message_id, session_id, created, data in message_rows:
        try:
            message = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(message, dict):
            continue
        message["id"] = message_id
        message.setdefault("sessionID", session_id)
        message.setdefault("time", {"created": created})
        messages_by_session.setdefault(session_id, []).append(message)

    parts_by_session: dict[str, dict[str, list[dict]]] = {}
    for part_id, message_id, session_id, created, data in part_rows:
        try:
            part = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(part, dict):
            continue
        part["id"] = part_id
        part.setdefault("messageID", message_id)
        part["_storage_time_created"] = created
        parts_by_session.setdefault(session_id, {}).setdefault(message_id, []).append(part)

    events = []
    for session_id, session in sessions.items():
        events.extend(
            _convert_session(
                session,
                messages_by_session.get(session_id, []),
                parts_by_session.get(session_id, {}),
                f"{path}:{session_id}",
            )
        )
    return events


def _convert_session(
    session: dict,
    messages: list[dict],
    parts_by_message: dict[str, list[dict]],
    file_identity: str,
) -> list[dict]:
    raw_thread_id = str(session.get("id") or file_identity)
    thread_id = anonymous_id(f"opencode:{raw_thread_id}")
    file_id = anonymous_id(f"opencode-file:{file_identity}")
    raw_parent_thread_id = session.get("parentID")
    parent_thread_id = anonymous_id_or_none("opencode", raw_parent_thread_id)
    is_sidechain = raw_parent_thread_id is not None

    messages.sort(
        key=lambda message: (
            (message.get("time") or {}).get("created") or 0,
            message.get("id") or "",
        )
    )

    assistants_by_parent = {
        message.get("parentID"): message
        for message in messages
        if message.get("role") == "assistant" and message.get("parentID")
    }
    events = []

    def add_event(**values: object) -> None:
        provider = values.pop("provider", PROVIDER)
        values.setdefault("agent_id", thread_id if is_sidechain else None)
        values.setdefault("is_sidechain", is_sidechain)
        values.setdefault("parent_thread_id", parent_thread_id)
        events.append(
            base_event(SOURCE, provider, thread_id, file_id, len(events), **values)
        )

    session_time = session.get("time") or {}
    add_event(
        timestamp=_timestamp(session_time.get("created")),
        event_id=anonymous_id_or_none("opencode-event", raw_thread_id),
        raw_event_type="session",
    )

    current_run_start = None
    for message in messages:
        raw_message_id = message.get("id")
        role = message.get("role")
        message_time = message.get("time") or {}
        created = message_time.get("created")
        completed = message_time.get("completed")
        model_info = message.get("model") or {}
        provider_id = message.get("providerID") or model_info.get("providerID")
        model = message.get("modelID") or model_info.get("modelID")
        if role == "user" and (provider_id is None or model is None):
            child = assistants_by_parent.get(raw_message_id) or {}
            provider_id = provider_id or child.get("providerID")
            model = model or child.get("modelID")

        if role == "user":
            current_run_start = created
        message_event_kind = "user_prompt" if role == "user" else "other"
        run_end_status = None
        duration_ms = None
        message_timestamp = created
        if role == "assistant" and completed:
            message_event_kind = "run_end"
            message_timestamp = completed
            run_end_status = "aborted" if message.get("error") else "completed"
            if current_run_start is not None:
                duration_ms = max(completed - current_run_start, 0)

        message_id = anonymous_id_or_none("opencode-message", raw_message_id)
        add_event(
            provider=_provider(provider_id),
            timestamp=_timestamp(message_timestamp),
            event_id=anonymous_id_or_none("opencode-event", raw_message_id),
            parent_event_id=anonymous_id_or_none("opencode-event", message.get("parentID")),
            event_kind=message_event_kind,
            raw_event_type=f"message.{role or 'unknown'}",
            is_run_start=role == "user",
            run_end_status=run_end_status,
            duration_ms=duration_ms,
            model=model,
            message_id=message_id,
        )

        parts = parts_by_message.get(str(raw_message_id), [])
        parts.sort(
            key=lambda part: (
                part.get("_storage_time_created") or 0,
                part.get("id") or "",
            )
        )

        for part in parts:
            part_type = part.get("type") or "unknown"
            state = part.get("state") or {}
            part_time = part.get("time") or state.get("time") or {}
            part_timestamp = part_time.get("start")
            if not part_timestamp:
                part_timestamp = completed if part_type in {"step-finish", "patch"} else created

            event_kind = "other"
            if role == "assistant" and part_type == "text":
                event_kind = "assistant_message"
            elif role == "assistant" and part_type == "reasoning":
                event_kind = "reasoning"
            elif part_type == "tool":
                event_kind = "tool_call"

            tool_success = None
            if state.get("status") == "completed":
                tool_success = True
            elif state.get("status") == "error":
                tool_success = False

            tokens = part.get("tokens") or {}
            cache = tokens.get("cache") or {}
            has_usage = part_type == "step-finish"
            input_tokens = tokens.get("input") if has_usage else None
            output_tokens = tokens.get("output") if has_usage else None
            cached_tokens = cache.get("read") if has_usage else None
            cache_creation_tokens = cache.get("write") if has_usage else None
            reasoning_tokens = tokens.get("reasoning") if has_usage else None
            total_tokens = tokens.get("total") if has_usage else None
            if total_tokens is not None:
                # Newer OpenCode records report visible output and reasoning
                # separately. Older records omit total and include reasoning in
                # output already.
                output_tokens = (output_tokens or 0) + (reasoning_tokens or 0)
            elif has_usage:
                total_tokens = sum(
                    value or 0
                    for value in (
                        input_tokens,
                        output_tokens,
                        cached_tokens,
                        cache_creation_tokens,
                    )
                )

            text = part.get("text") if part_type in {"text", "reasoning"} else None
            add_event(
                provider=_provider(provider_id),
                timestamp=_timestamp(part_timestamp),
                event_id=anonymous_id_or_none("opencode-event", part.get("id")),
                parent_event_id=anonymous_id_or_none("opencode-event", raw_message_id),
                event_kind=event_kind,
                raw_event_type=f"{role or 'unknown'}.{part_type}",
                duration_ms=(
                    max(part_time["end"] - part_time["start"], 0)
                    if part_time.get("start") and part_time.get("end")
                    else None
                ),
                model=model,
                message_id=message_id,
                tool_call_id=anonymous_id_or_none("opencode-tool", part.get("callID")),
                tool_name=part.get("tool") if part_type == "tool" else None,
                tool_success=tool_success,
                usage_kind="model_call" if has_usage else None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_tokens,
                cache_creation_input_tokens=cache_creation_tokens,
                reasoning_output_tokens=reasoning_tokens,
                total_tokens=total_tokens,
                reported_cost_usd=part.get("cost") if has_usage else None,
                text_length=serialized_length(text),
                tool_input_length=(
                    serialized_length(state.get("input")) if part_type == "tool" else None
                ),
                tool_output_length=(
                    serialized_length(state.get("output")) if part_type == "tool" else None
                ),
            )

    return events


def convert_thread(path: Path) -> list[dict]:
    if path.name == "opencode.db":
        return _convert_database(path)
    return _convert_json_thread(path)


def finalize_events(events: list[dict]) -> list[dict]:
    """Mark each distinct step-finish part as one canonical model call."""
    init_usage_fields(events)
    seen: set[str] = set()
    usage_events = sorted(
        (event for event in events if event.get("usage_kind") == "model_call"),
        key=event_sort_key,
    )
    for event in usage_events:
        event_id = event.get("event_id")
        if event_id and event_id in seen:
            event["usage_dedup_reason"] = "duplicate_step_id"
            continue
        if event_id:
            seen.add(event_id)

        uncached = event.get("input_tokens") or 0
        cached = event.get("cached_input_tokens") or 0
        creation = event.get("cache_creation_input_tokens") or 0
        mark_canonical_usage(
            event,
            "step_finish_usage",
            served_input_tokens=uncached + cached + creation,
            cached_input_tokens=cached,
            cache_creation_input_tokens=creation,
            output_tokens=event.get("output_tokens") or 0,
            reasoning_output_tokens=event.get("reasoning_output_tokens"),
        )
    return events
