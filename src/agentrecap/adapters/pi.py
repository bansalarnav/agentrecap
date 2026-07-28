"""Adapter for pi coding-agent session logs.

pi stores one JSONL file per session under ``~/.pi/agent/sessions/``, in a
directory named after the working directory (``/`` replaced by ``-``). The
first line is a ``session`` header; every later line is a tree entry carrying
``id``/``parentId``, so branching happens in place rather than by starting a
new file. Legacy v1 sessions have no ``version`` and no entry ids; their
header holds the initial ``provider``/``modelId``/``thinkingLevel`` instead.

Model calls live on ``message`` entries whose message has ``role:
"assistant"``: each one records ``provider``, ``model``, ``stopReason`` and a
``usage`` object whose ``input`` counts only uncached prompt tokens, with
``cacheRead``/``cacheWrite`` reported alongside it (``totalTokens`` is their
sum plus ``output``). ``output`` already includes ``reasoning``. Summarizing
entries (``compaction``, ``branch_summary``) carry their own ``usage`` for the
extra call that produced the summary; that call uses whatever model the
session was on. Tool results can carry ``usage`` too, for LLM work an
extension's tool did on its own, and those report no model of their own.

``/fork``, ``/clone`` and ``/tree``'s branch extraction copy the source
entries verbatim into a new file, keeping their ids and timestamps, so the
same API call appears in several sessions; finalize_events dedups on that.
"""

import json
from functools import lru_cache
from pathlib import Path

from .common import (
    anonymous_id,
    anonymous_id_or_none,
    base_event,
    diff_line_counts,
    event_sort_key,
    init_usage_fields,
    line_count,
    mark_canonical_usage,
    read_jsonl_records,
    serialized_length,
)

SOURCE = "pi"
# Fallback only: pi is multi-provider and every assistant message names the
# provider that served it.
PROVIDER = "pi"
DISPLAY_NAME = "pi"
GRAPH_COLOR = "tab:purple"
DEFAULT_INPUT = Path.home() / ".pi" / "agent"
INPUT_HELP = "pi agent directory containing sessions/ (PI_CODING_AGENT_DIR overrides it)"

# pi's model catalog is generated from models.dev, so its model ids already
# match. These are the provider ids pi renames or splits by region/plan.
PROVIDER_ALIASES = {
    "ant-ling": "bailing",
    "azure-openai-responses": "azure",
    "fireworks": "fireworks-ai",
    "kimi-coding": "kimi-for-coding",
    "openai-codex": "openai",
    "qwen-token-plan": "alibaba-token-plan",
    "qwen-token-plan-cn": "alibaba-token-plan-cn",
    "together": "togetherai",
    "vercel-ai-gateway": "vercel",
    "zai-coding-cn": "zai-coding-plan",
}

# message role -> standardized event kind for the roles that are a single
# event. Assistant messages expand into one event per content block instead.
ROLE_KINDS = {
    "user": "user_prompt",
    "toolResult": "tool_result",
}

CONTENT_BLOCK_KINDS = {
    "text": "assistant_message",
    "thinking": "reasoning",
    "toolCall": "tool_call",
}

# stopReason -> run_end_status. "toolUse" keeps the run open: the agent loop
# continues with the tool results. "pending" only appears on partial streaming
# events, never in persisted entries.
RUN_END_STATUSES = {
    "stop": "completed",
    "length": "completed",
    "aborted": "aborted",
    "error": "error",
}

# usage_kind -> usage_source recorded on canonical rows.
USAGE_SOURCES = {
    "model_call": "assistant_usage",
    "summary_call": "summary_usage",
    "tool_call_usage": "tool_result_usage",
}


def discover_sessions(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    # The agent directory also holds themes, prompts and custom tools, so only
    # descend into sessions/ when it is there.
    for candidate in (path / "sessions", path / "agent" / "sessions"):
        if candidate.is_dir():
            return sorted(candidate.rglob("*.jsonl"))
    return sorted(path.rglob("*.jsonl"))


def _provider(value: object) -> str | None:
    if not value:
        return None
    provider = str(value)
    return PROVIDER_ALIASES.get(provider, provider)


@lru_cache(maxsize=None)
def _session_id_of(path: str) -> str | None:
    """Read just the header of a session file to learn its session id."""
    try:
        with Path(path).open(encoding="utf-8", errors="replace") as file:
            header = json.loads(file.readline())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(header, dict) or header.get("type") != "session":
        return None
    return str(header["id"]) if header.get("id") else None


def _parent_thread_id(parent_session: object) -> str | None:
    """Resolve a header's parentSession path to the parent's thread id."""
    if not isinstance(parent_session, str) or not parent_session:
        return None
    return anonymous_id_or_none("pi", _session_id_of(parent_session))


def _content_text(content: object) -> str | None:
    """Flatten a message's content, which is a string or a block list."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    texts = [
        block["text"]
        for block in content
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    ]
    return "\n".join(texts) if texts else None


def _tool_loc(tool_name: object, arguments: object, details: object) -> tuple[int | None, int | None]:
    """Added/removed LOC for a completed edit or write call.

    ``edit`` results carry a unified ``patch`` on newer sessions and a
    line-numbered display ``diff`` on older ones; both count the same way. When
    neither is present the edit blocks themselves are the only estimate.
    """
    if not isinstance(arguments, dict):
        arguments = {}
    if not isinstance(details, dict):
        details = {}

    if tool_name == "edit":
        patch = details.get("patch") or details.get("diff")
        if patch:
            return diff_line_counts(patch)
        edits = arguments.get("edits")
        if not isinstance(edits, list):
            # Older sessions record a single replacement inline.
            edits = [arguments]
        added = sum(line_count(edit.get("newText")) for edit in edits if isinstance(edit, dict))
        removed = sum(line_count(edit.get("oldText")) for edit in edits if isinstance(edit, dict))
        return added, removed
    if tool_name == "write":
        return line_count(arguments.get("content")), 0
    return None, None


def _usage_values(usage: object) -> dict | None:
    if not isinstance(usage, dict):
        return None
    uncached = usage.get("input") or 0
    cached = usage.get("cacheRead") or 0
    creation = usage.get("cacheWrite") or 0
    output = usage.get("output") or 0
    cost = usage.get("cost")
    return {
        "input_tokens": uncached,
        "output_tokens": output,
        "cached_input_tokens": cached,
        "cache_creation_input_tokens": creation,
        # Only Anthropic splits out the 1h-TTL share of its cache writes.
        "cache_creation_1h_input_tokens": usage.get("cacheWrite1h"),
        # A subset of output, reported only by providers that break it out.
        "reasoning_output_tokens": usage.get("reasoning"),
        "total_tokens": usage.get("totalTokens") or uncached + output + cached + creation,
        "reported_cost_usd": cost.get("total") if isinstance(cost, dict) else None,
    }


def convert_thread(path: Path) -> list[dict]:
    records = read_jsonl_records(path)
    if not records:
        return []

    header = next((record for record in records if record.get("type") == "session"), {})
    raw_thread_id = str(header.get("id") or path.stem)
    thread_id = anonymous_id(f"pi:{raw_thread_id}")
    file_id = anonymous_id(f"pi-file:{path}")
    parent_thread_id = _parent_thread_id(header.get("parentSession"))
    # v1 headers seed the session's starting model and thinking level; later
    # versions record them as their own entries instead.
    provider = _provider(header.get("provider"))
    model = header.get("modelId")
    thinking_level = header.get("thinkingLevel")
    tool_arguments_by_id: dict[str, object] = {}
    events = []

    def add_event(record: dict, **values: object) -> None:
        """One event for a session entry, defaulting to the session's state."""
        values.setdefault("timestamp", record.get("timestamp"))
        values.setdefault("event_id", anonymous_id_or_none("pi-event", record.get("id")))
        values.setdefault(
            "parent_event_id", anonymous_id_or_none("pi-event", record.get("parentId"))
        )
        values.setdefault("raw_event_type", record.get("type", "unknown"))
        values.setdefault("model", model)
        values.setdefault("reasoning_effort", thinking_level)
        events.append(
            base_event(
                SOURCE,
                provider or PROVIDER,
                thread_id,
                file_id,
                len(events),
                parent_thread_id=parent_thread_id,
                **values,
            )
        )

    add_event(
        {"type": "session", "id": raw_thread_id, "timestamp": header.get("timestamp")}
    )

    for record in records:
        record_type = record.get("type", "unknown")
        if record_type == "session":
            continue

        if record_type == "model_change":
            provider = _provider(record.get("provider")) or provider
            model = record.get("modelId") or model
            add_event(record)
            continue

        if record_type == "thinking_level_change":
            thinking_level = record.get("thinkingLevel") or thinking_level
            add_event(record)
            continue

        if record_type in {"compaction", "branch_summary"}:
            # The summary itself costs one model call on the session's model.
            usage = _usage_values(record.get("usage")) or {}
            add_event(
                record,
                usage_kind="summary_call" if usage else None,
                text_length=serialized_length(record.get("summary")),
                **usage,
            )
            continue

        if record_type == "custom_message":
            add_event(record, text_length=serialized_length(_content_text(record.get("content"))))
            continue

        if record_type != "message":
            add_event(record)
            continue

        message = record.get("message") or {}
        role = message.get("role")

        if role == "assistant":
            provider = _provider(message.get("provider")) or provider
            # The requested model id, which is the one pi resolved from its
            # catalog; responseModel can name a provider-side alias instead.
            model = message.get("model") or model
            usage = _usage_values(message.get("usage")) or {}
            blocks = message.get("content")
            if not isinstance(blocks, list) or not blocks:
                blocks = [None]

            for block_index, block in enumerate(blocks):
                block = block if isinstance(block, dict) else {}
                block_type = block.get("type")
                text = block.get("text") if block_type == "text" else block.get("thinking")
                arguments = block.get("arguments") if block_type == "toolCall" else None
                if block_type == "toolCall" and block.get("id"):
                    tool_arguments_by_id[block["id"]] = arguments
                # Usage covers the whole response, so it belongs to one event.
                block_usage = usage if block_index == 0 else {}
                add_event(
                    record,
                    event_kind=CONTENT_BLOCK_KINDS.get(block_type, "other"),
                    raw_event_type=f"assistant.{block_type}" if block_type else "assistant",
                    tool_call_id=anonymous_id_or_none("pi-tool", block.get("id")),
                    tool_name=block.get("name") if block_type == "toolCall" else None,
                    usage_kind="model_call" if block_usage else None,
                    text_length=serialized_length(text),
                    tool_input_length=serialized_length(arguments),
                    **block_usage,
                )

            run_end_status = RUN_END_STATUSES.get(message.get("stopReason"))
            if run_end_status is not None:
                add_event(
                    record,
                    event_kind="run_end",
                    raw_event_type=f"assistant.{message.get('stopReason')}",
                    run_end_status=run_end_status,
                )
            continue

        if role == "toolResult":
            raw_tool_call_id = message.get("toolCallId")
            is_error = bool(message.get("isError"))
            loc_added, loc_removed = (None, None)
            if not is_error:
                loc_added, loc_removed = _tool_loc(
                    message.get("toolName"),
                    tool_arguments_by_id.get(raw_tool_call_id),
                    message.get("details"),
                )
            # Present only when an extension's tool did LLM work of its own,
            # which it reports without naming the model it used.
            usage = _usage_values(message.get("usage")) or {}
            add_event(
                record,
                event_kind="tool_result",
                raw_event_type="message.toolResult",
                model=None if usage else model,
                tool_call_id=anonymous_id_or_none("pi-tool", raw_tool_call_id),
                tool_name=message.get("toolName"),
                tool_success=not is_error,
                usage_kind="tool_call_usage" if usage else None,
                tool_output_length=serialized_length(message.get("content")),
                loc_added=loc_added,
                loc_removed=loc_removed,
                **usage,
            )
            continue

        if role == "bashExecution":
            # A command the user ran themselves from the prompt, not a model
            # tool call, so it is not counted as one.
            add_event(
                record,
                raw_event_type="message.bashExecution",
                tool_output_length=serialized_length(message.get("output")),
            )
            continue

        add_event(
            record,
            event_kind=ROLE_KINDS.get(role, "other"),
            raw_event_type=f"message.{role or 'unknown'}",
            is_run_start=role == "user",
            text_length=serialized_length(
                _content_text(message.get("content") or message.get("summary"))
            ),
        )

    return events


def _fork_depths(events: list[dict]) -> dict[str, int]:
    """How many forks each thread sits behind the session that first ran it."""
    parents = {event["thread_id"]: event["parent_thread_id"] for event in events}
    depths = {}
    for thread_id in parents:
        depth = 0
        # A parent outside the scanned input still counts as one fork; the
        # visited set keeps a corrupted parent cycle from looping forever.
        visited = {thread_id}
        parent = parents[thread_id]
        while parent is not None and parent not in visited:
            visited.add(parent)
            depth += 1
            parent = parents.get(parent)
        depths[thread_id] = depth
    return depths


def finalize_events(events: list[dict]) -> list[dict]:
    """Mark one canonical model call per distinct usage record.

    Forking a session copies its entries verbatim into the new file, ids and
    timestamps included, so every session descended from it reports the same
    calls and the same edits. The copy in the session that originally ran them
    stays canonical; the rest are kept with a usage_dedup_reason, and their
    duplicated LOC is dropped so edits are not counted once per fork.
    """
    depths = _fork_depths(events)
    ordered = sorted(events, key=lambda event: (depths[event["thread_id"]], *event_sort_key(event)))

    seen_edits: set[tuple] = set()
    for event in ordered:
        if event["event_id"] is None or (event["loc_added"] is None and event["loc_removed"] is None):
            continue
        identity = (event["event_id"], event["timestamp"])
        if identity in seen_edits:
            event["loc_added"] = None
            event["loc_removed"] = None
        else:
            seen_edits.add(identity)

    init_usage_fields(events)
    seen: set[tuple] = set()
    for event in ordered:
        if not event["usage_kind"]:
            continue
        fingerprint = (
            # v1 sessions predate entry ids; the timestamp and token counts
            # still identify a copied call on their own.
            event["event_id"],
            event["timestamp"],
            event["model"],
            event["input_tokens"],
            event["output_tokens"],
            event["cached_input_tokens"],
            event["cache_creation_input_tokens"],
        )
        if fingerprint in seen:
            event["usage_dedup_reason"] = "duplicate_copied_entry"
            continue
        seen.add(fingerprint)

        uncached = event.get("input_tokens") or 0
        cached = event.get("cached_input_tokens") or 0
        creation = event.get("cache_creation_input_tokens") or 0
        mark_canonical_usage(
            event,
            USAGE_SOURCES[event["usage_kind"]],
            served_input_tokens=uncached + cached + creation,
            cached_input_tokens=cached,
            cache_creation_input_tokens=creation,
            cache_creation_1h_input_tokens=event.get("cache_creation_1h_input_tokens") or 0,
            output_tokens=event.get("output_tokens") or 0,
            reasoning_output_tokens=event.get("reasoning_output_tokens"),
        )
    return events
