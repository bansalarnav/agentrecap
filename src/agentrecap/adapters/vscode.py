"""Adapter for VS Code chat sessions (GitHub Copilot Chat).

VS Code persists each chat as a single JSON document (``version: 3`` observed)
holding a ``requests`` array. Every request bundles the user prompt, the
serialized response parts (markdown, thinking, tool invocations, references),
and a ``result`` with timings and metadata. Sessions live under
``workspaceStorage/<hash>/chatSessions/*.json`` plus
``globalStorage/emptyWindowChatSessions/*.json`` for chats opened without a
workspace. ``chatEditingSessions`` state files are working-set snapshots, not
transcripts, and are ignored.

The session files record no token usage, so ``finalize_events`` only
initializes the usage columns and no canonical model-call rows are emitted.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .common import (
    anonymous_id,
    anonymous_id_or_none,
    base_event,
    init_usage_fields,
    serialized_length,
)

SOURCE = "vscode"
# All requests are served through the user's GitHub Copilot subscription, and
# models.dev lists the Copilot-served model catalog (multiple families) under
# the "github-copilot" provider, so that is the pricing provider regardless of
# which vendor's model handled a given request.
PROVIDER = "github-copilot"
DISPLAY_NAME = "VS Code Chat"

if sys.platform == "darwin":
    DEFAULT_INPUT = Path.home() / "Library" / "Application Support" / "Code" / "User"
elif sys.platform == "win32":
    DEFAULT_INPUT = Path.home() / "AppData" / "Roaming" / "Code" / "User"
else:
    DEFAULT_INPUT = Path.home() / ".config" / "Code" / "User"

INPUT_HELP = "VS Code user data directory containing workspaceStorage chat sessions"

RUN_END_TYPE = "request.result"

# Response part kinds that carry tool activity. ``toolInvocationSerialized`` is
# the persisted form; ``toolInvocation`` is the live in-progress form that can
# appear if a window closed mid-request.
TOOL_INVOCATION_KINDS = {"toolInvocationSerialized", "toolInvocation"}


def discover_sessions(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    workspace_storage = path / "workspaceStorage"
    empty_window = path / "globalStorage" / "emptyWindowChatSessions"
    if workspace_storage.is_dir() or empty_window.is_dir():
        paths = sorted(workspace_storage.glob("*/chatSessions/*.json"))
        paths.extend(sorted(empty_window.glob("*.json")))
        return paths
    # Fallback for non-standard layouts: any chat session JSON under the path.
    return sorted(path.rglob("*.json"))


def _iso_timestamp(epoch_ms: object) -> str | None:
    if not isinstance(epoch_ms, (int, float)) or isinstance(epoch_ms, bool):
        return None
    try:
        return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _model_for_request(request: dict) -> str | None:
    """Best-effort models.dev-style model id for one request.

    ``modelId`` looks like ``copilot/gpt-4o`` (vendor prefix + model id). With
    ``copilot/auto`` the resolved model only appears as a display name in
    ``result.details`` (e.g. "GPT-5 mini"), which we slugify.
    """
    model_id = request.get("modelId")
    model = None
    if isinstance(model_id, str) and model_id:
        model = model_id.rsplit("/", 1)[-1]
    if model in (None, "auto"):
        details = (request.get("result") or {}).get("details")
        if isinstance(details, str) and 0 < len(details) <= 40:
            model = details.strip().lower().replace(" ", "-")
    return model


def _base_event(context: dict, event_kind: str, raw_event_type: str) -> dict:
    """One standardized event carrying the request's shared context.

    ``file_event_index`` stays None here; convert_thread assigns the final
    ordering once every event for the session has been emitted.
    """
    return base_event(
        SOURCE,
        PROVIDER,
        context["thread_id"],
        context["file_id"],
        None,
        timestamp=context["timestamp"],
        event_kind=event_kind,
        raw_event_type=raw_event_type,
        model=context["model"],
        request_id=context["request_id"],
    )


def _response_part_events(context: dict, part: object) -> list[dict]:
    if not isinstance(part, dict):
        event = _base_event(context, "other", "response.unknown")
        event["text_length"] = serialized_length(part)
        return [event]

    kind = part.get("kind")
    if kind is None and "value" in part:
        # Serialized markdown chunks carry no kind, only the rendered value.
        event = _base_event(context, "assistant_message", "response.markdown")
        event["text_length"] = serialized_length(part.get("value"))
        return [event]

    if kind == "thinking":
        event = _base_event(context, "reasoning", "response.thinking")
        event["text_length"] = serialized_length(part.get("value"))
        return [event]

    if kind in TOOL_INVOCATION_KINDS:
        tool_call_id = anonymous_id_or_none("vscode-tool", part.get("toolCallId"))
        tool_name = part.get("toolId") or part.get("toolName")
        is_complete = bool(part.get("isComplete"))
        result_details = part.get("resultDetails")
        is_error = part.get("isError")
        if is_error is None and isinstance(result_details, dict):
            is_error = result_details.get("isError")
        tool_success = (not is_error) if is_complete else None

        call = _base_event(context, "tool_call", f"response.{kind}")
        call["tool_call_id"] = tool_call_id
        call["tool_name"] = tool_name
        call["tool_success"] = tool_success
        call["tool_input_length"] = serialized_length(part.get("toolSpecificData"))
        events = [call]
        if is_complete:
            # The serialized invocation also records the completed outcome;
            # surface it as the paired tool_result event.
            result = _base_event(context, "tool_result", f"response.{kind}.result")
            result["tool_call_id"] = tool_call_id
            result["tool_name"] = tool_name
            result["tool_success"] = tool_success
            result["tool_output_length"] = serialized_length(result_details)
            events.append(result)
        return events

    event = _base_event(context, "other", f"response.{kind}")
    if "value" in part:
        event["text_length"] = serialized_length(part.get("value"))
    return [event]


def convert_thread(path: Path) -> list[dict]:
    try:
        with path.open(encoding="utf-8", errors="replace") as file:
            session = json.load(file)
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(session, dict):
        return []

    raw_thread_id = str(session.get("sessionId") or path.stem)
    thread_id = anonymous_id(f"vscode:{raw_thread_id}")
    file_id = anonymous_id(f"vscode-file:{path}")
    events: list[dict] = []

    requests = session.get("requests")
    if not isinstance(requests, list):
        requests = []

    for request_index, request in enumerate(requests):
        if not isinstance(request, dict):
            continue
        raw_request_id = request.get("requestId") or f"{raw_thread_id}:request:{request_index}"
        result = request.get("result") if isinstance(request.get("result"), dict) else {}
        timings = result.get("timings") if isinstance(result.get("timings"), dict) else {}
        timestamp = _iso_timestamp(request.get("timestamp"))
        context = {
            "thread_id": thread_id,
            "file_id": file_id,
            "timestamp": timestamp,
            "model": _model_for_request(request),
            "request_id": anonymous_id_or_none("vscode-request", raw_request_id),
        }

        message = request.get("message") if isinstance(request.get("message"), dict) else {}
        prompt = _base_event(context, "user_prompt", "request")
        prompt["event_id"] = anonymous_id_or_none("vscode-event", raw_request_id)
        prompt["message_id"] = anonymous_id_or_none("vscode-message", raw_request_id)
        prompt["is_run_start"] = True
        prompt["text_length"] = serialized_length(message.get("text"))
        events.append(prompt)

        response = request.get("response")
        for part in response if isinstance(response, list) else []:
            events.extend(_response_part_events(context, part))

        # A request can be persisted while still running. Only emit an
        # explicit terminal event when VS Code saved a result or cancellation;
        # otherwise downstream analysis should treat the run as incomplete.
        if isinstance(request.get("result"), dict) or request.get("isCanceled"):
            total_elapsed = timings.get("totalElapsed")
            run_end_context = dict(context)
            if (
                isinstance(request.get("timestamp"), (int, float))
                and not isinstance(request.get("timestamp"), bool)
                and isinstance(total_elapsed, (int, float))
                and not isinstance(total_elapsed, bool)
            ):
                run_end_context["timestamp"] = (
                    _iso_timestamp(request["timestamp"] + total_elapsed) or timestamp
                )
            run_end = _base_event(run_end_context, "run_end", RUN_END_TYPE)
            run_end["event_id"] = anonymous_id_or_none(
                "vscode-event", f"{raw_request_id}:result"
            )
            run_end["message_id"] = anonymous_id_or_none(
                "vscode-message", request.get("responseId")
            )
            if request.get("isCanceled"):
                run_end["run_end_status"] = "aborted"
            elif result.get("errorDetails"):
                run_end["run_end_status"] = "error"
            else:
                run_end["run_end_status"] = "completed"
            run_end["duration_ms"] = total_elapsed
            run_end["time_to_first_token_ms"] = timings.get("firstProgress")
            events.append(run_end)

    for index, event in enumerate(events):
        event["file_event_index"] = index
    return events


def finalize_events(events: list[dict]) -> list[dict]:
    """VS Code chat sessions record no token usage; only initialize the columns."""
    init_usage_fields(events)
    return events
