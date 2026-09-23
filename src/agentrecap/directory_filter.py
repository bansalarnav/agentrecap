"""Find sessions whose working directory or tool inputs refer to a directory.

Raw paths are inspected only while selecting sessions. They are never added to
the metadata-only event export.
"""

import ast
import json
import re
import shlex
import sqlite3
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from .adapters.common import anonymous_id, read_jsonl_records
from .adapters import opencode


PATH_KEYS = {
    "path", "paths", "file", "files", "file_path", "filePath", "filename",
    "directory", "cwd", "workdir", "working_directory", "workingDirectory",
    "root", "uri", "target", "source", "relative_path", "relativePath",
}
COMMAND_KEYS = {"cmd", "command", "script", "patch", "patchText"}
PATCH_PATH = re.compile(r"^\*\*\* (?:Add File|Update File|Delete File|Move to): (.+)$", re.MULTILINE)
JS_ARGUMENT = re.compile(
    r"\b[\"']?(cmd|command|workdir|cwd)[\"']?\s*:\s*(\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')"
)


def _within(value: object, directory: Path, cwd: Path | None = None) -> bool:
    if not isinstance(value, str) or not value or "\n" in value or "\0" in value:
        return False
    if value.startswith("file://"):
        value = unquote(urlsplit(value).path)
    try:
        path = Path(value).expanduser()
        if not path.is_absolute():
            if cwd is None:
                return False
            path = cwd / path
        path.resolve().relative_to(directory)
        return True
    except (OSError, ValueError, RuntimeError):
        return False


def _command_touches(command: object, directory: Path, cwd: Path | None) -> bool:
    if not isinstance(command, str):
        return False
    for path in PATCH_PATH.findall(command):
        if _within(path.strip(), directory, cwd):
            return True
    # Codex can wrap exec_command calls inside a functions.exec JavaScript
    # snippet. Inspect its string arguments without executing that snippet.
    js_args: dict[str, list[str]] = {}
    for key, literal in JS_ARGUMENT.findall(command):
        try:
            js_args.setdefault(key, []).append(ast.literal_eval(literal))
        except (SyntaxError, ValueError):
            continue
    if js_args:
        workdirs = [cwd]
        for js_cwd in js_args.get("workdir", []) + js_args.get("cwd", []):
            if _within(js_cwd, directory, cwd):
                return True
            candidate = Path(js_cwd).expanduser()
            workdirs.append(candidate if candidate.is_absolute() else cwd / candidate if cwd else None)
        for nested in js_args.get("cmd", []) + js_args.get("command", []):
            if nested != command and any(
                _command_touches(nested, directory, workdir) for workdir in workdirs
            ):
                return True
    try:
        tokens = shlex.split(command, comments=False)
    except ValueError:
        tokens = command.split()
    for token in tokens:
        # Shell assignments and redirections can carry paths too.
        candidate = token.rsplit("=", 1)[-1].strip(";,(){}<>")
        if _within(candidate, directory, cwd):
            return True
    return False


def _tool_input_touches(value: object, directory: Path, cwd: Path | None) -> bool:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return _command_touches(value, directory, cwd)
        return _tool_input_touches(parsed, directory, cwd)
    if isinstance(value, list):
        return any(_tool_input_touches(item, directory, cwd) for item in value)
    if not isinstance(value, dict):
        return False

    local_cwd = cwd
    for key in ("workdir", "cwd", "working_directory", "workingDirectory", "directory"):
        raw = value.get(key)
        if isinstance(raw, str):
            candidate = Path(raw).expanduser()
            local_cwd = (candidate if candidate.is_absolute() else (cwd / candidate if cwd else None))
            if local_cwd is not None:
                local_cwd = local_cwd.resolve()
            if _within(raw, directory, cwd):
                return True

    for key, item in value.items():
        if key in COMMAND_KEYS and _command_touches(item, directory, local_cwd):
            return True
        if key in PATH_KEYS:
            if isinstance(item, str) and _within(item, directory, local_cwd):
                return True
            if isinstance(item, list) and any(_within(part, directory, local_cwd) for part in item):
                return True
        if isinstance(item, (dict, list)) and _tool_input_touches(item, directory, local_cwd):
            return True
    return False


def _jsonl_matches(source: str, path: Path, directory: Path) -> set[str]:
    records = read_jsonl_records(path)
    if not records:
        return set()
    if source == "codex":
        meta = next((r.get("payload") for r in records if r.get("type") == "session_meta"), {}) or {}
        raw_id = meta.get("id") or meta.get("session_id") or path.name
        cwd = meta.get("cwd")
    elif source == "claude":
        raw_id = next((r.get("sessionId") for r in records if r.get("sessionId")), path.name)
        cwd = next((r.get("cwd") for r in records if r.get("cwd")), None)
    else:
        header = next((r for r in records if r.get("type") == "session"), {})
        raw_id = header.get("id") or path.stem
        cwd = header.get("cwd")

    thread_id = anonymous_id(f"{source}:{raw_id}")
    working_dir = Path(cwd).expanduser() if isinstance(cwd, str) and cwd else None
    working_dir = working_dir.resolve() if working_dir is not None and working_dir.is_absolute() else None
    if working_dir and _within(str(working_dir), directory):
        return {thread_id}

    for record in records:
        payload = record.get("payload") or {}
        record_cwd = record.get("cwd") or (payload.get("cwd") if isinstance(payload, dict) else None)
        if isinstance(record_cwd, str) and record_cwd:
            candidate = Path(record_cwd).expanduser()
            working_dir = (candidate if candidate.is_absolute() else (working_dir / candidate if working_dir else None))
            if working_dir:
                working_dir = working_dir.resolve()
                if _within(str(working_dir), directory):
                    return {thread_id}

        tool_inputs = []
        if source == "codex" and record.get("type") == "response_item":
            if isinstance(payload, dict) and payload.get("type") in {"function_call", "custom_tool_call"}:
                tool_inputs.append(payload.get("arguments") or payload.get("input"))
        elif source == "claude":
            message = record.get("message") or {}
            for block in message.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_inputs.append(block.get("input"))
        elif source in {"pi", "omp"}:
            message = record.get("message") or {}
            for block in message.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "toolCall":
                    tool_inputs.append(block.get("arguments"))

        if any(_tool_input_touches(item, directory, working_dir) for item in tool_inputs):
            return {thread_id}
    return set()


def _opencode_matches(path: Path, directory: Path) -> set[str]:
    if path.name != "opencode.db":
        session = opencode._read_json(path)
        storage = opencode._storage_dir(path)
        if not session or storage is None:
            return set()
        raw_id = session.get("id") or path.stem
        cwd = session.get("directory")
        if _within(cwd, directory):
            return {anonymous_id(f"opencode:{raw_id}")}
        part_root = storage / "part"
        for message_path in (storage / "message" / str(raw_id)).glob("*.json"):
            message = opencode._read_json(message_path) or {}
            for part_path in (part_root / str(message.get("id"))).glob("*.json"):
                part = opencode._read_json(part_path) or {}
                if part.get("type") == "tool" and _tool_input_touches(
                    (part.get("state") or {}).get("input"), directory,
                    Path(cwd).resolve() if isinstance(cwd, str) and Path(cwd).is_absolute() else None,
                ):
                    return {anonymous_id(f"opencode:{raw_id}")}
        return set()

    try:
        connection = sqlite3.connect(f"file:{quote(str(path), safe='/:')}?mode=ro", uri=True)
    except sqlite3.Error:
        return set()
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(session)")}
        select = "SELECT id, directory FROM session" if "directory" in columns else "SELECT id, NULL FROM session"
        cwd_by_id = dict(connection.execute(select))
        matched = {sid for sid, cwd in cwd_by_id.items() if _within(cwd, directory)}
        for sid, data in connection.execute("SELECT session_id, data FROM part"):
            if sid in matched or sid not in cwd_by_id:
                continue
            try:
                part = json.loads(data)
            except (TypeError, ValueError):
                continue
            if isinstance(part, dict) and part.get("type") == "tool" and _tool_input_touches(
                (part.get("state") or {}).get("input"), directory,
                Path(cwd_by_id[sid]).resolve()
                if cwd_by_id[sid] and Path(cwd_by_id[sid]).is_absolute() else None,
            ):
                matched.add(sid)
        return {anonymous_id(f"opencode:{sid}") for sid in matched}
    except sqlite3.Error:
        return set()
    finally:
        connection.close()


def matching_thread_ids(source: str, path: Path, directory: Path) -> set[str]:
    """Return session ids selected by --dir for one discovered input file."""
    if source == "opencode":
        return _opencode_matches(path, directory)
    return _jsonl_matches(source, path, directory)
