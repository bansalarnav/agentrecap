"""Helpers shared by all session adapters."""

import hashlib
import json
from pathlib import Path


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


def speed_status(speed: object, service_tier: object) -> str:
    values = {str(value).lower() for value in (speed, service_tier) if value is not None}
    if values & {"fast", "priority"}:
        return "fast"
    if values & {"standard", "default"}:
        return "standard"
    return "unknown"


def read_jsonl_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
