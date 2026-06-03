"""Shared recent-queries store: a small JSON file at the repo root, holding the most
recent agent exchanges (question + rendered result spec) so the home gallery can show
the SAME examples to every visitor and re-render them with no engine/agent call."""
from __future__ import annotations
import json
import os
import threading
import datetime
import uuid

_STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recent_chats.json")
_CAP = 60
_LOCK = threading.Lock()


def _load() -> list:
    try:
        with open(_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save(items: list) -> None:
    tmp = _STORE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(items, f)
    os.replace(tmp, _STORE_PATH)  # atomic on the same filesystem


def list_recent() -> list:
    with _LOCK:
        return _load()


def add_recent(item: dict) -> dict:
    """Append one example (newest first), dedup by (question, kind), cap at 60."""
    # Drop keys explicitly passed as None so the id/ts defaults below still apply
    # (a caller that forwards {"ts": None} must not defeat the auto-timestamp).
    entry = {k: v for k, v in item.items() if v is not None}
    if not entry.get("id"):
        entry["id"] = uuid.uuid4().hex[:12]
    if not entry.get("ts"):
        entry["ts"] = datetime.datetime.utcnow().isoformat() + "Z"
    key = (entry.get("question"), entry.get("kind"))
    with _LOCK:
        items = [i for i in _load() if (i.get("question"), i.get("kind")) != key]
        items.insert(0, entry)
        items = items[:_CAP]
        _save(items)
    return entry
