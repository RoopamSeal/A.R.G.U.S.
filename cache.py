"""Small JSON cache for PubMed and LLM results."""
import hashlib
import json
import os
import tempfile
import time
import config

def _key(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()

def _load() -> dict:
    if not os.path.exists(config.CACHE_FILE):
        return {}
    try:
        with open(config.CACHE_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}

def _save(data: dict):
    directory = os.path.dirname(os.path.abspath(config.CACHE_FILE))
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".argus-cache-", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
        os.replace(temp_path, config.CACHE_FILE)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def get(query: str):
    """Return a cached payload; supports the previous cache envelope too."""
    entry = _load().get(_key(query))
    if not isinstance(entry, dict):
        return None
    return entry.get("payload", entry.get("insights"))

def set(query: str, payload):
    """Store any JSON-serializable workspace result under a stable query key."""
    data = _load()
    data[_key(query)] = {"query":query, "payload":payload, "cached_at":time.time()}
    _save(data)

def clear():
    _save({})
