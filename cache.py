"""
Very simple local cache so repeated queries don't re-spend PubMed/LLM calls.
Stores everything in a single JSON file on disk. Fine for v1 - swap for
SQLite later if the cache needs to be queried, not just looked up by key.
"""
import json
import os
import hashlib
import time
import config


def _key(query: str) -> str:
    return hashlib.md5(query.strip().lower().encode()).hexdigest()


def _load() -> dict:
    if not os.path.exists(config.CACHE_FILE):
        return {}
    with open(config.CACHE_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def _save(data: dict):
    with open(config.CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get(query: str):
    """Return cached insights for a query, or None if not cached."""
    data = _load()
    return data.get(_key(query), {}).get("insights")


def set(query: str, insights: list):
    """Save insights for a query."""
    data = _load()
    data[_key(query)] = {
        "query": query,
        "insights": insights,
        "cached_at": time.time(),
    }
    _save(data)


def clear():
    """Wipe the entire cache."""
    _save({})
