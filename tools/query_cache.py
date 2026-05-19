"""Short-TTL in-memory cache for idempotent SQL tool JSON results (same SQL → same answer)."""

from __future__ import annotations

import hashlib
import json
import threading
import time

DEFAULT_TTL_SEC = 300.0
DEFAULT_MAX_ENTRIES = 128


class SqlResultCache:
    """Thread-safe cache keyed by tool name + optional extra + normalized SQL."""

    def __init__(self, ttl_sec: float = DEFAULT_TTL_SEC, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self._ttl = ttl_sec
        self._max = max_entries
        self._data: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def make_key(tool: str, sql: str | None, extra: str = "") -> str:
        if not sql or not str(sql).strip():
            return ""
        norm = " ".join(str(sql).strip().split())
        raw = f"{tool}\n{extra}\n{norm}".encode()
        return hashlib.sha256(raw).hexdigest()

    def get(self, tool: str, sql: str | None, extra: str = "") -> str | None:
        k = self.make_key(tool, sql, extra)
        if not k:
            return None
        now = time.monotonic()
        with self._lock:
            if k not in self._data:
                return None
            exp, val = self._data[k]
            if now > exp:
                del self._data[k]
                return None
            return val

    def set(self, tool: str, sql: str | None, result_json: str, extra: str = "") -> None:
        if not _should_cache_result(result_json):
            return
        k = self.make_key(tool, sql, extra)
        if not k:
            return
        now = time.monotonic()
        with self._lock:
            while len(self._data) >= self._max:
                self._data.pop(next(iter(self._data)))
            self._data[k] = (now + self._ttl, result_json)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_default_cache = SqlResultCache()


def _should_cache_result(result_json: str) -> bool:
    try:
        d = json.loads(result_json)
    except json.JSONDecodeError:
        return False
    if not isinstance(d, dict):
        return False
    if d.get("retry_limit_reached"):
        return False
    if "error" in d:
        return False
    return True


def cache_get(tool: str, sql: str | None, extra: str = "") -> str | None:
    return _default_cache.get(tool, sql, extra)


def cache_set(tool: str, sql: str | None, result_json: str, extra: str = "") -> None:
    _default_cache.set(tool, sql, result_json, extra=extra)


def cache_clear() -> None:
    """Clear all cached entries (e.g. tests)."""
    _default_cache.clear()
