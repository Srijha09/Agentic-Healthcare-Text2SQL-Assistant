"""Tests for SQL result cache."""

import json

from tools import query_cache
from tools.query_cache import SqlResultCache, cache_clear, cache_get, cache_set


def setup_function() -> None:
    cache_clear()


def test_cache_round_trip() -> None:
    cache_set("query_database", "SELECT 1 LIMIT 1", json.dumps({"ok": True}))
    got = cache_get("query_database", "SELECT 1 LIMIT 1")
    assert got is not None
    assert json.loads(got)["ok"] is True


def test_cache_ignores_errors() -> None:
    cache_set("query_database", "SELECT 1 LIMIT 1", json.dumps({"error": "bad"}))
    assert cache_get("query_database", "SELECT 1 LIMIT 1") is None


def test_cache_key_normalizes_whitespace() -> None:
    cache_set("query_database", "SELECT  1   LIMIT 1", json.dumps({"x": 1}))
    got = cache_get("query_database", "SELECT 1 LIMIT 1")
    assert got is not None


def test_ttl_expiry(monkeypatch) -> None:
    c = SqlResultCache(ttl_sec=1.0, max_entries=8)
    clock = [0.0]
    monkeypatch.setattr("tools.query_cache.time.monotonic", lambda: clock[0])
    c.set("query_database", "SELECT 1 LIMIT 1", '{"ok": true}')
    assert c.get("query_database", "SELECT 1 LIMIT 1") is not None
    clock[0] = 100.0
    assert c.get("query_database", "SELECT 1 LIMIT 1") is None
