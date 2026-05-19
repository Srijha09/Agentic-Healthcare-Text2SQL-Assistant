"""Tests for SQL exploration guard (LIMIT policy)."""

from tools.sql_guard import (
    cap_sql_limit,
    prepare_sql_for_execution,
    validate_exploration_sql,
    validate_read_only_sql,
)


def test_select_from_requires_limit() -> None:
    err = validate_exploration_sql("SELECT * FROM demographics")
    assert err is not None
    assert "LIMIT" in err


def test_select_from_with_limit_ok() -> None:
    assert validate_exploration_sql("SELECT * FROM demographics LIMIT 10") is None


def test_describe_skips_limit_rule() -> None:
    assert validate_exploration_sql('DESCRIBE "demographics"') is None


def test_multi_statement_rejected() -> None:
    err = validate_exploration_sql("SELECT 1 LIMIT 1; SELECT 2 LIMIT 1")
    assert err is not None
    assert "one SQL" in err.lower() or "semicolon" in err.lower()


def test_limit_max_cap() -> None:
    err = validate_exploration_sql("SELECT * FROM demographics LIMIT 200000")
    assert err is not None


def test_limit_zero_rejected() -> None:
    err = validate_exploration_sql("SELECT * FROM demographics LIMIT 0")
    assert err is not None
    assert "positive" in err.lower() or "limit" in err.lower()


def test_select_from_allows_trailing_semicolon_stripped() -> None:
    """Semicolons are stripped; single statement with LIMIT is valid."""
    assert validate_exploration_sql("SELECT 1 FROM demographics LIMIT 1;") is None


def test_insert_rejected() -> None:
    err = validate_read_only_sql("INSERT INTO demographics VALUES (1)")
    assert err is not None
    assert "read-only" in err.lower() or "not allowed" in err.lower()


def test_with_select_ok() -> None:
    assert validate_read_only_sql(
        "WITH x AS (SELECT 1 AS n FROM demographics LIMIT 1) SELECT n FROM x LIMIT 1"
    ) is None


def test_cap_sql_limit_reduces_high_limit() -> None:
    sql = "SELECT * FROM demographics LIMIT 50000"
    capped = cap_sql_limit(sql, server_max=10000)
    assert "LIMIT 10000" in capped
    assert "50000" not in capped


def test_prepare_sql_for_execution_caps() -> None:
    exec_sql, err = prepare_sql_for_execution("SELECT 1 FROM demographics LIMIT 99999")
    assert err is None
    assert exec_sql is not None
    assert "LIMIT 10000" in exec_sql
