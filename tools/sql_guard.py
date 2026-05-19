"""
SQL guardrails for agent-issued queries: LIMIT policy and wall-clock timeout (DuckDB interrupt).

Does not modify tools/db_query.py; uses the existing connection on DuckDBQuery.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import TYPE_CHECKING

from observability import log_sql_guard_outcome, trace_span
from tools.sql_error_hints import build_tool_error_payload

if TYPE_CHECKING:
    from tools.db_query import DuckDBQuery

_MAX_LIMIT = 100_000
SERVER_MAX_LIMIT = int(os.getenv("SQL_SERVER_MAX_LIMIT", "10000"))
DEFAULT_QUERY_TIMEOUT_SEC = 60.0

_NO_LIMIT_PREFIXES = (
    "DESCRIBE",
    "SHOW",
    "EXPLAIN",
    "PRAGMA",
    "SUMMARIZE",
    "CALL",
)

_ALLOWED_STARTS = frozenset(
    {"SELECT", "WITH", "DESCRIBE", "SHOW", "EXPLAIN", "PRAGMA", "SUMMARIZE", "CALL"}
)

_FORBIDDEN_SQL_TOKENS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|COPY|ATTACH|DETACH|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

_LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+)\b", re.IGNORECASE)


def _statement_head(sql: str) -> str:
    s = sql.strip().lstrip("(")
    return (s.split(None, 1)[0] if s else "").upper()


def validate_read_only_sql(sql: str | None) -> str | None:
    """Reject mutating or disallowed statement types."""
    if sql is None or not str(sql).strip():
        return "Missing or empty SQL."
    s = sql.strip().rstrip(";")
    head = _statement_head(s)
    if head not in _ALLOWED_STARTS:
        return (
            f"Only read-only exploration SQL is allowed (got {head!r}). "
            "Use SELECT/WITH or schema helpers (DESCRIBE, SHOW, etc.)."
        )
    if _FORBIDDEN_SQL_TOKENS.search(s):
        return "Mutating or administrative SQL keywords are not allowed."
    return None


def cap_sql_limit(sql: str, *, server_max: int = SERVER_MAX_LIMIT) -> str:
    """
    Cap the outermost LIMIT to server_max (replaces last LIMIT in the string).
    Caller must have already validated that LIMIT exists where required.
    """
    matches = list(_LIMIT_RE.finditer(sql))
    if not matches:
        return sql
    last = matches[-1]
    n = int(last.group(1))
    if n <= server_max:
        return sql
    capped = str(server_max)
    return sql[: last.start(1)] + capped + sql[last.end(1) :]


def validate_exploration_sql(sql: str | None) -> str | None:
    """Return error message if invalid, else None. SELECT ... FROM ... requires LIMIT n."""
    ro_err = validate_read_only_sql(sql)
    if ro_err:
        return ro_err
    if sql is None or not str(sql).strip():
        return "Missing or empty SQL."
    s = sql.strip().rstrip(";")

    if ";" in s:
        return "Only one SQL statement per call (no semicolon-separated batches)."

    head = _statement_head(s)
    if head in _NO_LIMIT_PREFIXES:
        return None

    upper = s.upper()
    if "SELECT" in upper and "FROM" in upper:
        m = _LIMIT_RE.search(s)
        if not m:
            return (
                "SELECT queries that read FROM a table must include LIMIT <n> "
                f"(max {_MAX_LIMIT}). Example: ... LIMIT 5000. "
                "For a single aggregate row use LIMIT 1."
            )
        n = int(m.group(1))
        if n < 1:
            return "LIMIT must be a positive integer."
        if n > _MAX_LIMIT:
            return f"LIMIT must be at most {_MAX_LIMIT}."
    return None


def prepare_sql_for_execution(sql: str) -> tuple[str | None, str | None]:
    """
    Validate and apply server-side LIMIT cap.
    Returns (executable_sql, error_message_or_none).
    """
    err = validate_exploration_sql(sql)
    if err:
        return None, err
    return cap_sql_limit(sql.strip().rstrip(";")), None


def query_with_columns_timed(
    db: "DuckDBQuery",
    sql: str,
    *,
    timeout_sec: float = DEFAULT_QUERY_TIMEOUT_SEC,
) -> tuple[list[str] | None, list[tuple] | None, str | None]:
    """
    Run SQL; return (columns, rows, error_json_string_or_none).
    """
    exec_sql, err = prepare_sql_for_execution(sql)
    if err:
        log_sql_guard_outcome(ok=False, message=err)
        return None, None, json.dumps(build_tool_error_payload(err, policy=True))

    with trace_span("sql.execute", timeout_sec=timeout_sec):
        conn = db.conn
        timer = threading.Timer(timeout_sec, conn.interrupt)
        timer.start()
        t0 = time.perf_counter()
        try:
            result = conn.execute(exec_sql)
            desc = result.description
            columns = [d[0] for d in desc] if desc else []
            rows = result.fetchall()
            elapsed = (time.perf_counter() - t0) * 1000.0
            log_sql_guard_outcome(
                ok=True,
                duration_ms=elapsed,
                row_count=len(rows) if rows is not None else 0,
            )
            return columns, rows, None
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000.0
            msg = str(e).lower()
            name = type(e).__name__
            if "interrupt" in msg or "interrupt" in name.lower():
                interrupt_msg = (
                    f"Query interrupted after {timeout_sec:.0f}s "
                    "(wall-clock limit or cancellation)."
                )
                log_sql_guard_outcome(ok=False, message=interrupt_msg, duration_ms=elapsed)
                return None, None, json.dumps(
                    build_tool_error_payload(interrupt_msg, interrupt=True)
                )
            log_sql_guard_outcome(ok=False, message=str(e), duration_ms=elapsed)
            return None, None, json.dumps(build_tool_error_payload(str(e)))
        finally:
            timer.cancel()


async def async_query_with_columns_timed(
    db: "DuckDBQuery",
    sql: str,
    *,
    timeout_sec: float = DEFAULT_QUERY_TIMEOUT_SEC,
) -> tuple[list[str] | None, list[tuple] | None, str | None]:
    """Async wrapper — runs query_with_columns_timed in a thread pool."""
    import asyncio

    return await asyncio.to_thread(query_with_columns_timed, db, sql, timeout_sec=timeout_sec)
