"""
SQL guardrails for agent-issued queries: LIMIT policy and wall-clock timeout (DuckDB interrupt).

Does not modify tools/db_query.py; uses the existing connection on DuckDBQuery.
"""

from __future__ import annotations

import json
import re
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools.db_query import DuckDBQuery

_MAX_LIMIT = 100_000
DEFAULT_QUERY_TIMEOUT_SEC = 60.0

_NO_LIMIT_PREFIXES = (
    "DESCRIBE",
    "SHOW",
    "EXPLAIN",
    "PRAGMA",
    "SUMMARIZE",
    "CALL",
)


def validate_exploration_sql(sql: str | None) -> str | None:
    """Return error message if invalid, else None. SELECT ... FROM ... requires LIMIT n."""
    if sql is None or not str(sql).strip():
        return "Missing or empty SQL."
    s = sql.strip().rstrip(";")

    if ";" in s:
        return "Only one SQL statement per call (no semicolon-separated batches)."

    head = s.split(None, 1)[0].upper() if s else ""
    if head in _NO_LIMIT_PREFIXES:
        return None

    upper = s.upper()
    if "SELECT" in upper and "FROM" in upper:
        m = re.search(r"\bLIMIT\s+(\d+)\b", s, re.IGNORECASE)
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


def query_with_columns_timed(
    db: "DuckDBQuery",
    sql: str,
    *,
    timeout_sec: float = DEFAULT_QUERY_TIMEOUT_SEC,
) -> tuple[list[str] | None, list[tuple] | None, str | None]:
    """
    Run SQL; return (columns, rows, error_json_string_or_none).
    """
    err = validate_exploration_sql(sql)
    if err:
        return None, None, json.dumps({"error": err})

    conn = db.conn
    timer = threading.Timer(timeout_sec, conn.interrupt)
    timer.start()
    try:
        result = conn.execute(sql)
        desc = result.description
        columns = [d[0] for d in desc] if desc else []
        rows = result.fetchall()
        return columns, rows, None
    except Exception as e:
        msg = str(e).lower()
        name = type(e).__name__
        if "interrupt" in msg or "interrupt" in name.lower():
            return None, None, json.dumps(
                {
                    "error": f"Query exceeded time limit ({timeout_sec:.0f}s).",
                    "timeout": True,
                    "hint": "Narrow filters, add LIMIT, or aggregate in SQL.",
                }
            )
        raise
    finally:
        timer.cancel()
