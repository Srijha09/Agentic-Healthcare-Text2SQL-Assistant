"""OpenAI tool schemas (`OPENAI_TOOLS`) and `dispatch_tool` — all SQL via `DuckDBQuery`."""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from tools.chart_tool import create_chart_from_sql
from tools.db_query import DuckDBQuery
from tools.session_state import SessionState
from tools.sql_guard import query_with_columns_timed


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)


def _resolve_table_name(db: DuckDBQuery, table_name: str | None) -> str | None:
    """Match user-provided name to an actual table (case-insensitive)."""
    if not table_name or not table_name.strip():
        return None
    want = table_name.strip().lower()
    for t in db.list_tables():
        if t.lower() == want:
            return t
    return None


def _profile_table_impl(db: DuckDBQuery, table_name: str) -> dict:
    canon = _resolve_table_name(db, table_name)
    if not canon:
        return {"error": f"Unknown table: {table_name!r}. Use list_tables first."}
    # Cap scanned rows so huge tables stay responsive; enough for stats
    safe = canon.replace('"', '""')
    sql = f'SUMMARIZE (SELECT * FROM "{safe}" LIMIT 250000)'
    columns, rows = db.query_with_columns(sql)
    max_rows = 120
    out_rows = [list(r) for r in rows[:max_rows]]
    return {
        "table": canon,
        "summary_columns": columns,
        "summary_rows": out_rows,
        "total_summary_rows": len(rows),
        "truncated": len(rows) > max_rows,
        "note": "DuckDB SUMMARIZE stats per column (sample up to 250k rows).",
    }


# OpenAI Chat Completions tool definitions
OPENAI_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "List all available tables in the healthcare database",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Execute a SQL query on the healthcare database and return results",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": (
                            "DuckDB SQL. If the query uses SELECT and FROM, it must end with LIMIT n "
                            "(e.g. LIMIT 5000 or LIMIT 1 for a single aggregate row)."
                        ),
                    }
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_table",
            "description": "Get the schema/structure of a specific table including column names and types",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "The name of the table to describe",
                    }
                },
                "required": ["table_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "table_info",
            "description": "Get comprehensive information about a table including row count, schema, and sample data",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "The name of the table to get info about",
                    }
                },
                "required": ["table_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "profile_table",
            "description": (
                "Get column-level summary statistics for a table (DuckDB SUMMARIZE on a row sample). "
                "Use after list_tables when exploring data quality or column ranges before writing complex SQL."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "A valid table name from the database",
                    }
                },
                "required": ["table_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_chart",
            "description": (
                "Run a read-only SQL query that returns exactly two columns: first = category or date label, "
                "second = numeric value to plot. SQL is validated and timed the same way as query_database (LIMIT required for SELECT...FROM; see sql_guard). "
                "Saves a PNG under outputs/visualization/ and returns markdown_embed. Prefer bar for categories, line for time-ordered labels. "
                "Always pass a title that names cohort or population, the metric, and the time range or grouping."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": (
                            "SELECT with two columns; must include LIMIT n (same rule as query_database). "
                            "Example: ... GROUP BY 1 ORDER BY 1 LIMIT 60"
                        ),
                    },
                    "chart_type": {
                        "type": "string",
                        "enum": ["bar", "line"],
                        "description": "bar for categorical x; line when x is ordered (e.g. months)",
                    },
                    "title": {
                        "type": "string",
                        "description": (
                            "Figure title: include cohort or filter, metric, and time window or dimension "
                            "(e.g. 'PROVIDER_BILLED by state, mx_events, top 15' or 'Monthly metformin fills, E11 cohort')."
                        ),
                    },
                },
                "required": ["sql", "chart_type", "title"],
            },
        },
    },
]

# Optional simple SQL hints for session state (best-effort)
_FROM_RE = re.compile(
    r"\bfrom\s+([a-zA-Z_][a-zA-Z0-9_]*)",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_table_hints(sql: str) -> list[str]:
    return list(dict.fromkeys(_FROM_RE.findall(sql)))[:8]


def dispatch_tool(
    tool_name: str,
    arguments: dict,
    db: DuckDBQuery,
    state: SessionState | None,
) -> str:
    """Run a registered tool and return JSON string. Updates SessionState when provided."""
    try:
        if tool_name == "list_tables":
            tables = db.list_tables()
            return json.dumps({"tables": tables}, cls=CustomEncoder)

        if tool_name == "query_database":
            sql = arguments.get("sql")
            if state:
                state.note_sql(sql if isinstance(sql, str) else None)
                if isinstance(sql, str):
                    for t in _extract_table_hints(sql):
                        state.note_table(t)
            columns, rows, err_json = query_with_columns_timed(db, sql)
            if err_json:
                return err_json
            limited_rows = rows[:100]
            result = {
                "columns": columns,
                "rows": [list(row) for row in limited_rows],
                "total_rows": len(rows),
                "truncated": len(rows) > 100,
            }
            return json.dumps(result, cls=CustomEncoder)

        if tool_name == "describe_table":
            table_name = arguments.get("table_name")
            canon = _resolve_table_name(db, table_name)
            if not canon:
                return json.dumps({"error": f"Unknown table: {table_name!r}"})
            if state:
                state.note_table(canon)
            safe = canon.replace('"', '""')
            columns, rows = db.query_with_columns(f'DESCRIBE "{safe}"')
            result = {
                "table": canon,
                "schema": [{"column": row[0], "type": row[1]} for row in rows],
            }
            return json.dumps(result, cls=CustomEncoder)

        if tool_name == "table_info":
            table_name = arguments.get("table_name")
            canon = _resolve_table_name(db, table_name)
            if not canon:
                return json.dumps({"error": f"Unknown table: {table_name!r}"})
            if state:
                state.note_table(canon)
            safe = canon.replace('"', '""')
            count = db.query(f'SELECT COUNT(*) FROM "{safe}"')[0][0]
            columns, schema_rows = db.query_with_columns(f'DESCRIBE "{safe}"')
            _, sample_rows = db.query_with_columns(f'SELECT * FROM "{safe}" LIMIT 5')
            result = {
                "table": canon,
                "row_count": count,
                "schema": [{"column": row[0], "type": row[1]} for row in schema_rows],
                "sample_data": {
                    "columns": [row[0] for row in schema_rows],
                    "rows": [list(row) for row in sample_rows],
                },
            }
            return json.dumps(result, cls=CustomEncoder)

        if tool_name == "profile_table":
            table_name = arguments.get("table_name")
            result = _profile_table_impl(db, table_name)
            if state and "table" in result:
                state.note_profile(result["table"])
            return json.dumps(result, cls=CustomEncoder)

        if tool_name == "create_chart":
            sql = arguments.get("sql")
            chart_type = arguments.get("chart_type") or "bar"
            title = arguments.get("title")
            if state and isinstance(sql, str):
                state.note_sql(sql)
                for t in _extract_table_hints(sql):
                    state.note_table(t)
            result = create_chart_from_sql(db, sql, chart_type=chart_type, title=title)
            if state and "chart_path" in result:
                state.note_chart(result["chart_path"])
            return json.dumps(result, cls=CustomEncoder)

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        return json.dumps({"error": str(e)})
