"""OpenAI tool schemas (`OPENAI_TOOLS`) and `dispatch_tool` — all SQL via `DuckDBQuery`."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import date, datetime
import time

from observability import get_logger, log_tool_result, trace_span
from tools.chart_tool import create_chart_from_sql
from tools import permissions as perm
from tools.db_query import DuckDBQuery
from tools.session_state import SessionState
from tools.query_cache import cache_get, cache_set
from tools.sql_error_hints import build_tool_error_payload, structured_tool_error
from tools.sql_guard import query_with_columns_timed


_SQL_CARE_GAP_DIABETES_LAB = """
WITH dm AS (
  SELECT DISTINCT "PATIENT_NUMBER"
  FROM mx_events
  WHERE "ADMIT_DIAGNOSIS_CODE" LIKE 'E11%'
),
lab AS (
  SELECT DISTINCT "PATIENT_NUMBER"
  FROM mx_events
  WHERE "VISIT_TYPE" = 'LAB'
)
SELECT
  COUNT(*) AS dm_patients,
  SUM(CASE WHEN lab."PATIENT_NUMBER" IS NULL THEN 1 ELSE 0 END) AS dm_without_lab_visit_type
FROM dm
LEFT JOIN lab ON dm."PATIENT_NUMBER" = lab."PATIENT_NUMBER"
LIMIT 1
""".strip()

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


def _summarize_sql_stats_impl(
    db: DuckDBQuery,
    sql: str | None,
    value_column: str | None,
) -> dict:
    """Run guarded SQL and summarize one numeric column (pandas)."""
    import pandas as pd

    columns, rows, err_json = query_with_columns_timed(db, sql)
    if err_json:
        return json.loads(err_json)
    if not columns or rows is None:
        return structured_tool_error(
            "Empty query result.",
            error_kind="empty_result",
            next_step="Run a SELECT that returns rows (check filters and JOINs), then retry summarize_sql_stats.",
        )
    df = pd.DataFrame(rows, columns=columns)
    numeric = df.select_dtypes(include=["number"])
    extra_cols = list(numeric.columns)
    if value_column and str(value_column).strip():
        vc = str(value_column).strip()
        if vc not in df.columns:
            return {
                **structured_tool_error(
                    f"Column {value_column!r} not in result.",
                    error_kind="summarize_bad_column",
                    next_step="Pick a column name from `columns` or omit value_column when only one numeric exists.",
                ),
                "columns": list(df.columns),
            }
        series = df[vc]
        if not pd.api.types.is_numeric_dtype(series):
            return structured_tool_error(
                f"Column {value_column!r} is not numeric.",
                error_kind="summarize_not_numeric",
                next_step="Choose a numeric column or cast in SQL (e.g. TRY_CAST) before summarizing.",
            )
    else:
        if numeric.shape[1] == 0:
            return {
                **structured_tool_error(
                    (
                        "No numeric columns in result; narrow the SELECT, use aggregates, "
                        "or set value_column to a numeric field."
                    ),
                    error_kind="summarize_no_numeric",
                    next_step="Return at least one numeric column from SQL, or set value_column to a numeric field.",
                ),
                "columns": list(df.columns),
            }
        if numeric.shape[1] > 1:
            return {
                **structured_tool_error(
                    "Multiple numeric columns; set value_column to one of them.",
                    error_kind="summarize_ambiguous_numeric",
                    next_step="Set value_column to exactly one of numeric_columns.",
                ),
                "numeric_columns": extra_cols,
            }
        series = numeric.iloc[:, 0]
    vc = series.dropna()
    stats: dict = {
        "column": str(series.name),
        "count_non_null": int(vc.shape[0]),
        "count_null": int(series.isna().sum()),
        "row_count": int(len(df)),
    }
    if len(vc) == 0:
        stats["note"] = "All values null for this column."
        return {"stats": stats}

    stats["min"] = float(vc.min())
    stats["max"] = float(vc.max())
    stats["mean"] = float(vc.mean())
    stats["std"] = float(vc.std(ddof=1)) if len(vc) > 1 else 0.0
    stats["p25"] = float(vc.quantile(0.25))
    stats["p50"] = float(vc.quantile(0.50))
    stats["p75"] = float(vc.quantile(0.75))
    return {"stats": stats}


def _profile_table_impl(db: DuckDBQuery, table_name: str) -> dict:
    canon = _resolve_table_name(db, table_name)
    if not canon:
        return structured_tool_error(
            f"Unknown table: {table_name!r}. Use list_tables first.",
            error_kind="unknown_table",
            next_step="Call list_tables, then use an exact table name with profile_table or describe_table.",
        )
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


_DQ_DEFAULT_SAMPLE = 50_000
_DQ_MAX_SAMPLE = 250_000  # same row cap as profile_table SUMMARIZE subquery


def _iqr_outlier_stats(series) -> dict | None:
    """Tukey fences on non-null numeric values; None if not numeric."""
    import pandas as pd

    if not pd.api.types.is_numeric_dtype(series):
        return None
    v = pd.to_numeric(series, errors="coerce").dropna()
    n = len(v)
    if n == 0:
        return {"method": "iqr_1.5", "count": 0, "rate_of_non_null": 0.0, "note": "all_null"}
    if n < 4:
        return {
            "method": "iqr_1.5",
            "count": 0,
            "rate_of_non_null": 0.0,
            "note": "fewer_than_4_non_null",
        }
    q1 = float(v.quantile(0.25))
    q3 = float(v.quantile(0.75))
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    out = int(((v < low) | (v > high)).sum())
    return {
        "method": "iqr_1.5",
        "count": out,
        "rate_of_non_null": float(out / n),
    }


def _data_quality_check_impl(
    db: DuckDBQuery,
    table_name: str | None,
    sample_limit: int | None = None,
    flag_null_rate_above: float | None = None,
) -> dict:
    """
    Null rates, distinct counts (sample), and IQR-based outlier rates for numeric columns.
    Uses the same capped SELECT pattern as profile_table (no sql_guard on agent SQL).
    """
    import pandas as pd

    thr: float | None = None
    if flag_null_rate_above is not None:
        try:
            thr = float(flag_null_rate_above)
        except (TypeError, ValueError):
            return structured_tool_error(
                "flag_null_rate_above must be a number between 0 and 1.",
                error_kind="dq_bad_args",
                next_step="Omit the flag or pass a float such as 0.15 for 15% nulls.",
            )
        if not 0.0 < thr <= 1.0:
            return structured_tool_error(
                "flag_null_rate_above must be in (0, 1].",
                error_kind="dq_bad_args",
                next_step="Use a fraction like 0.2 for 20%, or omit for flags off.",
            )

    canon = _resolve_table_name(db, table_name)
    if not canon:
        return structured_tool_error(
            f"Unknown table: {table_name!r}. Use list_tables first.",
            error_kind="unknown_table",
            next_step="Call list_tables, then data_quality_check with an exact table name.",
        )

    lim = _DQ_DEFAULT_SAMPLE if sample_limit is None else int(sample_limit)
    lim = max(1, min(lim, _DQ_MAX_SAMPLE))

    safe = canon.replace('"', '""')
    _, schema_rows = db.query_with_columns(f'DESCRIBE "{safe}"')
    type_by_col = {row[0]: str(row[1]) for row in schema_rows}

    sql = f'SELECT * FROM "{safe}" LIMIT {lim}'
    columns, rows = db.query_with_columns(sql)
    df = pd.DataFrame(rows, columns=columns) if columns else pd.DataFrame()

    col_stats: list[dict] = []
    threshold_flags: list[dict] = []

    for col in df.columns:
        s = df[col]
        if len(s) == 0:
            null_rate = 0.0
            distinct_non_null = 0
            out = None
        else:
            null_rate = float(s.isna().mean())
            distinct_non_null = int(s.dropna().nunique())
            out = _iqr_outlier_stats(s)

        entry: dict = {
            "name": col,
            "duckdb_type": type_by_col.get(col, ""),
            "null_rate": null_rate,
            "distinct_count_non_null": distinct_non_null,
            "outliers": out,
        }

        if thr is not None and null_rate > thr:
            entry["high_null"] = True
            threshold_flags.append(
                {
                    "column": col,
                    "null_rate": null_rate,
                    "threshold": thr,
                }
            )
        col_stats.append(entry)

    return {
        "table": canon,
        "sample_rows": int(len(df)),
        "sample_limit": lim,
        "flag_null_rate_above": thr if thr is not None else None,
        "threshold_hits": threshold_flags,
        "columns": col_stats,
        "note": (
            "Metrics are computed on a sequential row sample (LIMIT), like profile_table's SUMMARIZE sample. "
            "Distinct counts and outlier rates apply to this sample only, not the full table."
        ),
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
            "name": "summarize_sql_stats",
            "description": (
                "Run read-only SQL (same LIMIT rules as query_database) and compute summary statistics "
                "(min, max, mean, std, quartiles) on a single numeric column. "
                "If the query returns exactly one numeric column, it is used automatically; "
                "if multiple numeric columns are returned, you must set value_column."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "DuckDB SELECT with LIMIT when SELECT...FROM applies (same as query_database).",
                    },
                    "value_column": {
                        "type": "string",
                        "description": "Name of the numeric column to summarize when multiple numeric columns exist.",
                    },
                },
                "required": ["sql"],
            },
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
            "name": "data_quality_check",
            "description": (
                "Column-level data quality on a table: null rates, distinct counts (non-null), and IQR outlier rates "
                "for numeric columns. Uses a capped row sample (same spirit as profile_table). "
                "Optional flag_null_rate_above highlights columns exceeding a null-rate threshold."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "A valid table name from list_tables",
                    },
                    "sample_limit": {
                        "type": "integer",
                        "description": (
                            f"Max rows to scan (default {_DQ_DEFAULT_SAMPLE}, max {_DQ_MAX_SAMPLE}). "
                            "Larger samples give more stable distinct/outlier estimates but cost more time."
                        ),
                    },
                    "flag_null_rate_above": {
                        "type": "number",
                        "description": (
                            "Optional fraction in (0,1], e.g. 0.15 flags columns with null_rate > 15%. "
                            "Omit for raw stats only."
                        ),
                    },
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
    {
        "type": "function",
        "function": {
            "name": "analyze_care_gap",
            "description": (
                "Run a predefined care-gap style analysis on synthetic claims data (utilization proxies, not clinical quality measures). "
                "Supported analyses compare a cohort to missing event patterns (e.g. E11 diabetes signal vs no LAB visit type on mx_events). "
                "Prefer this when the user asks about care gaps, missing follow-up labs, or similar — instead of inventing ad-hoc SQL."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gap_type": {
                        "type": "string",
                        "enum": ["diabetes_lab_utilization_proxy"],
                        "description": (
                            "diabetes_lab_utilization_proxy: Among patients with an E11% admit diagnosis on mx_events, "
                            "count those with no mx_events row where VISIT_TYPE = 'LAB' (lab utilization proxy only)."
                        ),
                    }
                },
                "required": ["gap_type"],
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
    user_role: str | None = None,
) -> str:
    """Run a registered tool and return JSON string. Updates SessionState when provided."""
    logger = get_logger("agent.tool")
    role = perm.normalize_role(user_role)
    policy = perm.get_policy(role)
    sql_preview = ""
    if isinstance(arguments.get("sql"), str):
        sql_preview = str(arguments["sql"]).strip()[:80]
    with trace_span("tool.dispatch", tool=tool_name, role=role):
        t0 = time.perf_counter()
        try:
            out = _dispatch_tool_impl(
                tool_name, arguments, db, state, role=role, policy=policy
            )
        except Exception as exc:
            logger.exception(
                "tool dispatch raised",
                extra={
                    "event": "tool.exception",
                    "tool": tool_name,
                    "error_type": type(exc).__name__,
                },
            )
            out = json.dumps(build_tool_error_payload(str(exc)))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        log_tool_result(tool_name, out, duration_ms=elapsed_ms)
        if sql_preview and tool_name in ("query_database", "summarize_sql_stats", "create_chart"):
            logger.debug(
                "sql preview",
                extra={"event": "tool.sql_preview", "tool": tool_name, "sql": sql_preview},
            )
        return out


def _dispatch_tool_impl(
    tool_name: str,
    arguments: dict,
    db: DuckDBQuery,
    state: SessionState | None,
    *,
    role: str,
    policy: perm.RolePolicy,
) -> str:
    try:
        if tool_name not in perm.ALL_TOOL_NAMES:
            return json.dumps(
                structured_tool_error(
                    f"Unknown tool: {tool_name}",
                    error_kind="unknown_tool",
                    next_step="Use only tools from the registered schema (list_tables, query_database, etc.).",
                )
            )
        if not perm.tool_allowed_for_role(policy, tool_name):
            return json.dumps(perm.forbidden_tool_payload(tool_name, role))

        if tool_name == "list_tables":
            tables = db.list_tables()
            if policy.allowed_table_names is not None:
                allow = {t.lower() for t in policy.allowed_table_names}
                tables = [t for t in tables if t.lower() in allow]
            return json.dumps({"tables": tables}, cls=CustomEncoder)

        if tool_name == "query_database":
            sql = arguments.get("sql")
            viol = perm.sql_violates_table_policy(policy, sql, role)
            if viol:
                return json.dumps(viol, cls=CustomEncoder)
            if state:
                state.note_sql(sql if isinstance(sql, str) else None)
                if isinstance(sql, str):
                    for t in _extract_table_hints(sql):
                        state.note_table(t)
            cached = cache_get("query_database", sql if isinstance(sql, str) else None)
            if cached is not None:
                if state and isinstance(sql, str) and sql.strip():
                    state.note_cohort_sql(sql)
                return cached
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
            out = json.dumps(result, cls=CustomEncoder)
            cache_set("query_database", sql if isinstance(sql, str) else None, out)
            if state and isinstance(sql, str) and sql.strip():
                state.note_cohort_sql(sql)
            return out

        if tool_name == "summarize_sql_stats":
            sql = arguments.get("sql")
            value_column = arguments.get("value_column")
            extra = str(value_column or "")
            viol = perm.sql_violates_table_policy(policy, sql, role)
            if viol:
                return json.dumps(viol, cls=CustomEncoder)
            if state:
                state.note_sql(sql if isinstance(sql, str) else None)
                if isinstance(sql, str):
                    for t in _extract_table_hints(sql):
                        state.note_table(t)
            cached = cache_get("summarize_sql_stats", sql if isinstance(sql, str) else None, extra=extra)
            if cached is not None:
                if state and isinstance(sql, str) and sql.strip():
                    try:
                        d = json.loads(cached)
                        if isinstance(d, dict) and "error" not in d:
                            state.note_cohort_sql(sql)
                    except json.JSONDecodeError:
                        pass
                return cached
            result = _summarize_sql_stats_impl(
                db,
                sql if isinstance(sql, str) else None,
                value_column if isinstance(value_column, str) else None,
            )
            out = json.dumps(result, cls=CustomEncoder)
            cache_set("summarize_sql_stats", sql if isinstance(sql, str) else None, out, extra=extra)
            if state and isinstance(sql, str) and sql.strip() and isinstance(result, dict) and "error" not in result:
                state.note_cohort_sql(sql)
            return out

        if tool_name == "describe_table":
            table_name = arguments.get("table_name")
            canon = _resolve_table_name(db, table_name)
            if not canon:
                return json.dumps(
                    structured_tool_error(
                        f"Unknown table: {table_name!r}",
                        error_kind="unknown_table",
                        next_step="Use list_tables to see valid names, then describe_table with an exact match.",
                    )
                )
            if not perm.table_name_allowed_for_role(policy, canon):
                return json.dumps(perm.forbidden_table_payload(canon, role), cls=CustomEncoder)
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
                return json.dumps(
                    structured_tool_error(
                        f"Unknown table: {table_name!r}",
                        error_kind="unknown_table",
                        next_step="Use list_tables to see valid names, then table_info with an exact match.",
                    )
                )
            if not perm.table_name_allowed_for_role(policy, canon):
                return json.dumps(perm.forbidden_table_payload(canon, role), cls=CustomEncoder)
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
            canon = _resolve_table_name(db, table_name)
            if canon and not perm.table_name_allowed_for_role(policy, canon):
                return json.dumps(perm.forbidden_table_payload(canon, role), cls=CustomEncoder)
            result = _profile_table_impl(db, table_name)
            if state and "table" in result:
                state.note_profile(result["table"])
            return json.dumps(result, cls=CustomEncoder)

        if tool_name == "data_quality_check":
            table_name = arguments.get("table_name")
            canon = _resolve_table_name(db, table_name)
            if canon and not perm.table_name_allowed_for_role(policy, canon):
                return json.dumps(perm.forbidden_table_payload(canon, role), cls=CustomEncoder)
            sl_raw = arguments.get("sample_limit")
            sample_limit_parsed: int | None = None
            if sl_raw is not None and not isinstance(sl_raw, bool):
                try:
                    sample_limit_parsed = int(sl_raw)
                except (TypeError, ValueError):
                    sample_limit_parsed = None
            fn_raw = arguments.get("flag_null_rate_above")
            fn_parsed: float | None = None
            if fn_raw is not None and type(fn_raw) in (int, float) and not isinstance(fn_raw, bool):
                fn_parsed = float(fn_raw)
            result = _data_quality_check_impl(
                db,
                table_name if isinstance(table_name, str) else None,
                sample_limit=sample_limit_parsed,
                flag_null_rate_above=fn_parsed,
            )
            if state and isinstance(result, dict) and result.get("table") and "error" not in result:
                state.note_profile(result["table"])
            return json.dumps(result, cls=CustomEncoder)

        if tool_name == "create_chart":
            sql = arguments.get("sql")
            viol = perm.sql_violates_table_policy(policy, sql, role)
            if viol:
                return json.dumps(viol, cls=CustomEncoder)
            chart_type = arguments.get("chart_type") or "bar"
            title = arguments.get("title")
            if state and isinstance(sql, str):
                state.note_sql(sql)
                for t in _extract_table_hints(sql):
                    state.note_table(t)
            result = create_chart_from_sql(db, sql, chart_type=chart_type, title=title)
            if state and isinstance(sql, str) and sql.strip() and isinstance(result, dict) and "error" not in result:
                state.note_cohort_sql(sql)
            if state and "chart_path" in result:
                state.note_chart(result["chart_path"])
            return json.dumps(result, cls=CustomEncoder)

        if tool_name == "analyze_care_gap":
            if not perm.table_name_allowed_for_role(policy, "mx_events"):
                return json.dumps(perm.forbidden_table_payload("mx_events", role), cls=CustomEncoder)
            gap_type = arguments.get("gap_type")
            if gap_type != "diabetes_lab_utilization_proxy":
                return json.dumps(
                    structured_tool_error(
                        f"Unsupported gap_type: {gap_type!r}. Use diabetes_lab_utilization_proxy.",
                        error_kind="unsupported_gap_type",
                        next_step="Set gap_type to the supported value, or answer with query_database if the gap differs.",
                    )
                )
            sql = _SQL_CARE_GAP_DIABETES_LAB
            if state:
                state.note_sql(sql)
                for t in _extract_table_hints(sql):
                    state.note_table(t)
            columns, rows, err_json = query_with_columns_timed(db, sql)
            if err_json:
                return err_json
            limited_rows = rows[:100]
            result = {
                "gap_type": gap_type,
                "label": "E11 cohort vs LAB visit-type utilization (proxy)",
                "note": (
                    "Synthetic data. LAB visit type on claims is a utilization proxy, not a specific lab result or HbA1c gap."
                ),
                "columns": columns,
                "rows": [list(row) for row in limited_rows],
                "total_rows": len(rows),
                "truncated": len(rows) > 100,
            }
            out = json.dumps(result, cls=CustomEncoder)
            if state and sql.strip():
                state.note_cohort_sql(sql)
            return out

        return json.dumps(
            structured_tool_error(
                f"Unknown tool: {tool_name}",
                error_kind="unknown_tool",
                next_step="Use only tools from the registered schema (list_tables, query_database, etc.).",
            )
        )

    except Exception as e:
        return json.dumps(build_tool_error_payload(str(e)))


async def async_dispatch_tool(
    tool_name: str,
    arguments: dict,
    db: DuckDBQuery,
    state: SessionState | None,
    user_role: str | None = None,
) -> str:
    """Async wrapper — runs dispatch_tool in a thread pool so DuckDB calls don't block the event loop."""
    return await asyncio.to_thread(dispatch_tool, tool_name, arguments, db, state, user_role)
