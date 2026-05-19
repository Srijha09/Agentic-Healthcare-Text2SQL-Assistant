"""
Heuristic classification of SQL / DuckDB error strings for user-facing next steps (no LLM).
"""

from __future__ import annotations

import json
import re


def structured_tool_error(message: str, *, error_kind: str, next_step: str) -> dict:
    """Consistent validation / tool-level errors (not DuckDB SQL strings)."""
    return {
        "error": (message or "").strip() or "Unknown error.",
        "error_kind": error_kind,
        "next_step": next_step,
    }


def build_tool_error_payload(
    message: str,
    *,
    timeout: bool = False,
    policy: bool = False,
    interrupt: bool = False,
) -> dict:
    """
    Return a dict suitable for json.dumps to the model/user.
    Always includes "error"; may include error_kind and next_step.
    """
    raw = (message or "").strip()
    if not raw:
        return {"error": "Unknown error.", "error_kind": "unknown", "next_step": "Retry with simpler SQL or use list_tables."}

    if interrupt:
        return {
            "error": raw,
            "error_kind": "interrupted",
            "next_step": (
                "DuckDB raised an interrupt (often the wall-clock limit). "
                "Narrow filters, add LIMIT, or split the query. "
                "Some clients also cancel in-flight work; retry if you did not intend to stop."
            ),
        }

    if timeout:
        return {
            "error": raw,
            "error_kind": "timeout",
            "next_step": "Narrow filters, reduce LIMIT, pre-aggregate, or split into smaller queries.",
        }

    if policy:
        return {
            "error": raw,
            "error_kind": "policy",
            "next_step": "Follow the SQL rules: add LIMIT on SELECT...FROM, one statement per call.",
        }

    low = raw.lower()

    if "interrupt" in low or "cancel" in low:
        return {
            "error": raw,
            "error_kind": "interrupted",
            "next_step": (
                "Statement was interrupted or cancelled. If not intentional, narrow scope or add LIMIT and retry."
            ),
        }

    if "catalog" in low and ("does not exist" in low or "not found" in low):
        return {
            "error": raw,
            "error_kind": "catalog",
            "next_step": "Verify table/view names with list_tables, then describe_table / table_info.",
        }

    if "ambiguous" in low or ("multiple" in low and "column" in low) or (
        "more than one" in low and "column" in low
    ):
        return {
            "error": raw,
            "error_kind": "ambiguous_column",
            "next_step": "Qualify columns with table aliases (e.g. t.col) or simplify SELECT so each name is unique.",
        }

    if any(
        s in low
        for s in (
            "cannot be cast",
            "conversion error",
            "could not convert",
            "type mismatch",
            "invalid input for",
            "cannot convert",
        )
    ):
        return {
            "error": raw,
            "error_kind": "type_or_cast",
            "next_step": "Align types (CAST/TRY_CAST), fix literals, or compare compatible columns after joins.",
        }

    if any(s in low for s in ("constraint", "unique", "violates", "not null constraint")):
        return {
            "error": raw,
            "error_kind": "constraint",
            "next_step": "This environment is read-only; if you see constraint errors, simplify the expression or remove writes.",
        }

    if "binder" in low or ("column" in low and ("not found" in low or "cannot" in low or "referenced" in low)):
        return {
            "error": raw,
            "error_kind": "binder",
            "next_step": "Check column names and types with DESCRIBE or table_info on the tables you join.",
        }

    if "parser" in low or "syntax" in low or "syntax error" in low:
        return {
            "error": raw,
            "error_kind": "syntax",
            "next_step": "Fix SQL syntax; validate parentheses, commas, and DuckDB dialect for aggregates.",
        }

    if "out of memory" in low or "oom" in low:
        return {
            "error": raw,
            "error_kind": "resource",
            "next_step": "Reduce scanned rows (stronger filters, smaller LIMIT) or aggregate earlier.",
        }

    return {
        "error": raw,
        "error_kind": "other",
        "next_step": "Compare SQL to schema with describe_table; simplify joins or test a minimal SELECT...LIMIT 1.",
    }
