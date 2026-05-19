"""
Role-based access for tools and table names (enforced in dispatch above raw SQL tools).

Roles are simple strings; default is *analyst* (full access). *viewer* restricts tools and tables.
Override via `APP_USER_ROLE` env or `run_user_turn(..., user_role=...)`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from tools.sql_error_hints import structured_tool_error

# All tools registered in tool_registry.OPENAI_TOOLS
ALL_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "list_tables",
        "summarize_sql_stats",
        "query_database",
        "describe_table",
        "table_info",
        "profile_table",
        "data_quality_check",
        "create_chart",
        "analyze_care_gap",
    }
)

# Viewer: read-focused; no data_quality_check or analyze_care_gap; subset of tables.
VIEWER_TOOL_NAMES: frozenset[str] = ALL_TOOL_NAMES - {
    "data_quality_check",
    "analyze_care_gap",
}
VIEWER_TABLE_NAMES: frozenset[str] = frozenset({"demographics", "mx_events"})

DEFAULT_ROLE = "analyst"


@dataclass(frozen=True)
class RolePolicy:
    allowed_tool_names: frozenset[str]
    # If None, all tables in the database are allowed (for SQL and describe targets).
    allowed_table_names: frozenset[str] | None


_ROLE_POLICIES: dict[str, RolePolicy] = {
    "analyst": RolePolicy(ALL_TOOL_NAMES, None),
    "admin": RolePolicy(ALL_TOOL_NAMES, None),
    "viewer": RolePolicy(VIEWER_TOOL_NAMES, VIEWER_TABLE_NAMES),
}

_TABLE_REF = re.compile(
    r'\b(?:FROM|JOIN)\s+(?:"([^"]+)"|([a-zA-Z_][a-zA-Z0-9_]*))',
    re.IGNORECASE,
)


def normalize_role(role: str | None) -> str:
    r = (role or DEFAULT_ROLE or "").strip().lower()
    if r in _ROLE_POLICIES:
        return r
    return DEFAULT_ROLE


def get_policy(role: str | None) -> RolePolicy:
    r = normalize_role(role)
    return _ROLE_POLICIES[r]


def tool_allowed_for_role(policy: RolePolicy, tool_name: str) -> bool:
    return tool_name in policy.allowed_tool_names


def table_name_allowed_for_role(policy: RolePolicy, table_name: str) -> bool:
    if policy.allowed_table_names is None:
        return True
    want = table_name.strip().lower()
    return want in {t.lower() for t in policy.allowed_table_names}


def extract_sql_table_refs(sql: str) -> list[str]:
    """Best-effort table identifiers from FROM / JOIN (same spirit as session export)."""
    if not sql or not str(sql).strip():
        return []
    out: list[str] = []
    for m in _TABLE_REF.finditer(str(sql)):
        name = (m.group(1) or m.group(2) or "").strip()
        if not name:
            continue
        up = name.upper()
        if up in ("SELECT", "LATERAL", "UNNEST"):
            continue
        if name not in out:
            out.append(name)
    return out


def forbidden_tool_payload(tool_name: str, role: str) -> dict[str, Any]:
    return structured_tool_error(
        f"Tool {tool_name!r} is not allowed for your role ({role!r}).",
        error_kind="forbidden_tool",
        next_step="Use an allowed tool, or switch role (e.g. analyst) if permitted by policy.",
    )


def forbidden_table_payload(table_name: str, role: str) -> dict[str, Any]:
    return structured_tool_error(
        f"Table {table_name!r} is not allowed for your role ({role!r}).",
        error_kind="forbidden_table",
        next_step="Narrow to allowed tables, or use list_tables to see which names you may query.",
    )


def sql_violates_table_policy(
    policy: RolePolicy, sql: str | None, role_label: str
) -> dict[str, Any] | None:
    if policy.allowed_table_names is None or not sql:
        return None
    for t in extract_sql_table_refs(str(sql)):
        if not table_name_allowed_for_role(policy, t):
            return forbidden_table_payload(t, role_label)
    return None


def filter_openai_tools(tools: list[dict], role: str | None) -> list[dict]:
    """Return tool schema entries allowed for the given role."""
    pol = get_policy(role)
    out: list[dict] = []
    for t in tools:
        fn = t.get("function") or {}
        name = fn.get("name")
        if isinstance(name, str) and name in pol.allowed_tool_names:
            out.append(t)
    return out
