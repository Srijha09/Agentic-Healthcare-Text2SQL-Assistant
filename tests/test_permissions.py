"""Role-based tool and table access (no DB for pure logic)."""

import json

from tools.permissions import (
    ALL_TOOL_NAMES,
    extract_sql_table_refs,
    filter_openai_tools,
    get_policy,
    normalize_role,
    sql_violates_table_policy,
)
from tools.tool_registry import OPENAI_TOOLS


def test_normalize_role_unknown_defaults_to_analyst() -> None:
    assert normalize_role("not_a_role") == "analyst"


def test_viewer_excludes_certain_tools() -> None:
    pol = get_policy("viewer")
    assert "data_quality_check" not in pol.allowed_tool_names
    assert "analyze_care_gap" not in pol.allowed_tool_names
    assert "query_database" in pol.allowed_tool_names


def test_filter_openai_tools_viewer_fewer_than_all() -> None:
    f = filter_openai_tools(OPENAI_TOOLS, "viewer")
    names = {t["function"]["name"] for t in f}
    assert len(f) < len(OPENAI_TOOLS)
    assert names.issubset(ALL_TOOL_NAMES)
    assert "data_quality_check" not in names


def test_sql_violates_for_viewer_on_disallowed_table() -> None:
    pol = get_policy("viewer")
    d = sql_violates_table_policy(
        pol,
        'SELECT * FROM "some_other_table" LIMIT 1',
        "viewer",
    )
    assert d is not None
    assert d.get("error_kind") == "forbidden_table"


def test_sql_allows_viewer_on_demographics() -> None:
    pol = get_policy("viewer")
    assert (
        sql_violates_table_policy(
            pol,
            "SELECT COUNT(*) FROM demographics LIMIT 1",
            "viewer",
        )
        is None
    )


def test_extract_sql_table_refs_join() -> None:
    sql = "SELECT 1 AS a FROM demographics d JOIN mx_events m ON 1=1 LIMIT 1"
    refs = extract_sql_table_refs(sql)
    assert "demographics" in refs
    assert "mx_events" in refs


def test_forbidden_table_payload_json_serializable() -> None:
    from tools.permissions import forbidden_table_payload

    d = forbidden_table_payload("x", "viewer")
    s = json.dumps(d)
    assert "forbidden" in s.lower() or "error" in s.lower()
