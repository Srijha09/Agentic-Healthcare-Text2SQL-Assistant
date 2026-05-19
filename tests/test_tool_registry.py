"""Tests for tool dispatch (requires healthcare.duckdb)."""

import json
import pytest

from tools.db_query import DuckDBQuery
from tools.session_state import SessionState
from tools.tool_registry import dispatch_tool


@pytest.fixture(scope="module")
def db():
    try:
        d = DuckDBQuery()
    except FileNotFoundError:
        pytest.skip("healthcare.duckdb not present")
    yield d
    d.close()


def test_list_tables(db: DuckDBQuery) -> None:
    raw = dispatch_tool("list_tables", {}, db, SessionState())
    d = json.loads(raw)
    assert "tables" in d
    assert "demographics" in d["tables"]


def test_query_database_cached_twice_same_result(db: DuckDBQuery) -> None:
    from tools import query_cache

    query_cache.cache_clear()
    sql = "SELECT 1 AS x FROM demographics LIMIT 1"
    st = SessionState()
    a = json.loads(dispatch_tool("query_database", {"sql": sql}, db, st))
    b = json.loads(dispatch_tool("query_database", {"sql": sql}, db, st))
    assert a.get("columns") == b.get("columns")
    assert a.get("total_rows") == b.get("total_rows")


def test_summarize_sql_stats_single_numeric_column(db: DuckDBQuery) -> None:
    sql = "SELECT PROVIDER_BILLED AS amt FROM mx_events LIMIT 500"
    raw = dispatch_tool(
        "summarize_sql_stats",
        {"sql": sql},
        db,
        SessionState(),
    )
    d = json.loads(raw)
    assert "stats" in d
    assert "mean" in d["stats"]
    assert d["stats"]["column"] == "amt"


def test_summarize_sql_stats_unknown_column(db: DuckDBQuery) -> None:
    sql = "SELECT 1 AS a, 2 AS b FROM demographics LIMIT 1"
    raw = dispatch_tool(
        "summarize_sql_stats",
        {"sql": sql, "value_column": "nope"},
        db,
        SessionState(),
    )
    d = json.loads(raw)
    assert "error" in d
    assert d.get("error_kind") == "summarize_bad_column"


def test_query_database_success_sets_last_cohort_sql(db: DuckDBQuery) -> None:
    from tools import query_cache

    query_cache.cache_clear()
    sql = "SELECT PATIENT_NUMBER FROM demographics LIMIT 3"
    st = SessionState()
    dispatch_tool("query_database", {"sql": sql}, db, st)
    assert st.last_cohort_sql == sql


def test_query_database_error_does_not_set_cohort(db: DuckDBQuery) -> None:
    st = SessionState()
    raw = dispatch_tool(
        "query_database",
        {"sql": "SELECT no_such_column FROM demographics LIMIT 1"},
        db,
        st,
    )
    assert "error" in json.loads(raw)
    assert st.last_cohort_sql is None


def test_describe_table_returns_schema(db: DuckDBQuery) -> None:
    raw = dispatch_tool("describe_table", {"table_name": "demographics"}, db, SessionState())
    d = json.loads(raw)
    assert d.get("table") == "demographics"
    assert "schema" in d
    assert len(d["schema"]) >= 1


def test_data_quality_check_demographics(db: DuckDBQuery) -> None:
    raw = dispatch_tool(
        "data_quality_check",
        {"table_name": "demographics", "sample_limit": 200},
        db,
        SessionState(),
    )
    d = json.loads(raw)
    assert d.get("table") == "demographics"
    assert d.get("sample_rows", 0) > 0
    assert "columns" in d and len(d["columns"]) >= 1
    c0 = d["columns"][0]
    assert "null_rate" in c0
    assert "distinct_count_non_null" in c0
    assert "outliers" in c0


def test_data_quality_check_flag_null_threshold(db: DuckDBQuery) -> None:
    raw = dispatch_tool(
        "data_quality_check",
        {"table_name": "demographics", "sample_limit": 500, "flag_null_rate_above": 0.5},
        db,
        SessionState(),
    )
    d = json.loads(raw)
    assert "error" not in d
    assert isinstance(d["threshold_hits"], list)


def test_dispatch_unknown_tool_returns_error(db: DuckDBQuery) -> None:
    raw = dispatch_tool("not_a_registered_tool", {}, db, None)
    d_out = json.loads(raw)
    assert "error" in d_out
    assert "Unknown tool" in d_out["error"]
    assert d_out.get("error_kind") == "unknown_tool"

def test_analyze_care_gap_diabetes_lab_proxy(db: DuckDBQuery) -> None:
    raw = dispatch_tool(
        "analyze_care_gap",
        {"gap_type": "diabetes_lab_utilization_proxy"},
        db,
        SessionState(),
    )
    d = json.loads(raw)
    assert "error" not in d
    assert d.get("gap_type") == "diabetes_lab_utilization_proxy"
    assert "columns" in d and "rows" in d


def test_data_quality_forbidden_for_viewer_role(db: DuckDBQuery) -> None:
    raw = dispatch_tool(
        "data_quality_check",
        {"table_name": "demographics"},
        db,
        SessionState(),
        user_role="viewer",
    )
    d = json.loads(raw)
    assert d.get("error_kind") == "forbidden_tool"


def test_list_tables_respects_viewer_table_filter(db: DuckDBQuery) -> None:
    full = set(json.loads(dispatch_tool("list_tables", {}, db, None))["tables"])
    v = json.loads(dispatch_tool("list_tables", {}, db, None, user_role="viewer"))["tables"]
    assert set(v).issubset(full)
    for t in v:
        assert t.lower() in ("demographics", "mx_events")
