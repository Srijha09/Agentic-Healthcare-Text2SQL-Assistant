"""Heuristic SQL error classification (no golden dataset)."""

from tools.sql_error_hints import build_tool_error_payload, structured_tool_error


def test_timeout_kind() -> None:
    d = build_tool_error_payload("Query exceeded time limit (60s).", timeout=True)
    assert d["error_kind"] == "timeout"
    assert "next_step" in d


def test_interrupt_kind() -> None:
    d = build_tool_error_payload("Query interrupted after 60s (wall-clock limit or cancellation).", interrupt=True)
    assert d["error_kind"] == "interrupted"
    assert "next_step" in d


def test_policy_kind() -> None:
    d = build_tool_error_payload("SELECT must include LIMIT.", policy=True)
    assert d["error_kind"] == "policy"


def test_catalog_duckdb_style() -> None:
    d = build_tool_error_payload('Catalog Error: Table with name "foo" does not exist!')
    assert d["error_kind"] == "catalog"
    assert "list_tables" in d["next_step"]


def test_syntax_kind() -> None:
    d = build_tool_error_payload("Parser Error: syntax error at or near")
    assert d["error_kind"] == "syntax"


def test_ambiguous_column() -> None:
    d = build_tool_error_payload("Binder Error: Ambiguous column reference col")
    assert d["error_kind"] == "ambiguous_column"


def test_type_cast() -> None:
    d = build_tool_error_payload("Conversion Error: Could not convert string to INTEGER")
    assert d["error_kind"] == "type_or_cast"


def test_structured_tool_error() -> None:
    d = structured_tool_error("bad", error_kind="x", next_step="y")
    assert d == {"error": "bad", "error_kind": "x", "next_step": "y"}
