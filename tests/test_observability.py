"""Observability: logging context and span helpers."""

import json
import logging

import pytest

from observability import (
    configure_logging,
    current_turn_id,
    current_trace_id,
    log_tool_result,
    start_turn,
    trace_span,
)


@pytest.fixture
def log_capture():
    configure_logging(level="DEBUG", log_format="json")
    logger = logging.getLogger("agent.tool")
    records: list[logging.LogRecord] = []

    class _H(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    h = _H()
    logger.addHandler(h)
    logger.setLevel(logging.DEBUG)
    yield records
    logger.removeHandler(h)


def test_start_turn_sets_context() -> None:
    configure_logging(level="INFO", log_format="text")
    tid = start_turn("How many patients?")
    assert len(tid) == 10
    assert current_turn_id() == tid
    assert current_trace_id() != "-"


def test_trace_span_logs_ok_and_fail() -> None:
    configure_logging(level="DEBUG", log_format="text")
    start_turn("test")
    with trace_span("unit.test"):
        pass
    with pytest.raises(RuntimeError):
        with trace_span("unit.fail"):
            raise RuntimeError("boom")


def test_log_tool_result_error_kind(log_capture: list) -> None:
    start_turn("q")
    err_json = json.dumps(
        {"error": "bad sql", "error_kind": "policy", "next_step": "fix limit"}
    )
    log_tool_result("query_database", err_json, duration_ms=12.5)
    assert log_capture
    msg = log_capture[-1].getMessage()
    assert "tool failed" in msg.lower() or "query_database" in msg


def test_log_tool_result_success(log_capture: list) -> None:
    start_turn("q")
    ok_json = json.dumps({"columns": ["n"], "rows": [[1]], "total_rows": 1, "truncated": False})
    log_tool_result("query_database", ok_json, duration_ms=3.0)
    assert any("tool ok" in r.getMessage().lower() for r in log_capture)
