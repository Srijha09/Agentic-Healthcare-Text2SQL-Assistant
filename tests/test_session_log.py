"""Session export redaction and markdown shape."""

import json

from session_log import SessionLog


def test_redact_patient_number_in_tool_json() -> None:
    log = SessionLog()
    log.start_turn("test")
    log.set_planner_phase("plan")
    payload = json.dumps(
        {
            "columns": ["PATIENT_NUMBER", "n"],
            "rows": [[12345678901, 5]],
            "total_rows": 1,
            "truncated": False,
        }
    )
    log.add_tool_round(1, [("query_database", {"sql": "SELECT 1 LIMIT 1"}, payload)], assistant_reasoning=None)
    log.set_assistant("done")
    md = log.to_markdown()
    assert "[redacted]" in md or "redacted" in md.lower()
    assert "12345678901" not in md


def test_markdown_has_session_export_header() -> None:
    log = SessionLog()
    log.set_repro_metadata({"model": "gpt-4o"})
    log.start_turn("hello")
    log.set_assistant("reply")
    md = log.to_markdown()
    assert "Session export" in md
    assert "hello" in md
