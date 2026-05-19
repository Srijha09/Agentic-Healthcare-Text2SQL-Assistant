"""Session stats / digest from SessionLog (no LLM)."""

from session_log import SessionLog, tables_referenced_in_sql


def test_tables_referenced_in_sql_basic() -> None:
    assert "demographics" in tables_referenced_in_sql('SELECT * FROM demographics LIMIT 1')
    assert "mx_events" in tables_referenced_in_sql(
        'SELECT * FROM mx_events m JOIN demographics d ON m."PATIENT_NUMBER" = d."PATIENT_NUMBER" LIMIT 1'
    )


def test_compute_session_stats() -> None:
    log = SessionLog()
    log.start_turn("How many rows?")
    log.add_tool_round(
        1,
        [
            (
                "query_database",
                {"sql": "SELECT COUNT(*) AS n FROM demographics LIMIT 1"},
                '{"columns":["n"],"rows":[[10]],"total_rows":1,"truncated":false}',
            )
        ],
        assistant_reasoning=None,
    )
    log.set_assistant("There are 10 patients.")
    st = log.compute_session_stats()
    assert st["turns"] == 1
    assert st["sql_executions"] == 1
    assert "demographics" in st["distinct_tables"]
    assert st["tool_errors"] == 0


def test_to_markdown_has_executive_and_digest() -> None:
    log = SessionLog()
    log.start_turn("x")
    log.add_tool_round(
        1,
        [
            (
                "query_database",
                {"sql": "SELECT 1 LIMIT 1"},
                '{"error":"bad","error_kind":"other","next_step":"fix"}',
            )
        ],
        None,
    )
    log.set_assistant("oops")
    md = log.to_markdown()
    assert "Executive summary" in md
    assert "Session digest" in md
    assert "sql_executions" in md.lower() or "SQL executions" in md
