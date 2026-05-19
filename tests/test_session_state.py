"""SessionState cohort memory and context_block (no DB)."""

from agent_orchestrator import build_system_content
from tools.session_state import SessionState


def test_note_cohort_sql_truncates() -> None:
    st = SessionState()
    long_sql = "SELECT 1 " + ("x" * 5000)
    st.note_cohort_sql(long_sql)
    assert st.last_cohort_sql is not None
    assert len(st.last_cohort_sql) == st._max_cohort_sql


def test_context_block_includes_cohort_line() -> None:
    st = SessionState()
    st.note_cohort_sql("SELECT PATIENT_NUMBER FROM demographics LIMIT 10")
    block = st.context_block()
    assert "same cohort" in block
    assert "demographics" in block


def test_build_system_content_includes_cohort_when_set() -> None:
    st = SessionState()
    st.note_cohort_sql("SELECT 1 AS a FROM demographics LIMIT 1")
    content = build_system_content(st)
    assert "Session context (rolling)" in content
    assert "same cohort" in content
    assert "demographics" in content
