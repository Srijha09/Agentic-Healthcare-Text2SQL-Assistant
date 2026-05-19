"""HITL SQL approval policy."""

from tools.approval_policy import (
    check_approval_required,
    extract_limit_from_sql,
    sql_approval_key,
)
from tools.session_state import SessionState


def test_extract_limit() -> None:
    assert extract_limit_from_sql("SELECT * FROM t LIMIT 5000") == 5000


def test_approval_required_above_threshold() -> None:
    req = check_approval_required(
        "query_database",
        {"sql": "SELECT * FROM demographics LIMIT 5000"},
        enabled=True,
        min_limit=1000,
    )
    assert req is not None
    assert req.limit_value == 5000


def test_approval_skipped_below_threshold() -> None:
    req = check_approval_required(
        "query_database",
        {"sql": "SELECT * FROM demographics LIMIT 10"},
        enabled=True,
        min_limit=1000,
    )
    assert req is None


def test_approved_sql_skips_re_prompt() -> None:
    sql = "SELECT * FROM demographics LIMIT 9000"
    st = SessionState()
    st.mark_sql_approved(sql)
    req = check_approval_required(
        "query_database",
        {"sql": sql},
        enabled=True,
        min_limit=1000,
        approved_keys=st.approved_sql_keys,
    )
    assert req is None


def test_sql_approval_key_stable() -> None:
    a = sql_approval_key("SELECT  1  FROM t LIMIT 1")
    b = sql_approval_key("SELECT 1 FROM t LIMIT 1")
    assert a == b
