"""Ranked follow-up post-processing helpers (no DB)."""

from agent_orchestrator import (
    _extract_suggested_followup_candidates,
    rank_suggested_followups,
)
from session_log import SessionLog
from tools.session_state import SessionState


def test_extract_suggested_followup_candidates() -> None:
    text = (
        "Main answer.\n\n"
        "### Suggested follow-ups\n"
        "- First idea\n"
        "- Second idea\n"
        "- Third idea\n"
    )
    prefix, bullets, suffix = _extract_suggested_followup_candidates(text)
    assert "Main answer" in prefix
    assert bullets == ["First idea", "Second idea", "Third idea"]
    assert suffix == ""


def test_rank_suggested_followups_prefers_relevant_and_novel() -> None:
    st = SessionState()
    st.set_last_user_request("Break down paid amounts for demographics cohort by PATIENT_STATE.")
    st.note_table("demographics")
    st.note_table("mx_events")

    log = SessionLog()
    # Prior turn with a duplicate-style follow-up to penalize novelty.
    log.start_turn("prior")
    log.set_assistant(
        "Done.\n\n### Suggested follow-ups\n- Compare monthly trend by PATIENT_STATE on demographics."
    )
    # Current turn (excluded from prior scan via turns[:-1]).
    log.start_turn("current")

    text = (
        "Answer body.\n\n"
        "### Suggested follow-ups\n"
        "- Explore more.\n"
        "- Compare monthly trend by PATIENT_STATE on demographics.\n"
        "- Stratify PLAN_PAID by PATIENT_STATE using demographics and mx_events.\n"
        "- Compare two time windows for PLAN_PAID in mx_events by state.\n"
        "- Additional analysis could be done.\n"
        "- Segment by sex and age from demographics and compare distributions.\n"
    )
    out = rank_suggested_followups(text, session_state=st, session_log=log, top_k=3)
    assert out.count("\n- ") == 3
    assert "Stratify PLAN_PAID by PATIENT_STATE" in out
    assert "Compare two time windows for PLAN_PAID" in out
    assert "Explore more" not in out
