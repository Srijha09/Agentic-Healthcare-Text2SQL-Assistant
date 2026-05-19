"""Orchestrator unit tests (mocked OpenAI, no DB)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent_orchestrator import (
    TurnResult,
    _finalize_turn_sync,
    rank_suggested_followups,
)
from orchestrator_config import OrchestratorFlags
from session_log import SessionLog
from tools.session_state import SessionState


def _msg(content: str | None = None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


class _FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def create(self, **kwargs):
        text = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )


def test_finalize_turn_applies_peer_review_notice(monkeypatch) -> None:
    monkeypatch.setenv("DISABLE_CHART_NARRATIVE", "1")
    monkeypatch.setenv("DISABLE_REPORT_WRITER", "1")
    monkeypatch.setenv("DISABLE_FOLLOWUP_RANKING", "1")
    flags = OrchestratorFlags.from_env()
    client = MagicMock()
    client.chat.completions = _FakeCompletions(
        ["### Verdict\n**concerns** — Mismatch.\n\n### Checks\n- x\n"]
    )
    log = SessionLog()
    log.start_turn("Count patients")
    log.set_assistant("There are 99 patients.")
    result = TurnResult(assistant_text="There are 99 patients.")
    _finalize_turn_sync(
        client,
        result,
        session_state=SessionState(),
        session_log=log,
        user_input="Count patients",
        flags=flags,
        last_successful_chart_result=None,
    )
    assert result.peer_review_verdict == "concerns"
    assert result.assistant_text is not None
    assert result.assistant_text.startswith(">")


def test_rank_suggested_followups_top_k() -> None:
    st = SessionState()
    st.set_last_user_request("Count by PATIENT_STATE")
    st.note_table("demographics")
    log = SessionLog()
    text = (
        "Done.\n\n### Suggested follow-ups\n"
        "- Stratify by PATIENT_STATE on demographics.\n"
        "- Compare two years on mx_events.\n"
        "- Dig deeper into the data.\n"
        "- Segment by sex from demographics.\n"
        "- Trend metformin fills monthly.\n"
        "- Explore more patterns.\n"
    )
    out = rank_suggested_followups(text, session_state=st, session_log=log, top_k=3)
    assert out.count("\n- ") == 3
