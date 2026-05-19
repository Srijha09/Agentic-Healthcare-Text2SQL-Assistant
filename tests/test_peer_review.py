"""Peer review module tests — mocked OpenAI only (no golden dataset, no live API)."""

from unittest.mock import MagicMock

from peer_review import build_reviewer_evidence, run_peer_review
from session_log import TurnLog


def test_build_reviewer_evidence_includes_question() -> None:
    turn = TurnLog(user="Count patients in demographics")
    text = build_reviewer_evidence(turn)
    assert "Count patients" in text
    assert "User question" in text or "user" in text.lower()


def test_run_peer_review_uses_mock_client() -> None:
    turn = TurnLog(user="test")
    turn.tool_rounds.append(
        {
            "round": 1,
            "calls": [
                {
                    "function": "query_database",
                    "arguments": {"sql": "SELECT 1 LIMIT 1"},
                    "result": '{"columns":["n"],"rows":[[1]],"total_rows":1,"truncated":false}',
                    "result_summary": "ok, rows=1",
                }
            ],
        }
    )
    client = MagicMock()
    fake_msg = MagicMock()
    fake_msg.content = "### Verdict\n**pass** — numbers align with tool output.\n"
    fake_choice = MagicMock()
    fake_choice.message = fake_msg
    resp = MagicMock()
    resp.choices = [fake_choice]
    client.chat.completions.create.return_value = resp

    out = run_peer_review(client, turn, "There is 1 row.")
    assert "Verdict" in out
    assert client.chat.completions.create.called
    call_kw = client.chat.completions.create.call_args
    assert call_kw.kwargs.get("temperature") == 0.15
    assert "messages" in call_kw.kwargs
