"""Peer review verdict parsing and user notices."""

from peer_review import (
    apply_peer_review_notice,
    parse_peer_review_verdict,
    peer_review_notice_for_verdict,
)


def test_parse_peer_review_verdict_pass() -> None:
    md = "### Verdict\n**pass** — Numbers match tool output.\n"
    assert parse_peer_review_verdict(md) == "pass"


def test_parse_peer_review_verdict_concerns() -> None:
    md = "### Verdict\n**concerns** — Count not in evidence.\n"
    assert parse_peer_review_verdict(md) == "concerns"


def test_apply_peer_review_notice_prepends_for_concerns() -> None:
    out = apply_peer_review_notice("There are 10 patients.", "concerns")
    assert out.startswith(">")
    assert "concerns" in out.lower()
    assert "10 patients" in out


def test_peer_review_notice_none_for_pass() -> None:
    assert peer_review_notice_for_verdict("pass") is None
