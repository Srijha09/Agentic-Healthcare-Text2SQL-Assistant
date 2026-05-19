"""
Second-pass peer reviewer: checks assistant answer against tool evidence (no extra DB calls).

Invoked automatically after each successful agent turn from `agent_orchestrator.run_user_turn`.
Output is markdown for the Streamlit **Reviewer Evaluation** expander and session export — not a
replacement for the primary answer.
"""

from __future__ import annotations

import json
import re
from openai import AsyncOpenAI, OpenAI

from session_log import TurnLog

_DEFAULT_REVIEWER_MODEL = "gpt-4o"

REVIEWER_SYSTEM = """You are an independent peer reviewer for a synthetic healthcare **analytics** assistant (not clinical advice).

You receive: the user's question, the planner outline (if any), tool names and **truncated** tool outputs (SQL results, charts paths, errors), and the assistant's final markdown answer.

Your job:
1. Check whether **numeric claims and cohort statements** in the final answer are **consistent with** what appears in the tool outputs. Flag if numbers in prose do not appear in evidence or contradict it.
2. Note **obvious** SQL/schema risks only from what is shown (e.g. missing JOIN if the question required multi-table logic and only one table appears) — do not invent schema.
3. If tools returned **errors** or **truncated** results, verify the answer acknowledges that.
4. Stay concise. Do not restate the full answer.

Respond in **markdown** with these sections exactly:
### Verdict
One line: **pass**, **review_needed**, or **concerns** with a short reason.

### Checks
A short bullet list (✓ / ⚠ / ✗) for: evidence alignment, truncation/errors acknowledged, cohort/SQL plausibility (only if visible).

### Risks
Bullet list of specific risks or **None worth noting**.

### Suggestions
1–3 concrete next steps for the analyst (e.g. validate with a second query), or **None**.

Rules: Only reference facts present in the evidence package. If unsure, say so. Synthetic data only."""


_MAX_PLANNER = 3500
_MAX_RESULT_SNIP = 2200


def _truncate(s: str, n: int) -> str:
    s = s.strip()
    if len(s) <= n:
        return s
    return s[: n - 30] + "\n… [truncated for reviewer]"


def build_reviewer_evidence(turn: TurnLog) -> str:
    """Serialize one turn's tool evidence for the reviewer prompt."""
    parts: list[str] = [f"## User question\n{turn.user}\n"]
    if turn.planner_phase:
        parts.append(f"\n## Planner\n{_truncate(turn.planner_phase, _MAX_PLANNER)}\n")
    if not turn.tool_rounds:
        parts.append("\n## Tool rounds\n_(no tool calls in this turn — answer may be purely conversational)_\n")
    else:
        parts.append("\n## Tool rounds\n")
        for tr in turn.tool_rounds:
            parts.append(f"\n### Round {tr.get('round', '?')}\n")
            if tr.get("assistant_reasoning_with_tools"):
                parts.append(
                    f"_Assistant reasoning (with tools):_ {_truncate(tr['assistant_reasoning_with_tools'], 800)}\n"
                )
            for call in tr.get("calls", []):
                fn = call.get("function", "?")
                summary = call.get("result_summary", "")
                raw = call.get("result", "")
                snip = _truncate(raw, _MAX_RESULT_SNIP)
                parts.append(f"- **`{fn}`** — {summary}\n```\n{snip}\n```\n")
    return "".join(parts)


def run_peer_review(
    client: OpenAI,
    turn: TurnLog,
    final_answer: str,
    *,
    model: str | None = None,
) -> str:
    """Single no-tools completion; returns markdown for UI and export."""
    evidence = build_reviewer_evidence(turn)
    user_content = (
        f"{evidence}\n\n---\n\n## Final assistant answer (to review)\n\n{final_answer}\n"
    )
    use_model = model or _DEFAULT_REVIEWER_MODEL
    resp = client.chat.completions.create(
        model=use_model,
        messages=[
            {"role": "system", "content": REVIEWER_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        temperature=0.15,
        max_tokens=900,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        return "_(Peer review returned empty content.)_"
    return text


async def run_peer_review_async(
    client: AsyncOpenAI,
    turn: TurnLog,
    final_answer: str,
    *,
    model: str | None = None,
) -> str:
    """Async version of run_peer_review using AsyncOpenAI."""
    evidence = build_reviewer_evidence(turn)
    user_content = (
        f"{evidence}\n\n---\n\n## Final assistant answer (to review)\n\n{final_answer}\n"
    )
    use_model = model or _DEFAULT_REVIEWER_MODEL
    resp = await client.chat.completions.create(
        model=use_model,
        messages=[
            {"role": "system", "content": REVIEWER_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        temperature=0.15,
        max_tokens=900,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        return "_(Peer review returned empty content.)_"
    return text


_VERDICT_RE = re.compile(
    r"###\s*Verdict\s*\n\s*\*{0,2}(pass|review_needed|concerns)\*{0,2}",
    re.IGNORECASE,
)


def parse_peer_review_verdict(md: str) -> str:
    """
    Extract verdict from reviewer markdown: pass | review_needed | concerns | unknown.
    """
    if not md or not str(md).strip():
        return "unknown"
    m = _VERDICT_RE.search(md)
    if m:
        return m.group(1).lower()
    low = md.lower()
    if "**pass**" in low or "verdict\npass" in low:
        return "pass"
    if "review_needed" in low:
        return "review_needed"
    if "**concerns**" in low or "verdict\nconcerns" in low:
        return "concerns"
    return "unknown"


def peer_review_notice_for_verdict(verdict: str) -> str | None:
    """User-visible banner when review is not a clean pass."""
    if verdict == "pass":
        return None
    if verdict == "review_needed":
        return (
            "> **Reviewer: review needed** — The answer may not fully match tool evidence. "
            "See **Reviewer Evaluation** below and verify key numbers with SQL."
        )
    if verdict == "concerns":
        return (
            "> **Reviewer: concerns** — Possible mismatch with tool results or unacknowledged limits. "
            "Treat numbers as provisional until you confirm with the cited queries."
        )
    return (
        "> **Reviewer: inconclusive** — Peer review did not return a clear pass verdict. "
        "Check **Reviewer Evaluation** before relying on this answer."
    )


def apply_peer_review_notice(answer: str, verdict: str) -> str:
    """Prepend a short notice to the assistant answer when verdict is not pass."""
    notice = peer_review_notice_for_verdict(verdict)
    if not notice:
        return answer
    if notice.split("\n", 1)[0] in (answer or ""):
        return answer
    return f"{notice}\n\n{answer}"


def peer_review_json_fallback(md: str) -> dict | None:
    """Optional: extract JSON if the model returns a code block — not required for v1."""
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", md)
    if not m:
        return None
    try:
        return json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        return None
