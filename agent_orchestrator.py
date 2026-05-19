"""
Shared agent loop: planner → multi-round tools → final answer → peer review.

Used by `chat.py` (terminal) and `streamlit_app.py` (web). The system message is rebuilt
inside `run_user_turn` so rolling `SessionState` stays aligned; callers should refresh
`messages[0]` after a turn if they reuse the list for the next user message.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from observability import (
    end_turn,
    get_logger,
    start_turn,
    trace_span,
)
from orchestrator_config import OrchestratorFlags
from peer_review import (
    apply_peer_review_notice,
    parse_peer_review_verdict,
    run_peer_review,
    run_peer_review_async,
)
from session_log import SessionLog
from tools.db_query import DuckDBQuery
from tools.permissions import filter_openai_tools, normalize_role
from tools.session_state import SessionState
from tools.context_manager import needs_pruning, prune_messages
from tools.approval_policy import (
    ApprovalCheckpoint,
    ApprovalRequest,
    approval_enabled_from_env,
    first_call_needing_approval,
)
from tools.tool_registry import OPENAI_TOOLS, async_dispatch_tool, dispatch_tool

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o")
MAX_SQL_FAILURES_PER_TURN = 2
MAX_TOOL_ROUNDS_PER_TURN = 28

_LOG = get_logger("agent.orchestrator")


def _format_orchestrator_exception(exc: Exception) -> str:
    """Readable top-level error with a coarse tag (OpenAI/network; not tool JSON)."""
    if isinstance(exc, RateLimitError):
        return f"[api_rate_limit] {exc}\nNext: Wait briefly and retry, or reduce request rate."
    if isinstance(exc, APIConnectionError):
        return f"[api_network] {exc}\nNext: Check network, VPN, and proxy; retry."
    if isinstance(exc, APITimeoutError):
        return f"[api_timeout] {exc}\nNext: Retry with fewer tool rounds or shorter prompts."
    if isinstance(exc, AuthenticationError):
        return f"[api_auth] {exc}\nNext: Set a valid OPENAI_API_KEY."
    if isinstance(exc, BadRequestError):
        return f"[api_bad_request] {exc}\nNext: Inspect the error body; shorten content or fix tool arguments."
    return f"[unexpected] {exc}\nNext: Retry; check logs if the issue continues."

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
# Session markdown audit exports (terminal `export` / exit, and parity with Streamlit downloads)
REPORTS_DIR = OUTPUTS_DIR / "reports"

SYSTEM_PROMPT_BASE = """You are an analyst assistant for a synthetic healthcare claims dataset in DuckDB.
If the user thread already has a short numbered plan from before tools, use it as a guide; ground answers in tool output.

Workflow:
1. Clarify what the user needs. If the schema is unclear, use list_tables, describe_table, table_info, or profile_table before writing SQL.
2. Use query_database to retrieve facts. Write correct, efficient SQL (JOINs on PATIENT_NUMBER). Every SELECT that uses FROM must include LIMIT n (enforced by the tool).
3. For numeric distributions (min/max/mean/quartiles) on a single metric column, prefer summarize_sql_stats with the same guarded SQL; set value_column when multiple numeric columns are returned.
4. For charts, use create_chart with read-only SQL (two columns, LIMIT on SELECT...FROM), a descriptive title (cohort, metric, time or breakdown), and chart_type. Same SQL guard as query_database. Saves a PNG under outputs/visualization/; paste markdown_embed in your answer.
5. After tool results return, answer in plain language. Cite what you computed. If a tool returned an error or truncated rows, say so and suggest a follow-up query.
6. - For care-gap style questions (missing follow-up lab utilization on claims), prefer analyze_care_gap with gap_type diabetes_lab_utilization_proxy when it matches E11 + LAB visit-type story; otherwise use query_database.

SQL errors (self-correction):
- If query_database returns an error, read the message, fix the SQL, and try again.
- If you see retry_limit_reached in a tool result, stop issuing query_database for this user question: explain the issue and suggest schema checks or a simpler query. The user can ask a new question to continue.

Follow-up UX:
- When you give a final natural-language answer (no more tool calls needed), include a section titled exactly "### Suggested follow-ups" with **6 candidate** bullet lines.
- Candidate bullets must (a) reference the **same cohort, metric, or question** you just answered or the **Latest user request** in session context, and (b) propose a **specific next step** — include at least one candidate for **stratification** (e.g. by `PATIENT_STATE`, year/month, or sex from demographics) and at least one for **comparison** (e.g. two time windows) using tables you already named.
- Do not use generic bullets ("dig deeper", "explore more") without naming tables, columns, or filters.

Rules:
- Do not invent patient counts, rates, or IDs; only state what tools returned.
- The database is read-only; you cannot modify data.
- Protect privacy: avoid unnecessary row-level exports; prefer aggregates for summaries.
- If **Session context** includes a cohort memory line, when the user says "same cohort", "that cohort", or similar without new filters, reuse that SQL (subquery / JOIN) as the cohort definition."""

PLANNER_SYSTEM = """You outline analysis steps only (synthetic data; not clinical advice).
Give a numbered plan (4-7 steps): tables, joins on PATIENT_NUMBER, filters. No SQL; no made-up counts.
The next step in the app will run database tools."""

ANALYST_SYSTEM = """You are the Data Analyst Agent (SQL + stats planning only).
Return strict JSON with keys:
- "plan_markdown": short numbered plan (4-7 steps), no SQL, no made-up numbers.
- "prioritize_visualization": boolean, true if the user likely needs a chart.
- "visualization_hint": short string with best chart direction (or empty).

Rules: synthetic analytics only; join keys on PATIENT_NUMBER when relevant.
Return JSON only."""

VIZ_SYSTEM = """You are the Visualization Agent.
Given user question + proposed create_chart args + analyst hint, decide if create_chart should run now.
Return strict JSON with keys:
- "allow": boolean
- "reason": short reason
- "title_override": optional string (or empty)

Allow when chart request is coherent with user intent and title/chart_type are sensible.
Return JSON only."""

REPORT_WRITER_SYSTEM = """You are the Report Writer Agent.
Rewrite the draft answer for clarity and structure while preserving facts from evidence.
Do not invent numbers or tables not in evidence. Keep markdown.
Ensure there is a "### Suggested follow-ups" section with ~6 concrete candidate bullets (not generic).
Return markdown only."""


def _safe_json_dict(text: str) -> dict[str, Any]:
    try:
        d = json.loads((text or "").strip())
        return d if isinstance(d, dict) else {}
    except json.JSONDecodeError:
        return {}


def _run_data_analyst_agent(
    client: OpenAI,
    user_input: str,
    state: SessionState,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    payload = (
        f"User request:\n{user_input}\n\n"
        f"Session context:\n{state.context_block() or '(none)'}\n"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ANALYST_SYSTEM},
            {"role": "user", "content": payload},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = (resp.choices[0].message.content or "").strip()
    d = _safe_json_dict(text)
    return {
        "plan_markdown": str(d.get("plan_markdown", "")).strip(),
        "prioritize_visualization": bool(d.get("prioritize_visualization", False)),
        "visualization_hint": str(d.get("visualization_hint", "")).strip(),
    }


def _run_visualization_agent(
    client: OpenAI,
    *,
    user_input: str,
    function_args: dict[str, Any],
    analyst_intent: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    payload = (
        f"User request:\n{user_input}\n\n"
        f"Proposed create_chart args:\n{json.dumps(function_args, ensure_ascii=True)}\n\n"
        f"Analyst hint:\n{json.dumps(analyst_intent, ensure_ascii=True)}\n"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": VIZ_SYSTEM},
            {"role": "user", "content": payload},
        ],
        temperature=0.0,
        max_tokens=220,
    )
    d = _safe_json_dict((resp.choices[0].message.content or "").strip())
    return {
        "allow": bool(d.get("allow", True)),
        "reason": str(d.get("reason", "")).strip(),
        "title_override": str(d.get("title_override", "")).strip(),
    }


def _run_report_writer_agent(
    client: OpenAI,
    *,
    user_input: str,
    draft_answer: str,
    planner_text: str | None,
    tool_events: list[str],
    model: str,
) -> str:
    evidence = (
        f"User request:\n{user_input}\n\n"
        f"Planner/Analyst plan:\n{planner_text or '(none)'}\n\n"
        f"Tool trace:\n{chr(10).join(tool_events) if tool_events else '(none)'}\n\n"
        f"Draft answer:\n{draft_answer}\n"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": REPORT_WRITER_SYSTEM},
            {"role": "user", "content": evidence},
        ],
        temperature=0.2,
        max_tokens=1200,
    )
    return (resp.choices[0].message.content or "").strip()


def session_repro_metadata(db: DuckDBQuery | None) -> dict:
    flags = OrchestratorFlags.from_env()
    meta: dict = {
        "model": flags.chat_model,
        "disable_planner": "1" if flags.planner_disabled else "0",
        "disable_report_writer": "1" if flags.report_writer_disabled else "0",
        "disable_viz_agent": "1" if flags.viz_agent_disabled else "0",
        "disable_chart_narrative": "1" if flags.chart_narrative_disabled else "0",
        "disable_peer_review": "1" if flags.peer_review_disabled else "0",
        "peer_review_model": flags.peer_review_model or CHAT_MODEL,
    }
    try:
        meta["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        meta["git_head"] = "(not a git checkout or git unavailable)"
    db_path = PROJECT_ROOT / "healthcare.duckdb"
    if db_path.exists():
        st = db_path.stat()
        meta["database_file"] = db_path.name
        meta["database_size_bytes"] = st.st_size
        meta["database_mtime"] = datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
    else:
        meta["database_file"] = "(file missing)"
    if db is not None:
        meta["database_path_resolved"] = str(db.db_path.resolve())
    return meta


def build_system_content(state: SessionState) -> str:
    block = state.context_block()
    if block:
        return SYSTEM_PROMPT_BASE + "\n\n### Session context (rolling)\n" + block
    return SYSTEM_PROMPT_BASE


def query_tool_result_has_error(result_str: str) -> bool:
    try:
        d = json.loads(result_str)
        return isinstance(d, dict) and "error" in d
    except json.JSONDecodeError:
        return False


_FOLLOWUPS_HEADER = "### Suggested follow-ups"
_BULLET_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)(.+?)\s*$")
_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{2,}")


def _tokenize_text(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")}


def _extract_suggested_followup_candidates(text: str) -> tuple[str, list[str], str]:
    """
    Split answer into (prefix, candidate bullets, suffix) around the Suggested follow-ups section.
    If section not found, returns (text, [], "").
    """
    if not text or _FOLLOWUPS_HEADER not in text:
        return text, [], ""
    head_idx = text.find(_FOLLOWUPS_HEADER)
    prefix = text[:head_idx].rstrip()
    rest = text[head_idx + len(_FOLLOWUPS_HEADER):]
    lines = rest.splitlines()
    bullets: list[str] = []
    suffix_start = len(lines)
    saw_candidate_lines = False
    for i, line in enumerate(lines):
        m = _BULLET_RE.match(line)
        if m:
            bullets.append(m.group(1).strip())
            saw_candidate_lines = True
            continue
        if line.strip() == "" and not saw_candidate_lines:
            continue
        if saw_candidate_lines and line.strip() == "":
            # Allow blank lines between bullets
            continue
        if saw_candidate_lines:
            suffix_start = i
            break
    suffix = "\n".join(lines[suffix_start:]).strip()
    return prefix, bullets, suffix


def _collect_prior_followups(session_log: SessionLog) -> list[str]:
    prior: list[str] = []
    for turn in session_log.turns[:-1]:
        a = (turn.assistant or "").strip()
        if not a:
            continue
        _, bullets, _ = _extract_suggested_followup_candidates(a)
        prior.extend(bullets)
    return prior


def _score_followup_candidate(
    bullet: str,
    *,
    user_request_tokens: set[str],
    table_tokens: set[str],
    prior_followups: list[str],
) -> float:
    low = bullet.lower()
    b_tokens = _tokenize_text(bullet)
    score = 0.0

    # Dataset relevance to recent tables / user ask.
    if table_tokens and b_tokens.intersection(table_tokens):
        score += 2.0
    if user_request_tokens and b_tokens.intersection(user_request_tokens):
        score += 1.0

    # Analytical depth markers.
    if any(k in low for k in (" by ", " stratif", "group by", "segment", "cohort")):
        score += 1.0
    if any(k in low for k in (" compare", " vs ", "versus", "difference", "trend", "month", "year", "time window")):
        score += 1.0
    if any(k in low for k in ("state", "sex", "age", "visit_type", "patient_state")):
        score += 0.5

    # Penalize generic follow-ups.
    if any(k in low for k in ("dig deeper", "explore more", "learn more", "additional analysis")):
        score -= 2.0

    # Novelty relative to prior turns.
    for prev in prior_followups:
        p_low = prev.lower()
        if low == p_low:
            score -= 3.0
            continue
        p_tokens = _tokenize_text(prev)
        if not p_tokens:
            continue
        overlap = len(b_tokens.intersection(p_tokens)) / max(1, len(b_tokens.union(p_tokens)))
        if overlap >= 0.7:
            score -= 1.5
        elif overlap >= 0.5:
            score -= 0.8
    return score


def rank_suggested_followups(
    answer_text: str,
    *,
    session_state: SessionState,
    session_log: SessionLog,
    top_k: int = 3,
) -> str:
    """
    Convert over-generated candidate follow-ups into a ranked top-k section.
    Falls back to original text if section missing or no bullets.
    """
    prefix, candidates, suffix = _extract_suggested_followup_candidates(answer_text)
    if not candidates:
        return answer_text

    user_tokens = _tokenize_text(session_state.last_user_request or "")
    table_tokens = {t.lower() for t in session_state.recent_tables}
    prior = _collect_prior_followups(session_log)

    # Deduplicate candidates while preserving first occurrence.
    seen: set[str] = set()
    unique_candidates: list[str] = []
    for c in candidates:
        key = " ".join(c.lower().split())
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(c)

    ranked = sorted(
        unique_candidates,
        key=lambda b: _score_followup_candidate(
            b,
            user_request_tokens=user_tokens,
            table_tokens=table_tokens,
            prior_followups=prior,
        ),
        reverse=True,
    )
    chosen = ranked[: max(1, top_k)]
    lines = [f"- {b}" for b in chosen]
    body = f"{_FOLLOWUPS_HEADER}\n" + "\n".join(lines)
    out = f"{prefix}\n\n{body}".strip()
    if suffix:
        out = f"{out}\n\n{suffix}"
    return out

def _chart_narrative_user_prompt(chart_result: dict) -> str | None:
    """Build user prompt from chart tool JSON; None if chart_path missing."""
    chart_path = chart_result.get("chart_path")
    if not isinstance(chart_path, str) or not chart_path.strip():
        return None
    sample = chart_result.get("sample_rows")
    sample_lines = ""
    if isinstance(sample, list) and sample:
        sample_lines = "\n- sample_rows (label, value):\n" + json.dumps(
            sample[:12], ensure_ascii=True
        )
    return (
        "Chart metadata:\n"
        f"- chart_path: {chart_path}\n"
        f"- title: {chart_result.get('title')}\n"
        f"- chart_type: {chart_result.get('chart_type')}\n"
        f"- rows_plotted: {chart_result.get('rows_plotted')}\n"
        f"- columns: {chart_result.get('columns')}"
        f"{sample_lines}\n\n"
        "Output exactly one sentence describing the pattern in sample_rows when present."
    )


_CHART_NARRATIVE_SYSTEM = (
    "You are an analytics assistant. Write exactly one sentence in plain English "
    "interpreting a chart using only provided metadata and sample_rows. "
    "You may cite specific label/value pairs from sample_rows. "
    "Do not invent numbers not present in the payload."
)


def generate_chart_narrative(
    client: OpenAI,
    chart_result: dict,
    *,
    model: str | None = None,
) -> str:
    """
    Second no-tools LLM call: produce one plain-English chart interpretation sentence.
    Returns fallback text if chart_result is invalid or model returns empty content.
    """
    if not isinstance(chart_result, dict):
        return "Chart generated, but interpretation is unavailable due to invalid chart metadata."

    user_prompt = _chart_narrative_user_prompt(chart_result)
    if not user_prompt:
        return "Chart generated, but interpretation is unavailable because chart_path is missing."

    use_model = model or CHAT_MODEL
    resp = client.chat.completions.create(
        model=use_model,
        messages=[
            {"role": "system", "content": _CHART_NARRATIVE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=120,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        return "Chart generated; interpretation could not be produced."

    first_sentence = text.split("\n")[0].strip()
    if "." in first_sentence:
        first_sentence = first_sentence.split(".", 1)[0].strip() + "."
    return first_sentence


def _prune_messages_if_needed(messages: list, session_state: SessionState) -> None:
    if needs_pruning(messages):
        messages[:] = prune_messages(
            messages,
            session_digest=session_state.context_block(),
        )


def _dispatch_create_chart(
    client: OpenAI,
    *,
    function_name: str,
    function_args: dict,
    db: DuckDBQuery,
    session_state: SessionState,
    effective_role: str,
    user_input: str,
    analyst_intent: dict[str, Any],
    flags: OrchestratorFlags,
) -> str:
    """Run create_chart with optional visualization-agent gate."""
    if flags.should_run_viz_gate(analyst_intent):
        try:
            viz = _run_visualization_agent(
                client,
                user_input=user_input,
                function_args=function_args,
                analyst_intent=analyst_intent,
                model=flags.chat_model,
            )
            if not viz.get("allow", True):
                return json.dumps(
                    {
                        "error": "Visualization agent rejected create_chart for this turn.",
                        "error_kind": "viz_rejected",
                        "next_step": viz.get("reason")
                        or "Use query_database first to clarify trend dimensions, then retry create_chart.",
                    }
                )
            to_run = dict(function_args)
            t_override = viz.get("title_override", "")
            if t_override and isinstance(t_override, str):
                to_run["title"] = t_override
            return dispatch_tool(
                function_name, to_run, db, session_state, user_role=effective_role
            )
        except Exception:
            pass
    return dispatch_tool(
        function_name, function_args, db, session_state, user_role=effective_role
    )


async def _async_dispatch_create_chart(
    client: AsyncOpenAI,
    *,
    function_name: str,
    function_args: dict,
    db: DuckDBQuery,
    session_state: SessionState,
    effective_role: str,
    user_input: str,
    analyst_intent: dict[str, Any],
    flags: OrchestratorFlags,
) -> str:
    if flags.should_run_viz_gate(analyst_intent):
        try:
            viz = await _async_run_visualization_agent(
                client,
                user_input=user_input,
                function_args=function_args,
                analyst_intent=analyst_intent,
                model=flags.chat_model,
            )
            if not viz.get("allow", True):
                return json.dumps(
                    {
                        "error": "Visualization agent rejected create_chart for this turn.",
                        "error_kind": "viz_rejected",
                        "next_step": viz.get("reason")
                        or "Use query_database first to clarify trend dimensions, then retry create_chart.",
                    }
                )
            to_run = dict(function_args)
            t_override = viz.get("title_override", "")
            if t_override and isinstance(t_override, str):
                to_run["title"] = t_override
            return await async_dispatch_tool(
                function_name, to_run, db, session_state, user_role=effective_role
            )
        except Exception:
            pass
    return await async_dispatch_tool(
        function_name, function_args, db, session_state, user_role=effective_role
    )


def assistant_message_to_dict(response_message: Any) -> dict:
    if isinstance(response_message, dict):
        return response_message
    if hasattr(response_message, "model_dump"):
        return response_message.model_dump(exclude_none=True)
    # Streaming path uses SimpleNamespace to mirror ChatCompletionMessage
    if hasattr(response_message, "content") or hasattr(response_message, "tool_calls"):
        content = getattr(response_message, "content", None)
        tcs = getattr(response_message, "tool_calls", None)
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tcs:
            out: list[dict[str, Any]] = []
            for tc in tcs:
                fn = getattr(tc, "function", None)
                out.append(
                    {
                        "id": getattr(tc, "id", ""),
                        "type": getattr(tc, "type", "function"),
                        "function": {
                            "name": getattr(fn, "name", "") if fn is not None else "",
                            "arguments": getattr(fn, "arguments", "") if fn is not None else "",
                        },
                    }
                )
            msg["tool_calls"] = out
        return msg
    raise TypeError(f"Unexpected assistant message type: {type(response_message)!r}")


def _assistant_dict_to_message(msg: dict[str, Any]) -> Any:
    """Build a namespace with .content and .tool_calls for the executor loop."""
    tcs = msg.get("tool_calls")
    if not tcs:
        return SimpleNamespace(content=msg.get("content"), tool_calls=None)
    out_tc = []
    for tc in tcs:
        fn = tc.get("function") or {}
        out_tc.append(
            SimpleNamespace(
                id=tc.get("id", ""),
                type=tc.get("type", "function"),
                function=SimpleNamespace(
                    name=fn.get("name", ""),
                    arguments=fn.get("arguments", ""),
                ),
            )
        )
    return SimpleNamespace(content=msg.get("content"), tool_calls=out_tc)


def _streaming_chat_completion(
    client: OpenAI,
    api_params: dict,
    stream_delta: Callable[[str], None] | None,
) -> Any:
    """Stream one completion; return assistant message compatible with tool_calls iteration."""
    params = dict(api_params)
    params["stream"] = True
    stream = client.chat.completions.create(**params)
    content_parts: list[str] = []
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta is None:
            continue
        if getattr(delta, "content", None):
            content_parts.append(delta.content)
            if stream_delta:
                stream_delta(delta.content)
        if getattr(delta, "tool_calls", None):
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls_by_index:
                    tool_calls_by_index[idx] = {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                if tc.id:
                    tool_calls_by_index[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        tool_calls_by_index[idx]["function"]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_calls_by_index[idx]["function"]["arguments"] += tc.function.arguments or ""
    content_str = "".join(content_parts)
    if tool_calls_by_index:
        tls = []
        for i in sorted(tool_calls_by_index.keys()):
            tc = tool_calls_by_index[i]
            tls.append(
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
            )
        return _assistant_dict_to_message(
            {"role": "assistant", "content": content_str or None, "tool_calls": tls}
        )
    return _assistant_dict_to_message({"role": "assistant", "content": content_str or None})


@dataclass
class TurnResult:
    assistant_text: str | None = None
    planner_text: str | None = None
    tool_events: list[str] = field(default_factory=list)
    chart_paths: list[str] = field(default_factory=list)
    peer_review_text: str | None = None
    peer_review_verdict: str | None = None
    approval_request: ApprovalRequest | None = None
    approval_checkpoint: ApprovalCheckpoint | None = None
    error: str | None = None


def _approval_rejection_payload(reason: str) -> str:
    return json.dumps(
        {
            "error": "Query not approved by user.",
            "error_kind": "approval_denied",
            "next_step": reason or "Revise SQL (e.g. lower LIMIT) or ask the user to approve.",
        }
    )


async def _dispatch_one_tool_async(
    client: AsyncOpenAI,
    fn: str,
    args: dict,
    db: DuckDBQuery | None,
    session_state: SessionState,
    effective_role: str,
    user_input: str,
    analyst_intent: dict[str, Any],
    flags: OrchestratorFlags,
    sql_failures: int,
) -> str:
    """Execute a single tool call (shared by main loop and approval resume)."""
    with trace_span("tool.call", tool=fn, role=effective_role):
        if fn == "query_database" and sql_failures >= MAX_SQL_FAILURES_PER_TURN:
            _LOG.warning(
                "sql retry limit enforced",
                extra={
                    "event": "sql.retry_limit",
                    "failures": sql_failures,
                    "max": MAX_SQL_FAILURES_PER_TURN,
                },
            )
            return json.dumps(
                {
                    "error": "SQL retry limit reached for this user message.",
                    "retry_limit_reached": True,
                    "failures_recorded": sql_failures,
                    "hint": "Explain the error without issuing more query_database calls for this question.",
                }
            )
        if fn == "create_chart" and db is not None:
            return await _async_dispatch_create_chart(
                client,
                function_name=fn,
                function_args=args,
                db=db,
                session_state=session_state,
                effective_role=effective_role,
                user_input=user_input,
                analyst_intent=analyst_intent,
                flags=flags,
            )
        return await async_dispatch_tool(fn, args, db, session_state, user_role=effective_role)


def _log_turn_completion(result: TurnResult, tool_round: int) -> None:
    if result.approval_checkpoint is not None:
        end_turn(
            status="approval_pending",
            tool_rounds=tool_round,
            approval_pending=True,
        )
    elif result.error:
        end_turn(status="error", error=result.error, tool_rounds=tool_round)
    else:
        end_turn(
            status="ok",
            tool_rounds=tool_round,
            peer_review_verdict=result.peer_review_verdict,
        )


def _apply_chart_result_to_turn(result: TurnResult, resp: str) -> dict[str, Any] | None:
    try:
        d = json.loads(resp)
    except json.JSONDecodeError:
        return None
    p = d.get("chart_path")
    if p and isinstance(p, str) and "error" not in d:
        result.chart_paths.append(p)
        return d
    return None


def _finalize_turn_sync(
    client: OpenAI,
    result: TurnResult,
    *,
    session_state: SessionState,
    session_log: SessionLog,
    user_input: str,
    flags: OrchestratorFlags,
    last_successful_chart_result: dict[str, Any] | None,
) -> None:
    """Post-executor: chart narrative, report writer, follow-up ranking, peer review."""
    if (
        result.assistant_text
        and last_successful_chart_result is not None
        and not flags.chart_narrative_disabled
    ):
        try:
            with trace_span("finalize.chart_narrative"):
                narrative = generate_chart_narrative(
                    client,
                    last_successful_chart_result,
                    model=flags.chat_model,
                )
        except Exception as ex:
            _LOG.warning("chart narrative failed: %s", ex, extra={"event": "finalize.chart_narrative.fail"})
            narrative = f"Chart generated; interpretation unavailable ({ex})."
        result.assistant_text = f"{result.assistant_text}\n\nChart interpretation: {narrative}"
        session_log.set_assistant(result.assistant_text)

    if result.assistant_text and not flags.report_writer_disabled:
        try:
            with trace_span("finalize.report_writer"):
                rewritten = _run_report_writer_agent(
                    client,
                    user_input=user_input,
                    draft_answer=result.assistant_text,
                    planner_text=result.planner_text,
                    tool_events=result.tool_events,
                    model=flags.chat_model,
                )
            if rewritten:
                result.assistant_text = rewritten
                session_log.set_assistant(result.assistant_text)
        except Exception as ex:
            _LOG.warning("report writer failed: %s", ex, extra={"event": "finalize.report_writer.fail"})

    if result.assistant_text and not flags.followup_ranking_disabled:
        with trace_span("finalize.followup_ranking"):
            ranked_text = rank_suggested_followups(
                result.assistant_text,
                session_state=session_state,
                session_log=session_log,
            )
        if ranked_text != result.assistant_text:
            result.assistant_text = ranked_text
            session_log.set_assistant(result.assistant_text)

    if not result.error and result.assistant_text and session_log.turns and not flags.peer_review_disabled:
        review_model = flags.peer_review_model or flags.chat_model
        try:
            with trace_span("finalize.peer_review", model=review_model):
                pr = run_peer_review(
                    client,
                    session_log.turns[-1],
                    result.assistant_text,
                    model=review_model,
                )
            result.peer_review_text = pr
            session_log.set_peer_review(pr)
            verdict = parse_peer_review_verdict(pr)
            result.peer_review_verdict = verdict
            _LOG.info(
                "peer review complete",
                extra={"event": "peer_review.done", "verdict": verdict},
            )
            if verdict != "pass":
                updated = apply_peer_review_notice(result.assistant_text, verdict)
                result.assistant_text = updated
                session_log.set_assistant(updated)
        except Exception as ex:
            _LOG.exception("peer review failed", extra={"event": "peer_review.fail"})
            fallback = f"_(Peer review failed: {ex})_"
            result.peer_review_text = fallback
            result.peer_review_verdict = "unknown"
            session_log.set_peer_review(fallback)


def run_user_turn(
    client: OpenAI,
    messages: list,
    session_state: SessionState,
    session_log: SessionLog,
    db: DuckDBQuery | None,
    user_input: str,
    *,
    planner_disabled: bool,
    user_role: str | None = None,
    stream_delta: Callable[[str], None] | None = None,
    stream_reset: Callable[[], None] | None = None,
    planner_temperature: float = 0.2,
    planner_max_tokens: int = 600,
    executor_temperature: float = 0.2,
    executor_max_tokens: int | None = None,
) -> TurnResult:
    """
    Append user message, optional planner, executor loop with tools; update session_log.
    On success, runs peer review (second no-tools completion) when there is a final answer.
    On API failure, rolls back last user message and last turn in session_log.
    """
    result = TurnResult()
    flags = OrchestratorFlags.from_env(planner_disabled=planner_disabled)
    effective_role = normalize_role(user_role or os.getenv("APP_USER_ROLE"))
    session_state.set_last_user_request(user_input)
    entry_length = len(messages)
    messages.append({"role": "user", "content": user_input})
    session_log.start_turn(user_input)
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = build_system_content(session_state)

    analyst_intent: dict[str, Any] = {
        "plan_markdown": "",
        "prioritize_visualization": False,
        "visualization_hint": "",
    }
    if db and not flags.planner_disabled:
        try:
            analyst_intent = _run_data_analyst_agent(
                client,
                user_input,
                session_state,
                model=flags.chat_model,
                temperature=planner_temperature,
                max_tokens=planner_max_tokens,
            )
            plan_text = analyst_intent.get("plan_markdown", "").strip()
            if not plan_text:
                # Fallback to legacy planner if analyst JSON is empty/malformed.
                plan_messages = [{"role": "system", "content": PLANNER_SYSTEM}] + messages[1:]
                plan_resp = client.chat.completions.create(
                    model=flags.chat_model,
                    messages=plan_messages,
                    max_tokens=planner_max_tokens,
                    temperature=planner_temperature,
                )
                plan_text = (plan_resp.choices[0].message.content or "").strip()
            session_log.set_planner_phase(plan_text)
            result.planner_text = plan_text
            messages.append({"role": "assistant", "content": plan_text})
        except Exception as ex:
            session_log.set_planner_phase(f"Planner error: {ex}")
            result.planner_text = f"(Planner error: {ex})"
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = build_system_content(session_state)

    api_params: dict = {
        "model": flags.chat_model,
        "messages": messages,
        "temperature": executor_temperature,
    }
    if executor_max_tokens is not None:
        api_params["max_tokens"] = executor_max_tokens
    if db:
        api_params["tools"] = filter_openai_tools(OPENAI_TOOLS, effective_role)

    tool_round = 0
    sql_failures_this_turn = 0
    last_successful_chart_result: dict[str, Any] | None = None

    try:
        while True:
            if tool_round >= MAX_TOOL_ROUNDS_PER_TURN:
                session_log.set_assistant(None)
                result.error = f"Stopped: max tool rounds ({MAX_TOOL_ROUNDS_PER_TURN})"
                break
            _prune_messages_if_needed(messages, session_state)
            api_params["messages"] = messages
            if stream_reset:
                stream_reset()
            if stream_delta:
                response_message = _streaming_chat_completion(client, api_params, stream_delta)
            else:
                response = client.chat.completions.create(**api_params)
                response_message = response.choices[0].message
            messages.append(assistant_message_to_dict(response_message))

            if response_message.tool_calls:
                tool_round += 1
                result.tool_events.append(f"**Tool round {tool_round}** ({len(response_message.tool_calls)} call(s))")
                round_calls: list[tuple[str, dict, str]] = []

                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    preview = ", ".join(f"{k}={v}" for k, v in list(function_args.items())[:3])
                    if len(function_args) > 3:
                        preview += ", ..."
                    result.tool_events.append(f"- `{function_name}`({preview})")

                    if (
                        function_name == "query_database"
                        and sql_failures_this_turn >= MAX_SQL_FAILURES_PER_TURN
                    ):
                        function_response = json.dumps(
                            {
                                "error": "SQL retry limit reached for this user message.",
                                "retry_limit_reached": True,
                                "failures_recorded": sql_failures_this_turn,
                                "hint": "Explain the error without issuing more query_database calls for this question.",
                            }
                        )
                    elif function_name == "create_chart" and db is not None:
                        function_response = _dispatch_create_chart(
                            client,
                            function_name=function_name,
                            function_args=function_args,
                            db=db,
                            session_state=session_state,
                            effective_role=effective_role,
                            user_input=user_input,
                            analyst_intent=analyst_intent,
                            flags=flags,
                        )
                    else:
                        function_response = dispatch_tool(
                            function_name,
                            function_args,
                            db,
                            session_state,
                            user_role=effective_role,
                        )
                        if function_name == "query_database":
                            if query_tool_result_has_error(function_response):
                                sql_failures_this_turn += 1
                            else:
                                sql_failures_this_turn = 0
                        if function_name == "create_chart":
                            try:
                                d = json.loads(function_response)
                                p = d.get("chart_path")
                                if p and isinstance(p, str) and "error" not in d:
                                    result.chart_paths.append(p)
                                    last_successful_chart_result = d
                            except json.JSONDecodeError:
                                pass

                    round_calls.append((function_name, function_args, function_response))
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": function_response,
                        }
                    )

                session_log.add_tool_round(
                    tool_round,
                    round_calls,
                    assistant_reasoning=response_message.content,
                )
                api_params["messages"] = messages
            else:
                if response_message.content:
                    result.assistant_text = response_message.content
                    session_log.set_assistant(response_message.content)
                else:
                    session_log.set_assistant(None)
                break

        _finalize_turn_sync(
            client,
            result,
            session_state=session_state,
            session_log=session_log,
            user_input=user_input,
            flags=flags,
            last_successful_chart_result=last_successful_chart_result,
        )

    except Exception as e:
        result.error = _format_orchestrator_exception(e)
        del messages[entry_length:]
        if session_log.turns:
            session_log.turns.pop()
        return result

    return result


# ---------------------------------------------------------------------------
# Async sub-agents
# ---------------------------------------------------------------------------

async def _async_run_data_analyst_agent(
    client: AsyncOpenAI,
    user_input: str,
    state: SessionState,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    payload = (
        f"User request:\n{user_input}\n\n"
        f"Session context:\n{state.context_block() or '(none)'}\n"
    )
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ANALYST_SYSTEM},
            {"role": "user", "content": payload},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = (resp.choices[0].message.content or "").strip()
    d = _safe_json_dict(text)
    return {
        "plan_markdown": str(d.get("plan_markdown", "")).strip(),
        "prioritize_visualization": bool(d.get("prioritize_visualization", False)),
        "visualization_hint": str(d.get("visualization_hint", "")).strip(),
    }


async def _async_run_visualization_agent(
    client: AsyncOpenAI,
    *,
    user_input: str,
    function_args: dict[str, Any],
    analyst_intent: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    payload = (
        f"User request:\n{user_input}\n\n"
        f"Proposed create_chart args:\n{json.dumps(function_args, ensure_ascii=True)}\n\n"
        f"Analyst hint:\n{json.dumps(analyst_intent, ensure_ascii=True)}\n"
    )
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": VIZ_SYSTEM},
            {"role": "user", "content": payload},
        ],
        temperature=0.0,
        max_tokens=220,
    )
    d = _safe_json_dict((resp.choices[0].message.content or "").strip())
    return {
        "allow": bool(d.get("allow", True)),
        "reason": str(d.get("reason", "")).strip(),
        "title_override": str(d.get("title_override", "")).strip(),
    }


async def _async_run_report_writer_agent(
    client: AsyncOpenAI,
    *,
    user_input: str,
    draft_answer: str,
    planner_text: str | None,
    tool_events: list[str],
    model: str,
) -> str:
    evidence = (
        f"User request:\n{user_input}\n\n"
        f"Planner/Analyst plan:\n{planner_text or '(none)'}\n\n"
        f"Tool trace:\n{chr(10).join(tool_events) if tool_events else '(none)'}\n\n"
        f"Draft answer:\n{draft_answer}\n"
    )
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": REPORT_WRITER_SYSTEM},
            {"role": "user", "content": evidence},
        ],
        temperature=0.2,
        max_tokens=1200,
    )
    return (resp.choices[0].message.content or "").strip()


async def _finalize_turn_async(
    client: AsyncOpenAI,
    result: TurnResult,
    *,
    session_state: SessionState,
    session_log: SessionLog,
    user_input: str,
    flags: OrchestratorFlags,
    last_successful_chart_result: dict[str, Any] | None,
) -> None:
    with trace_span("finalize"):
        if (
            result.assistant_text
            and last_successful_chart_result is not None
            and not flags.chart_narrative_disabled
        ):
            try:
                with trace_span("finalize.chart_narrative"):
                    narrative = await _async_generate_chart_narrative(
                        client,
                        last_successful_chart_result,
                        model=flags.chat_model,
                    )
            except Exception as ex:
                _LOG.warning(
                    "chart narrative failed: %s",
                    ex,
                    extra={"event": "finalize.chart_narrative.fail"},
                )
                narrative = f"Chart generated; interpretation unavailable ({ex})."
            result.assistant_text = f"{result.assistant_text}\n\nChart interpretation: {narrative}"
            session_log.set_assistant(result.assistant_text)

        if result.assistant_text and not flags.report_writer_disabled:
            try:
                with trace_span("finalize.report_writer"):
                    rewritten = await _async_run_report_writer_agent(
                        client,
                        user_input=user_input,
                        draft_answer=result.assistant_text,
                        planner_text=result.planner_text,
                        tool_events=result.tool_events,
                        model=flags.chat_model,
                    )
                if rewritten:
                    result.assistant_text = rewritten
                    session_log.set_assistant(result.assistant_text)
            except Exception as ex:
                _LOG.warning(
                    "report writer failed: %s",
                    ex,
                    extra={"event": "finalize.report_writer.fail"},
                )

        if result.assistant_text and not flags.followup_ranking_disabled:
            with trace_span("finalize.followup_ranking"):
                ranked_text = rank_suggested_followups(
                    result.assistant_text,
                    session_state=session_state,
                    session_log=session_log,
                )
            if ranked_text != result.assistant_text:
                result.assistant_text = ranked_text
                session_log.set_assistant(result.assistant_text)

        if (
            not result.error
            and result.assistant_text
            and session_log.turns
            and not flags.peer_review_disabled
        ):
            review_model = flags.peer_review_model or flags.chat_model
            try:
                with trace_span("finalize.peer_review", model=review_model):
                    pr = await run_peer_review_async(
                        client,
                        session_log.turns[-1],
                        result.assistant_text,
                        model=review_model,
                    )
                result.peer_review_text = pr
                session_log.set_peer_review(pr)
                verdict = parse_peer_review_verdict(pr)
                result.peer_review_verdict = verdict
                _LOG.info(
                    "peer review complete",
                    extra={"event": "peer_review.done", "verdict": verdict},
                )
                if verdict != "pass":
                    updated = apply_peer_review_notice(result.assistant_text, verdict)
                    result.assistant_text = updated
                    session_log.set_assistant(updated)
            except Exception as ex:
                _LOG.exception("peer review failed", extra={"event": "peer_review.fail"})
                fallback = f"_(Peer review failed: {ex})_"
                result.peer_review_text = fallback
                result.peer_review_verdict = "unknown"
                session_log.set_peer_review(fallback)


async def _async_generate_chart_narrative(
    client: AsyncOpenAI,
    chart_result: dict,
    *,
    model: str | None = None,
) -> str:
    if not isinstance(chart_result, dict):
        return "Chart generated, but interpretation is unavailable due to invalid chart metadata."
    user_prompt = _chart_narrative_user_prompt(chart_result)
    if not user_prompt:
        return "Chart generated, but interpretation is unavailable because chart_path is missing."
    use_model = model or CHAT_MODEL
    resp = await client.chat.completions.create(
        model=use_model,
        messages=[
            {"role": "system", "content": _CHART_NARRATIVE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=120,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        return "Chart generated; interpretation could not be produced."
    first_sentence = text.split("\n")[0].strip()
    if "." in first_sentence:
        first_sentence = first_sentence.split(".", 1)[0].strip() + "."
    return first_sentence


async def _async_streaming_chat_completion(
    client: AsyncOpenAI,
    api_params: dict,
    stream_delta: Callable[[str], None] | None,
) -> Any:
    """Async streaming completion; returns an assistant message compatible with tool_calls iteration."""
    params = dict(api_params)
    params["stream"] = True
    stream = await client.chat.completions.create(**params)
    content_parts: list[str] = []
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta is None:
            continue
        if getattr(delta, "content", None):
            content_parts.append(delta.content)
            if stream_delta:
                stream_delta(delta.content)
        if getattr(delta, "tool_calls", None):
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls_by_index:
                    tool_calls_by_index[idx] = {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                if tc.id:
                    tool_calls_by_index[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        tool_calls_by_index[idx]["function"]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_calls_by_index[idx]["function"]["arguments"] += tc.function.arguments or ""
    content_str = "".join(content_parts)
    if tool_calls_by_index:
        tls = []
        for i in sorted(tool_calls_by_index.keys()):
            tc = tool_calls_by_index[i]
            tls.append({
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]},
            })
        return _assistant_dict_to_message(
            {"role": "assistant", "content": content_str or None, "tool_calls": tls}
        )
    return _assistant_dict_to_message({"role": "assistant", "content": content_str or None})


# ---------------------------------------------------------------------------
# Async orchestrator entry point
# ---------------------------------------------------------------------------

async def run_user_turn_async(
    client: AsyncOpenAI,
    messages: list,
    session_state: SessionState,
    session_log: SessionLog,
    db: DuckDBQuery | None,
    user_input: str,
    *,
    planner_disabled: bool,
    user_role: str | None = None,
    stream_delta: Callable[[str], None] | None = None,
    stream_reset: Callable[[], None] | None = None,
    planner_temperature: float = 0.2,
    planner_max_tokens: int = 600,
    executor_temperature: float = 0.2,
    executor_max_tokens: int | None = None,
    query_approval_enabled: bool | None = None,
) -> TurnResult:
    """
    Async version of run_user_turn.

    Key differences vs the sync version:
    - All OpenAI calls are awaited (AsyncOpenAI client).
    - Tool calls within a single round are dispatched concurrently via asyncio.gather
      (unless query approval is enabled — then sequential with optional pause).
    - When ENABLE_QUERY_APPROVAL is on and LIMIT >= threshold, returns early with
      approval_checkpoint; resume via resume_user_turn_after_approval_async.
    """
    hitl_enabled = (
        query_approval_enabled
        if query_approval_enabled is not None
        else approval_enabled_from_env()
    )
    result = TurnResult()
    flags = OrchestratorFlags.from_env(planner_disabled=planner_disabled)
    effective_role = normalize_role(user_role or os.getenv("APP_USER_ROLE"))
    session_state.set_last_user_request(user_input)
    entry_length = len(messages)
    messages.append({"role": "user", "content": user_input})
    session_log.start_turn(user_input)
    start_turn(user_input)
    _LOG.info(
        "orchestrator flags",
        extra={
            "event": "turn.config",
            "role": effective_role,
            "model": flags.chat_model,
            "planner_disabled": flags.planner_disabled,
            "hitl_enabled": hitl_enabled,
            "report_writer_disabled": flags.report_writer_disabled,
            "peer_review_disabled": flags.peer_review_disabled,
        },
    )
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = build_system_content(session_state)

    analyst_intent: dict[str, Any] = {
        "plan_markdown": "",
        "prioritize_visualization": False,
        "visualization_hint": "",
    }
    if db and not flags.planner_disabled:
        try:
            with trace_span("planner.analyst"):
                analyst_intent = await _async_run_data_analyst_agent(
                    client,
                    user_input,
                    session_state,
                    model=flags.chat_model,
                    temperature=planner_temperature,
                    max_tokens=planner_max_tokens,
                )
            plan_text = analyst_intent.get("plan_markdown", "").strip()
            if not plan_text:
                with trace_span("planner.legacy_fallback"):
                    plan_resp = await client.chat.completions.create(
                        model=flags.chat_model,
                        messages=[{"role": "system", "content": PLANNER_SYSTEM}] + messages[1:],
                        max_tokens=planner_max_tokens,
                        temperature=planner_temperature,
                    )
                    plan_text = (plan_resp.choices[0].message.content or "").strip()
            session_log.set_planner_phase(plan_text)
            result.planner_text = plan_text
            messages.append({"role": "assistant", "content": plan_text})
            _LOG.info(
                "planner ok",
                extra={
                    "event": "planner.done",
                    "prioritize_visualization": analyst_intent.get("prioritize_visualization"),
                    "plan_chars": len(plan_text),
                },
            )
        except Exception as ex:
            _LOG.exception("planner failed", extra={"event": "planner.fail"})
            session_log.set_planner_phase(f"Planner error: {ex}")
            result.planner_text = f"(Planner error: {ex})"
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = build_system_content(session_state)

    api_params: dict = {
        "model": flags.chat_model,
        "messages": messages,
        "temperature": executor_temperature,
    }
    if executor_max_tokens is not None:
        api_params["max_tokens"] = executor_max_tokens
    if db:
        api_params["tools"] = filter_openai_tools(OPENAI_TOOLS, effective_role)

    tool_round = 0
    sql_failures_this_turn = 0
    last_successful_chart_result: dict[str, Any] | None = None

    try:
        while True:
            if tool_round >= MAX_TOOL_ROUNDS_PER_TURN:
                session_log.set_assistant(None)
                result.error = f"Stopped: max tool rounds ({MAX_TOOL_ROUNDS_PER_TURN})"
                _LOG.error(
                    "max tool rounds exceeded",
                    extra={"event": "turn.max_tool_rounds", "max": MAX_TOOL_ROUNDS_PER_TURN},
                )
                break
            _prune_messages_if_needed(messages, session_state)
            api_params["messages"] = messages
            if stream_reset:
                stream_reset()
            with trace_span("executor.completion", round=tool_round + 1):
                if stream_delta:
                    response_message = await _async_streaming_chat_completion(
                        client, api_params, stream_delta
                    )
                else:
                    response = await client.chat.completions.create(**api_params)
                    response_message = response.choices[0].message
            messages.append(assistant_message_to_dict(response_message))

            if response_message.tool_calls:
                tool_round += 1
                _LOG.info(
                    "tool round",
                    extra={
                        "event": "executor.tool_round",
                        "round": tool_round,
                        "n_calls": len(response_message.tool_calls),
                    },
                )
                result.tool_events.append(
                    f"**Tool round {tool_round}** ({len(response_message.tool_calls)} call(s))"
                )

                if hitl_enabled and db is not None:
                    req, pending = first_call_needing_approval(
                        response_message.tool_calls,
                        enabled=True,
                        approved_keys=session_state.approved_sql_keys,
                    )
                    if req is not None:
                        for pc in pending:
                            fn = pc["name"]
                            args = pc["arguments"]
                            preview = ", ".join(
                                f"{k}={v}" for k, v in list(args.items())[:3]
                            )
                            result.tool_events.append(f"- `{fn}`({preview}) [awaiting approval]")
                        result.approval_request = req
                        result.approval_checkpoint = ApprovalCheckpoint(
                            user_input=user_input,
                            effective_role=effective_role,
                            tool_round=tool_round,
                            sql_failures_this_turn=sql_failures_this_turn,
                            analyst_intent=dict(analyst_intent),
                            planner_text=result.planner_text,
                            tool_events=list(result.tool_events),
                            chart_paths=list(result.chart_paths),
                            last_successful_chart_result=last_successful_chart_result,
                            pending_calls=pending,
                            assistant_reasoning=response_message.content,
                            planner_disabled=flags.planner_disabled,
                            planner_temperature=planner_temperature,
                            planner_max_tokens=planner_max_tokens,
                            executor_temperature=executor_temperature,
                            executor_max_tokens=executor_max_tokens,
                            query_approval_enabled=hitl_enabled,
                        )
                        _LOG.info(
                            "hitl approval required",
                            extra={
                                "event": "hitl.pause",
                                "tool": req.tool_name,
                                "limit": req.limit_value,
                            },
                        )
                        end_turn(
                            status="approval_pending",
                            tool_rounds=tool_round,
                            approval_pending=True,
                        )
                        return result

                failures_snapshot = sql_failures_this_turn

                async def _one_call(
                    tc: Any,
                    _failures: int = failures_snapshot,
                ) -> tuple[str, str, dict, str]:
                    fn = tc.function.name
                    args = json.loads(tc.function.arguments)
                    preview = ", ".join(f"{k}={v}" for k, v in list(args.items())[:3])
                    if len(args) > 3:
                        preview += ", ..."
                    result.tool_events.append(f"- `{fn}`({preview})")
                    resp = await _dispatch_one_tool_async(
                        client,
                        fn,
                        args,
                        db,
                        session_state,
                        effective_role,
                        user_input,
                        analyst_intent,
                        flags,
                        _failures,
                    )
                    return tc.id, fn, args, resp

                if hitl_enabled:
                    dispatched = []
                    fail_ct = sql_failures_this_turn
                    for tc in response_message.tool_calls:
                        tc_id, fn, args, resp = await _one_call(tc, fail_ct)
                        dispatched.append((tc_id, fn, args, resp))
                        if fn == "query_database":
                            if query_tool_result_has_error(resp):
                                fail_ct += 1
                            else:
                                fail_ct = 0
                    sql_failures_this_turn = fail_ct
                else:
                    dispatched = await asyncio.gather(
                        *[_one_call(tc) for tc in response_message.tool_calls]
                    )

                round_calls: list[tuple[str, dict, str]] = []
                for tc_id, fn, args, resp in dispatched:
                    round_calls.append((fn, args, resp))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": fn,
                        "content": resp,
                    })
                    if fn == "query_database":
                        if query_tool_result_has_error(resp):
                            sql_failures_this_turn += 1
                        else:
                            sql_failures_this_turn = 0
                    if fn == "create_chart":
                        chart_d = _apply_chart_result_to_turn(result, resp)
                        if chart_d is not None:
                            last_successful_chart_result = chart_d

                session_log.add_tool_round(
                    tool_round,
                    round_calls,
                    assistant_reasoning=response_message.content,
                )
                api_params["messages"] = messages
            else:
                if response_message.content:
                    result.assistant_text = response_message.content
                    session_log.set_assistant(response_message.content)
                else:
                    session_log.set_assistant(None)
                break

        await _finalize_turn_async(
            client,
            result,
            session_state=session_state,
            session_log=session_log,
            user_input=user_input,
            flags=flags,
            last_successful_chart_result=last_successful_chart_result,
        )

    except Exception as e:
        _LOG.exception("turn failed", extra={"event": "turn.exception"})
        result.error = _format_orchestrator_exception(e)
        del messages[entry_length:]
        if session_log.turns:
            session_log.turns.pop()
        end_turn(status="error", error=result.error, tool_rounds=tool_round)
        return result

    _log_turn_completion(result, tool_round)
    return result


async def resume_user_turn_after_approval_async(
    client: AsyncOpenAI,
    messages: list,
    session_state: SessionState,
    session_log: SessionLog,
    db: DuckDBQuery | None,
    checkpoint: ApprovalCheckpoint,
    *,
    approved: bool,
    stream_delta: Callable[[str], None] | None = None,
    stream_reset: Callable[[], None] | None = None,
) -> TurnResult:
    """Continue a turn paused for HITL SQL approval (or reject and let the model recover)."""
    _LOG.info(
        "hitl resume",
        extra={"event": "hitl.resume", "approved": approved, "n_calls": len(checkpoint.pending_calls)},
    )
    flags = OrchestratorFlags.from_env(planner_disabled=checkpoint.planner_disabled)
    result = TurnResult(
        planner_text=checkpoint.planner_text,
        tool_events=list(checkpoint.tool_events),
        chart_paths=list(checkpoint.chart_paths),
    )
    user_input = checkpoint.user_input
    effective_role = checkpoint.effective_role
    analyst_intent = checkpoint.analyst_intent
    tool_round = checkpoint.tool_round
    sql_failures_this_turn = checkpoint.sql_failures_this_turn
    last_successful_chart_result = checkpoint.last_successful_chart_result

    api_params: dict = {
        "model": flags.chat_model,
        "messages": messages,
        "temperature": checkpoint.executor_temperature,
    }
    if checkpoint.executor_max_tokens is not None:
        api_params["max_tokens"] = checkpoint.executor_max_tokens
    if db:
        api_params["tools"] = filter_openai_tools(OPENAI_TOOLS, effective_role)

    round_calls: list[tuple[str, dict, str]] = []
    for pc in checkpoint.pending_calls:
        fn = pc["name"]
        args = pc["arguments"]
        tc_id = pc["id"]
        sql = args.get("sql") if isinstance(args.get("sql"), str) else None
        if approved:
            if sql:
                session_state.mark_sql_approved(sql)
            resp = await _dispatch_one_tool_async(
                client,
                fn,
                args,
                db,
                session_state,
                effective_role,
                user_input,
                analyst_intent,
                flags,
                sql_failures_this_turn,
            )
        else:
            resp = _approval_rejection_payload(
                "User declined this query in the approval step."
            )
        round_calls.append((fn, args, resp))
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc_id,
                "name": fn,
                "content": resp,
            }
        )
        if fn == "query_database":
            if query_tool_result_has_error(resp):
                sql_failures_this_turn += 1
            else:
                sql_failures_this_turn = 0
        if fn == "create_chart":
            chart_d = _apply_chart_result_to_turn(result, resp)
            if chart_d is not None:
                last_successful_chart_result = chart_d

    session_log.add_tool_round(
        tool_round,
        round_calls,
        assistant_reasoning=checkpoint.assistant_reasoning,
    )
    api_params["messages"] = messages
    hitl_enabled = checkpoint.query_approval_enabled

    try:
        while True:
            if tool_round >= MAX_TOOL_ROUNDS_PER_TURN:
                session_log.set_assistant(None)
                result.error = f"Stopped: max tool rounds ({MAX_TOOL_ROUNDS_PER_TURN})"
                break
            _prune_messages_if_needed(messages, session_state)
            api_params["messages"] = messages
            if stream_reset:
                stream_reset()
            if stream_delta:
                response_message = await _async_streaming_chat_completion(
                    client, api_params, stream_delta
                )
            else:
                response = await client.chat.completions.create(**api_params)
                response_message = response.choices[0].message
            messages.append(assistant_message_to_dict(response_message))

            if response_message.tool_calls:
                tool_round += 1
                result.tool_events.append(
                    f"**Tool round {tool_round}** ({len(response_message.tool_calls)} call(s))"
                )

                if hitl_enabled and db is not None:
                    req, pending = first_call_needing_approval(
                        response_message.tool_calls,
                        enabled=True,
                        approved_keys=session_state.approved_sql_keys,
                    )
                    if req is not None:
                        for pc in pending:
                            fn = pc["name"]
                            args = pc["arguments"]
                            preview = ", ".join(
                                f"{k}={v}" for k, v in list(args.items())[:3]
                            )
                            result.tool_events.append(f"- `{fn}`({preview}) [awaiting approval]")
                        result.approval_request = req
                        result.approval_checkpoint = ApprovalCheckpoint(
                            user_input=user_input,
                            effective_role=effective_role,
                            tool_round=tool_round,
                            sql_failures_this_turn=sql_failures_this_turn,
                            analyst_intent=dict(analyst_intent),
                            planner_text=result.planner_text,
                            tool_events=list(result.tool_events),
                            chart_paths=list(result.chart_paths),
                            last_successful_chart_result=last_successful_chart_result,
                            pending_calls=pending,
                            assistant_reasoning=response_message.content,
                            planner_disabled=flags.planner_disabled,
                            planner_temperature=checkpoint.planner_temperature,
                            planner_max_tokens=checkpoint.planner_max_tokens,
                            executor_temperature=checkpoint.executor_temperature,
                            executor_max_tokens=checkpoint.executor_max_tokens,
                            query_approval_enabled=hitl_enabled,
                        )
                        return result

                fail_ct = sql_failures_this_turn
                round_calls = []
                for tc in response_message.tool_calls:
                    fn = tc.function.name
                    args = json.loads(tc.function.arguments)
                    preview = ", ".join(f"{k}={v}" for k, v in list(args.items())[:3])
                    result.tool_events.append(f"- `{fn}`({preview})")
                    resp = await _dispatch_one_tool_async(
                        client,
                        fn,
                        args,
                        db,
                        session_state,
                        effective_role,
                        user_input,
                        analyst_intent,
                        flags,
                        fail_ct,
                    )
                    round_calls.append((fn, args, resp))
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": fn,
                            "content": resp,
                        }
                    )
                    if fn == "query_database":
                        if query_tool_result_has_error(resp):
                            fail_ct += 1
                        else:
                            fail_ct = 0
                    if fn == "create_chart":
                        chart_d = _apply_chart_result_to_turn(result, resp)
                        if chart_d is not None:
                            last_successful_chart_result = chart_d
                sql_failures_this_turn = fail_ct
                session_log.add_tool_round(
                    tool_round,
                    round_calls,
                    assistant_reasoning=response_message.content,
                )
                api_params["messages"] = messages
            else:
                if response_message.content:
                    result.assistant_text = response_message.content
                    session_log.set_assistant(response_message.content)
                else:
                    session_log.set_assistant(None)
                break

        await _finalize_turn_async(
            client,
            result,
            session_state=session_state,
            session_log=session_log,
            user_input=user_input,
            flags=flags,
            last_successful_chart_result=last_successful_chart_result,
        )
    except Exception as e:
        _LOG.exception("turn failed after hitl resume", extra={"event": "turn.exception"})
        result.error = _format_orchestrator_exception(e)
        end_turn(status="error", error=result.error, tool_rounds=tool_round)
        return result

    _log_turn_completion(result, tool_round)
    return result
