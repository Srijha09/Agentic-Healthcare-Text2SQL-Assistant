"""
Web UI (Streamlit): same agent as `chat.py` — planner, tools, streaming answer, peer review,
charts, session audit export, suggested follow-up buttons.

Run: ``uv run streamlit run streamlit_app.py``
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import AsyncOpenAI

from observability import configure_logging

from agent_orchestrator import (
    CHAT_MODEL,
    PROJECT_ROOT,
    TurnResult,
    build_system_content,
    resume_user_turn_after_approval_async,
    run_user_turn_async,
    session_repro_metadata,
)
from session_log import SessionLog
from tools.approval_policy import approval_enabled_from_env, approval_min_limit
from tools.db_query import DuckDBQuery
from tools.session_state import SessionState

load_dotenv()
configure_logging()

st.set_page_config(
    page_title="Intelligent Healthcare analysis platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _run_async(coro):
    """
    Run an async coroutine safely whether or not an event loop is already running.

    asyncio.run() raises RuntimeError when called from inside a running event loop
    (which some Streamlit versions do). In that case we submit to a fresh thread
    where no loop is running, so asyncio.run() works normally there.
    """
    try:
        asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)


def _init_client() -> AsyncOpenAI | None:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    return AsyncOpenAI(api_key=key)


def _init_db() -> tuple[DuckDBQuery | None, str | None]:
    try:
        return DuckDBQuery(), None
    except Exception as e:
        return None, str(e)


def _reset_chat() -> None:
    st.session_state.session_state = SessionState()
    st.session_state.messages = [
        {"role": "system", "content": build_system_content(st.session_state.session_state)}
    ]
    st.session_state.session_log = SessionLog()
    st.session_state.session_log.set_repro_metadata(session_repro_metadata(st.session_state.db))
    st.session_state.ui_turns = []
    st.session_state.chart_paths_all = []
    st.session_state.followup_queue = None


def parse_suggested_followups(answer_md: str | None) -> list[tuple[str, str]]:
    """
    Parse ### Suggested follow-ups into up to 3 (button_label, full_prompt) pairs.
    full_prompt is the bullet text sent as the next user message.
    """
    if not answer_md or "### Suggested follow-ups" not in answer_md:
        return []
    low = answer_md.lower()
    idx = low.find("### suggested follow-ups")
    if idx < 0:
        return []
    chunk = answer_md[idx:]
    m = re.search(r"###\s*Suggested follow-ups\s*\n(.*)", chunk, re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    rest = m.group(1)
    if "\n### " in rest:
        rest = rest.split("\n### ")[0]
    out: list[tuple[str, str]] = []
    for line in rest.split("\n"):
        line = line.strip()
        if not line.startswith("- ") and not line.startswith("* "):
            continue
        body = line[2:].strip()
        if not body:
            continue
        bold = re.match(r"^\*\*(.+?)\*\*\s*:\s*(.*)$", body)
        if bold:
            label = bold.group(1).strip()[:100]
        else:
            label = (body[:47] + "…") if len(body) > 48 else body
        out.append((label, body))
        if len(out) >= 3:
            break
    return out


if "client" not in st.session_state:
    st.session_state.client = _init_client()

if "db" not in st.session_state:
    db, err = _init_db()
    st.session_state.db = db
    st.session_state.db_err = err

if "session_state" not in st.session_state:
    st.session_state.session_state = SessionState()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": build_system_content(st.session_state.session_state)}
    ]

if "session_log" not in st.session_state:
    st.session_state.session_log = SessionLog()
    st.session_state.session_log.set_repro_metadata(session_repro_metadata(st.session_state.db))

if "ui_turns" not in st.session_state:
    st.session_state.ui_turns = []

if "chart_paths_all" not in st.session_state:
    st.session_state.chart_paths_all = []

if "followup_queue" not in st.session_state:
    st.session_state.followup_queue = None

if "stream_answer" not in st.session_state:
    st.session_state.stream_answer = True

if "planner_temperature" not in st.session_state:
    st.session_state.planner_temperature = 0.2
if "executor_temperature" not in st.session_state:
    st.session_state.executor_temperature = 0.2
if "planner_max_tokens" not in st.session_state:
    st.session_state.planner_max_tokens = 600
if "executor_max_tokens_cap" not in st.session_state:
    st.session_state.executor_max_tokens_cap = 0

if "app_user_role" not in st.session_state:
    _r0 = (os.getenv("APP_USER_ROLE") or "analyst").strip().lower()
    st.session_state.app_user_role = _r0 if _r0 in ("analyst", "viewer", "admin") else "analyst"

if "query_approval_enabled" not in st.session_state:
    st.session_state.query_approval_enabled = approval_enabled_from_env()

if "hitl_checkpoint" not in st.session_state:
    st.session_state.hitl_checkpoint = None
if "hitl_request" not in st.session_state:
    st.session_state.hitl_request = None
if "hitl_user_text" not in st.session_state:
    st.session_state.hitl_user_text = None
if "hitl_planner" not in st.session_state:
    st.session_state.hitl_planner = None
if "hitl_tool_events" not in st.session_state:
    st.session_state.hitl_tool_events = None


# --- Sidebar ---
with st.sidebar:
    st.markdown("### Features")
    st.markdown(
        """
- Natural-language Q&A on claims-style synthetic data  
- **Planner → tools → answer** workflow (optional planner-off mode)  
- **Session audit** export (markdown: SQL, tools, reproducibility)  
- **Rolling context** (recent question, SQL, tables, charts in prompts)  
- **Charts** (`create_chart`) with per-plot PNG download  
- **Numeric summaries** via `summarize_sql_stats`  
- **Streaming** final answers (toggle)  
- SQL **LIMIT** guardrails, timeouts, and retry limits  
- **Peer review** — second-pass consistency check on answers vs. tool evidence  
"""
    )

    with st.expander("Settings", expanded=False):
        _opts = ["analyst", "viewer", "admin"]
        _cur = st.session_state.app_user_role
        if _cur not in _opts:
            _cur = "analyst"
        _idx = _opts.index(_cur)
        st.session_state.app_user_role = st.selectbox(
            "User role (RBAC)",
            options=_opts,
            index=_idx,
            help=(
                "Analyst / Admin: all tools and tables. "
                "Viewer: no data_quality_check or analyze_care_gap; only `demographics` and `mx_events`."
            ),
        )
        planner_off = st.checkbox("Disable planner (executor only)", value=False)
        st.session_state.planner_disabled = planner_off
        st.session_state.query_approval_enabled = st.checkbox(
            "Require approval for large SQL (HITL)",
            value=bool(st.session_state.query_approval_enabled),
            help=(
                f"Pauses before running SQL when LIMIT >= {approval_min_limit()} "
                "(or ENABLE_QUERY_APPROVAL=1 in .env). You approve or reject in the UI."
            ),
        )
        st.caption("Optional: set env vars to skip report writer, viz agent, chart narrative, or peer review.")
        st.caption("Logs: stderr by default; set LOG_TO_FILE=1 for outputs/logs/agent.log (see env.template).")
        stream_on = st.checkbox("Stream final answer (token-by-token)", value=True)
        st.session_state.stream_answer = stream_on
        st.session_state.session_log.set_repro_metadata(
            {"disable_planner": "1" if planner_off else "0"}
        )

    with st.expander("Hyperparameters", expanded=False):
        st.caption("Tune OpenAI sampling for planner vs. assistant (tool-calling) steps.")
        st.session_state.planner_temperature = st.slider(
            "Planner temperature",
            min_value=0.0,
            max_value=1.5,
            value=float(st.session_state.planner_temperature),
            step=0.05,
            help="Lower = more deterministic plan text before tools run.",
        )
        st.session_state.planner_max_tokens = st.number_input(
            "Planner max tokens",
            min_value=200,
            max_value=2000,
            value=int(st.session_state.planner_max_tokens),
            step=50,
        )
        st.session_state.executor_temperature = st.slider(
            "Assistant temperature",
            min_value=0.0,
            max_value=1.5,
            value=float(st.session_state.executor_temperature),
            step=0.05,
            help="Applies to each executor completion (tools + final answer).",
        )
        cap = st.number_input(
            "Assistant max tokens (0 = model default)",
            min_value=0,
            max_value=32000,
            value=int(st.session_state.executor_max_tokens_cap),
            step=256,
            help="Caps the assistant reply length; 0 leaves the API default.",
        )
        st.session_state.executor_max_tokens_cap = int(cap)

    md_text = st.session_state.session_log.to_markdown()
    st.download_button(
        label="Download session audit (.md)",
        data=md_text,
        file_name="session_export.md",
        mime="text/markdown",
        disabled=len(st.session_state.session_log.turns) == 0,
    )

    if st.button("New conversation", type="primary"):
        _reset_chat()
        st.rerun()

    st.divider()
    st.caption(f"Model: **{CHAT_MODEL}** · DB: **{'ok' if st.session_state.db else 'missing'}**")


# --- Main ---
st.title("Intelligent Healthcare analysis platform")
st.markdown(
    """
This **agentic** analytics assistant uses an LLM with **tool calling** to plan analyses, query a
healthcare-style **DuckDB** dataset, summarize metrics, build charts, and answer in natural language—with optional
**streaming** responses and a **markdown audit trail** you can download for reproducibility. Session context keeps
recent questions, SQL, and visuals grounded across turns.

**Capabilities:** planner → multi-step tools (`query_database`, `summarize_sql_stats`, exploration helpers, `create_chart`),
suggested follow-ups, SQL guardrails (LIMIT, timeouts), and short-lived **query caching** for repeat lookups.
"""
)

if st.session_state.client is None:
    st.error("Set `OPENAI_API_KEY` in a `.env` file in the project root.")
    st.stop()

if st.session_state.db is None:
    st.warning(
        f"Database not loaded: {st.session_state.db_err or 'unknown'}. "
        "Run `make setup-db` or `uv run python scripts/load_archives_to_duckdb.py`. Tools will be unavailable."
    )

_n_turns = len(st.session_state.ui_turns)
for t_idx, turn in enumerate(st.session_state.ui_turns):
    with st.chat_message("user"):
        st.markdown(turn["user"])
    with st.chat_message("assistant"):
        if turn.get("planner"):
            with st.expander("Planner (reasoning before tools)", expanded=False):
                st.markdown(turn["planner"])
        if turn.get("tools"):
            with st.expander("Tool trace", expanded=False):
                st.markdown(turn["tools"])
        for c_idx, cp in enumerate(turn.get("charts") or []):
            img_path = PROJECT_ROOT / cp
            if img_path.is_file():
                st.image(str(img_path), caption=cp)
                with open(img_path, "rb") as f:
                    png_bytes = f.read()
                st.download_button(
                    label="Download visualization",
                    data=png_bytes,
                    file_name=Path(cp).name,
                    mime="image/png",
                    key=f"viz_dl_{t_idx}_{c_idx}_{Path(cp).name}",
                )
        if turn.get("answer"):
            verdict = turn.get("peer_review_verdict")
            if verdict and verdict not in ("pass", None):
                st.warning(
                    f"Peer review verdict: **{verdict}** — verify numbers against SQL in the audit export."
                )
            st.markdown(turn["answer"])
        if turn.get("error"):
            st.error(turn["error"])
        # Reviewer for past turns only (latest turn: after suggested follow-ups below)
        pr = turn.get("peer_review")
        if pr and t_idx < _n_turns - 1:
            with st.expander("Reviewer Evaluation", expanded=False):
                st.markdown(pr)

# Clickable follow-ups (last assistant turn only): three buttons, full bullet text as next question
last_turn = st.session_state.ui_turns[-1] if st.session_state.ui_turns else None
if last_turn and last_turn.get("answer") and not last_turn.get("error"):
    pairs = parse_suggested_followups(last_turn["answer"])
    if pairs:
        st.markdown("##### Suggested follow-ups")
        fu_cols = st.columns(len(pairs))
        last_idx = len(st.session_state.ui_turns) - 1
        for i, (btn_label, full_prompt) in enumerate(pairs):
            with fu_cols[i]:
                if st.button(
                    btn_label,
                    key=f"followup_{last_idx}_{i}",
                    use_container_width=True,
                    help=full_prompt[:500] + ("…" if len(full_prompt) > 500 else ""),
                ):
                    st.session_state.followup_queue = full_prompt
                    st.rerun()

if last_turn and last_turn.get("peer_review"):
    with st.expander("Reviewer Evaluation", expanded=False):
        st.markdown(last_turn["peer_review"])


def _render_assistant_turn(
    user_text: str,
    result: TurnResult,
    *,
    stream_slot: st.delta_generator.DeltaGenerator | None = None,
) -> None:
    """Show planner, tools, charts, answer for one completed (or resumed) turn."""
    if result.planner_text:
        with st.expander("Planner (reasoning before tools)", expanded=False):
            st.markdown(result.planner_text)
    if result.tool_events:
        with st.expander("Tool trace", expanded=False):
            st.markdown("\n".join(result.tool_events))
    for c_idx, cp in enumerate(result.chart_paths or []):
        img_path = PROJECT_ROOT / cp
        if img_path.is_file():
            st.image(str(img_path), caption=cp)
            with open(img_path, "rb") as f:
                png_bytes = f.read()
            st.download_button(
                label="Download visualization",
                data=png_bytes,
                file_name=Path(cp).name,
                mime="image/png",
                key=f"live_viz_{user_text[:20]}_{c_idx}_{Path(cp).name}",
            )
    if result.error:
        st.error(result.error)
    elif result.assistant_text:
        if stream_slot is not None:
            stream_slot.empty()
        st.markdown(result.assistant_text)


# --- HITL approval gate (paused turn) ---
if st.session_state.hitl_checkpoint and st.session_state.hitl_request:
    st.info("**Query approval required** before the agent can run this SQL against DuckDB.")
    if st.session_state.hitl_user_text:
        st.markdown(f"**Your question:** {st.session_state.hitl_user_text}")
    req = st.session_state.hitl_request
    st.markdown(req.reason)
    st.code(req.sql_preview, language="sql")
    if st.session_state.hitl_planner:
        with st.expander("Planner (before pause)", expanded=False):
            st.markdown(st.session_state.hitl_planner)
    if st.session_state.hitl_tool_events:
        with st.expander("Tool trace (before pause)", expanded=False):
            st.markdown("\n".join(st.session_state.hitl_tool_events))
    ac1, ac2, _ = st.columns([1, 1, 2])
    with ac1:
        if st.button("Approve and run query", type="primary", use_container_width=True):
            with st.spinner("Running approved query…"):
                result = _run_async(
                    resume_user_turn_after_approval_async(
                        st.session_state.client,
                        st.session_state.messages,
                        st.session_state.session_state,
                        st.session_state.session_log,
                        st.session_state.db,
                        st.session_state.hitl_checkpoint,
                        approved=True,
                    )
                )
            ut = st.session_state.hitl_user_text or ""
            with st.chat_message("assistant"):
                _render_assistant_turn(ut, result)
            st.session_state.ui_turns.append(
                {
                    "user": ut,
                    "planner": result.planner_text,
                    "tools": "\n".join(result.tool_events) if result.tool_events else None,
                    "charts": list(result.chart_paths),
                    "answer": result.assistant_text,
                    "error": result.error,
                    "peer_review": result.peer_review_text,
                    "peer_review_verdict": result.peer_review_verdict,
                }
            )
            st.session_state.hitl_checkpoint = None
            st.session_state.hitl_request = None
            st.session_state.hitl_user_text = None
            st.session_state.hitl_planner = None
            st.session_state.hitl_tool_events = None
            if st.session_state.messages and st.session_state.messages[0].get("role") == "system":
                st.session_state.messages[0]["content"] = build_system_content(
                    st.session_state.session_state
                )
            st.rerun()
    with ac2:
        if st.button("Reject query", use_container_width=True):
            with st.spinner("Returning rejection to agent…"):
                result = _run_async(
                    resume_user_turn_after_approval_async(
                        st.session_state.client,
                        st.session_state.messages,
                        st.session_state.session_state,
                        st.session_state.session_log,
                        st.session_state.db,
                        st.session_state.hitl_checkpoint,
                        approved=False,
                    )
                )
            ut = st.session_state.hitl_user_text or ""
            with st.chat_message("assistant"):
                _render_assistant_turn(ut, result)
            st.session_state.ui_turns.append(
                {
                    "user": ut,
                    "planner": result.planner_text,
                    "tools": "\n".join(result.tool_events) if result.tool_events else None,
                    "charts": list(result.chart_paths),
                    "answer": result.assistant_text,
                    "error": result.error,
                    "peer_review": result.peer_review_text,
                    "peer_review_verdict": result.peer_review_verdict,
                }
            )
            st.session_state.hitl_checkpoint = None
            st.session_state.hitl_request = None
            st.session_state.hitl_user_text = None
            st.session_state.hitl_planner = None
            st.session_state.hitl_tool_events = None
            if st.session_state.messages and st.session_state.messages[0].get("role") == "system":
                st.session_state.messages[0]["content"] = build_system_content(
                    st.session_state.session_state
                )
            st.rerun()
    st.stop()

prompt = st.chat_input("Ask about the dataset (tables, cohorts, trends, charts)...")

from_followup = st.session_state.pop("followup_queue", None)
user_text = from_followup if from_followup else prompt

if user_text:
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        stream_slot = st.empty()
        acc: list[str] = []

        def _on_reset() -> None:
            acc.clear()
            stream_slot.empty()

        def _on_delta(t: str) -> None:
            acc.append(t)
            stream_slot.markdown("".join(acc) + "▌")

        use_stream = st.session_state.get("stream_answer", True)
        _exec_max = st.session_state.get("executor_max_tokens_cap") or 0
        _exec_max_kw = None if _exec_max == 0 else int(_exec_max)
        _run_kw = dict(
            planner_disabled=st.session_state.planner_disabled,
            planner_temperature=float(st.session_state.get("planner_temperature", 0.2)),
            planner_max_tokens=int(st.session_state.get("planner_max_tokens", 600)),
            executor_temperature=float(st.session_state.get("executor_temperature", 0.2)),
            executor_max_tokens=_exec_max_kw,
        )
        _spin = "Running planner and tools…" if use_stream else "Running agent…"
        with st.spinner(_spin):
            result = _run_async(run_user_turn_async(
                st.session_state.client,
                st.session_state.messages,
                st.session_state.session_state,
                st.session_state.session_log,
                st.session_state.db,
                user_text,
                user_role=st.session_state.app_user_role,
                stream_delta=_on_delta if use_stream else None,
                stream_reset=_on_reset if use_stream else None,
                query_approval_enabled=st.session_state.query_approval_enabled,
                **_run_kw,
            ))

        if result.approval_checkpoint and result.approval_request:
            st.session_state.hitl_checkpoint = result.approval_checkpoint
            st.session_state.hitl_request = result.approval_request
            st.session_state.hitl_user_text = user_text
            st.session_state.hitl_planner = result.planner_text
            st.session_state.hitl_tool_events = list(result.tool_events)
            st.warning("Large SQL needs your approval — use the panel above.")
            st.rerun()

        _render_assistant_turn(
            user_text, result, stream_slot=stream_slot if use_stream else None
        )

    st.session_state.ui_turns.append(
        {
            "user": user_text,
            "planner": result.planner_text,
            "tools": "\n".join(result.tool_events) if result.tool_events else None,
            "charts": list(result.chart_paths),
            "answer": result.assistant_text,
            "error": result.error,
            "peer_review": result.peer_review_text,
            "peer_review_verdict": result.peer_review_verdict,
        }
    )

    if st.session_state.messages and st.session_state.messages[0].get("role") == "system":
        st.session_state.messages[0]["content"] = build_system_content(st.session_state.session_state)

    st.rerun()
