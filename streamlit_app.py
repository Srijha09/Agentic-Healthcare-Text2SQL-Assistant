"""
Web UI (Streamlit): same agent as chat.py — planner, tools, charts, session audit export, downloads.

Run: uv run streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from agent_orchestrator import (
    CHAT_MODEL,
    PROJECT_ROOT,
    build_system_content,
    run_user_turn,
    session_repro_metadata,
)
from session_log import SessionLog
from tools.db_query import DuckDBQuery
from tools.session_state import SessionState

load_dotenv()

st.set_page_config(
    page_title="Healthcare analytics (synthetic)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_client() -> OpenAI | None:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key)


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


# --- Sidebar: Settings, then Bundle A → B → C (chart downloads live in chat under each plot) ---
with st.sidebar:
    st.header("Settings")
    planner_off = st.checkbox("Disable planner (executor only)", value=False)
    st.session_state.planner_disabled = planner_off
    st.session_state.session_log.set_repro_metadata(
        {"disable_planner": "1" if planner_off else "0"}
    )

    st.subheader("Bundle A — audit export")
    st.caption("Markdown includes repro metadata, planner, SQL log, tool I/O (redacted), final answer.")
    md_text = st.session_state.session_log.to_markdown()
    st.download_button(
        label="Download session audit (.md)",
        data=md_text,
        file_name="session_export.md",
        mime="text/markdown",
        disabled=len(st.session_state.session_log.turns) == 0,
    )

    st.subheader("Bundle B — session context")
    st.caption(
        "Rolling context (last question, SQL, tables, charts) is injected into the system prompt each turn."
    )

    st.subheader("Bundle C — visualization")
    st.caption(
        "Charts appear in the chat when the agent runs `create_chart`. "
        "Use **Download visualization** below each plot to save the PNG."
    )

    if st.button("New conversation", type="primary"):
        _reset_chat()
        st.rerun()

    st.divider()
    st.caption(f"Model: **{CHAT_MODEL}** · DB: **{'ok' if st.session_state.db else 'missing'}**")


# --- Main ---
st.title("Healthcare claims analytics (synthetic)")
st.markdown(
    "Ask questions in natural language. The app runs **planner → tools → answer** with **suggested follow-ups** "
    "(Bundle B). Export the **audit trail** from the sidebar (Bundle A). When a chart is generated, it appears here "
    "with a **Download visualization** button (Bundle C)."
)

if st.session_state.client is None:
    st.error("Set `OPENAI_API_KEY` in a `.env` file in the project root.")
    st.stop()

if st.session_state.db is None:
    st.warning(
        f"Database not loaded: {st.session_state.db_err or 'unknown'}. "
        "Run `make setup-db` or `uv run python scripts/load_archives_to_duckdb.py`. Tools will be unavailable."
    )

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
            st.markdown(turn["answer"])
        if turn.get("error"):
            st.error(turn["error"])

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

prompt = st.chat_input("Ask about the dataset (tables, cohorts, trends, charts)...")

from_followup = st.session_state.pop("followup_queue", None)
user_text = from_followup if from_followup else prompt

if user_text:
    if st.session_state.messages and st.session_state.messages[0].get("role") == "system":
        st.session_state.messages[0]["content"] = build_system_content(st.session_state.session_state)

    with st.spinner("Running agent..."):
        result = run_user_turn(
            st.session_state.client,
            st.session_state.messages,
            st.session_state.session_state,
            st.session_state.session_log,
            st.session_state.db,
            user_text,
            planner_disabled=st.session_state.planner_disabled,
        )

    st.session_state.ui_turns.append(
        {
            "user": user_text,
            "planner": result.planner_text,
            "tools": "\n".join(result.tool_events) if result.tool_events else None,
            "charts": list(result.chart_paths),
            "answer": result.assistant_text,
            "error": result.error,
        }
    )

    if st.session_state.messages and st.session_state.messages[0].get("role") == "system":
        st.session_state.messages[0]["content"] = build_system_content(st.session_state.session_state)

    st.rerun()
