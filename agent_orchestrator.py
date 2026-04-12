"""
Shared agent loop: planner, tool rounds, final answer. Used by chat.py and streamlit_app.py.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from session_log import SessionLog
from tools.db_query import DuckDBQuery
from tools.session_state import SessionState
from tools.tool_registry import OPENAI_TOOLS, dispatch_tool

CHAT_MODEL = "gpt-4o"
MAX_SQL_FAILURES_PER_TURN = 2
MAX_TOOL_ROUNDS_PER_TURN = 28

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
# Session markdown audit exports (terminal `export` / exit, and parity with Streamlit downloads)
REPORTS_DIR = OUTPUTS_DIR / "reports"

SYSTEM_PROMPT_BASE = """You are an analyst assistant for a synthetic healthcare claims dataset in DuckDB.
If the user thread already has a short numbered plan from before tools, use it as a guide; ground answers in tool output.

Workflow:
1. Clarify what the user needs. If the schema is unclear, use list_tables, describe_table, table_info, or profile_table before writing SQL.
2. Use query_database to retrieve facts. Write correct, efficient SQL (JOINs on PATIENT_NUMBER). Every SELECT that uses FROM must include LIMIT n (enforced by the tool).
3. For charts, use create_chart with read-only SQL (two columns, LIMIT on SELECT...FROM), a descriptive title (cohort, metric, time or breakdown), and chart_type. Same SQL guard as query_database. Saves a PNG under outputs/visualization/; paste markdown_embed in your answer.
4. After tool results return, answer in plain language. Cite what you computed. If a tool returned an error or truncated rows, say so and suggest a follow-up query.

SQL errors (self-correction):
- If query_database returns an error, read the message, fix the SQL, and try again.
- If you see retry_limit_reached in a tool result, stop issuing query_database for this user question: explain the issue and suggest schema checks or a simpler query. The user can ask a new question to continue.

Follow-up UX:
- When you give a final natural-language answer (no more tool calls needed), end with a section titled exactly "### Suggested follow-ups" with exactly **three** bullet lines.
- Each bullet must (a) reference the **same cohort, metric, or question** you just answered or the **Latest user request** in session context, and (b) propose a **specific next step** — at least one bullet should suggest **stratification** (e.g. by `PATIENT_STATE`, year/month, or sex from demographics) or **comparison** (e.g. two time windows) using tables you already named.
- Do not use generic bullets ("dig deeper", "explore more") without naming tables, columns, or filters.

Rules:
- Do not invent patient counts, rates, or IDs; only state what tools returned.
- The database is read-only; you cannot modify data.
- Protect privacy: avoid unnecessary row-level exports; prefer aggregates for summaries."""

PLANNER_SYSTEM = """You outline analysis steps only (synthetic data; not clinical advice).
Give a numbered plan (4-7 steps): tables, joins on PATIENT_NUMBER, filters. No SQL; no made-up counts.
The next step in the app will run database tools."""


def session_repro_metadata(db: DuckDBQuery | None) -> dict:
    meta: dict = {
        "model": CHAT_MODEL,
        "disable_planner": os.getenv("DISABLE_PLANNER", "") or "0",
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


def assistant_message_to_dict(response_message: Any) -> dict:
    if isinstance(response_message, dict):
        return response_message
    if hasattr(response_message, "model_dump"):
        return response_message.model_dump(exclude_none=True)
    raise TypeError(f"Unexpected assistant message type: {type(response_message)!r}")


@dataclass
class TurnResult:
    assistant_text: str | None = None
    planner_text: str | None = None
    tool_events: list[str] = field(default_factory=list)
    chart_paths: list[str] = field(default_factory=list)
    error: str | None = None


def run_user_turn(
    client: OpenAI,
    messages: list,
    session_state: SessionState,
    session_log: SessionLog,
    db: DuckDBQuery | None,
    user_input: str,
    *,
    planner_disabled: bool,
) -> TurnResult:
    """
    Append user message, optional planner, executor loop with tools; update session_log.
    On API failure, rolls back last user message and last turn in session_log.
    """
    result = TurnResult()
    session_state.set_last_user_request(user_input)
    messages.append({"role": "user", "content": user_input})
    session_log.start_turn(user_input)
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = build_system_content(session_state)

    if db and not planner_disabled:
        try:
            plan_messages = [{"role": "system", "content": PLANNER_SYSTEM}] + messages[1:]
            plan_resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=plan_messages,
                max_tokens=600,
                temperature=0.2,
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

    api_params: dict = {"model": CHAT_MODEL, "messages": messages}
    if db:
        api_params["tools"] = OPENAI_TOOLS

    tool_round = 0
    sql_failures_this_turn = 0

    try:
        while True:
            if tool_round >= MAX_TOOL_ROUNDS_PER_TURN:
                session_log.set_assistant(None)
                result.error = f"Stopped: max tool rounds ({MAX_TOOL_ROUNDS_PER_TURN})"
                break
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
                    else:
                        function_response = dispatch_tool(
                            function_name, function_args, db, session_state
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
                                if p and isinstance(p, str):
                                    result.chart_paths.append(p)
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

    except Exception as e:
        result.error = str(e)
        messages.pop()
        if session_log.turns:
            session_log.turns.pop()
        return result

    return result
