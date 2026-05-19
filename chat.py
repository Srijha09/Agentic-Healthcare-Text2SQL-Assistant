#!/usr/bin/env python3
"""
Terminal-based chat using OpenAI GPT-4o + tools.

Flow matches `streamlit_app.py`: optional planner, tool rounds, final answer, peer review.
Commands: `export` writes session markdown; `exit` / `quit` / Ctrl+C exit (export if there
is conversation). Requires `OPENAI_API_KEY` and optional `healthcare.duckdb` for tools.
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from observability import configure_logging

from agent_orchestrator import (
    CHAT_MODEL,
    REPORTS_DIR,
    build_system_content,
    resume_user_turn_after_approval_async,
    run_user_turn_async,
    session_repro_metadata,
)
from tools.approval_policy import approval_enabled_from_env
from session_log import SessionLog
from tools.db_query import DuckDBQuery
from tools.session_state import SessionState


def main() -> None:
    load_dotenv()
    configure_logging()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env file")
        sys.exit(1)
    try:
        client = AsyncOpenAI(api_key=api_key)
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        sys.exit(1)

    try:
        db = DuckDBQuery()
        print("Connected to healthcare database")
        print(f"Available tables: {', '.join(db.list_tables())}")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Database tools will not be available. Run 'make setup-db' to create the database.")
        db = None
    except Exception as e:
        print(f"Warning: Could not connect to database: {e}")
        db = None

    session_state = SessionState()
    messages: list = [{"role": "system", "content": build_system_content(session_state)}]

    print(f"\nOpenAI Chat Terminal ({CHAT_MODEL})")
    if db:
        print("Database query tools are available!")
        print("Orchestration: planner -> executor (tools) -> final answer + suggested follow-ups")
        if approval_enabled_from_env():
            print("HITL: ENABLE_QUERY_APPROVAL is on — large LIMIT SQL prompts for y/N approval")
    print("Commands: 'export' - save session to outputs/reports/  |  'exit' / 'quit' - quit")
    print("-" * 60)

    session_log = SessionLog()
    session_log.set_repro_metadata(session_repro_metadata(db))

    def save_session_markdown() -> Path | None:
        if not session_log.turns:
            print("Nothing to export yet — have at least one conversation turn.")
            return None
        path = session_log.save(REPORTS_DIR)
        print(f"Session written to: {path}")
        return path

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ("exit", "quit"):
                if session_log.turns:
                    save_session_markdown()
                print("Goodbye!")
                break
            if user_input.lower() == "export":
                save_session_markdown()
                continue
            if not user_input:
                continue

            planner_disabled = os.getenv("DISABLE_PLANNER", "").lower() in ("1", "true", "yes")
            hitl = approval_enabled_from_env()
            result = asyncio.run(run_user_turn_async(
                client,
                messages,
                session_state,
                session_log,
                db,
                user_input,
                planner_disabled=planner_disabled,
                user_role=os.getenv("APP_USER_ROLE"),
                query_approval_enabled=hitl,
            ))

            while result.approval_checkpoint and result.approval_request:
                req = result.approval_request
                print(f"\n[Approval required] {req.reason}\n")
                print("--- SQL ---")
                print(req.sql_preview)
                print("-----------")
                ans = input("Approve and run this SQL? [y/N]: ").strip().lower()
                approved = ans in ("y", "yes")
                result = asyncio.run(
                    resume_user_turn_after_approval_async(
                        client,
                        messages,
                        session_state,
                        session_log,
                        db,
                        result.approval_checkpoint,
                        approved=approved,
                    )
                )

            if result.error:
                print(f"\nError: {result.error}")
                continue
            if result.planner_text and not planner_disabled:
                print(f"\n[Planner]\n{result.planner_text}\n")
            if result.assistant_text:
                print(f"\nAssistant: {result.assistant_text}")
            if result.peer_review_text:
                print(f"\n[Peer review]\n{result.peer_review_text}\n")
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = build_system_content(session_state)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            if session_log.turns:
                save_session_markdown()
            break
        except EOFError:
            print("\nGoodbye!")
            if session_log.turns:
                save_session_markdown()
            break

    if db:
        db.close()


if __name__ == "__main__":
    main()
