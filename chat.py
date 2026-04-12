#!/usr/bin/env python3
"""
Terminal-based chat using OpenAI GPT-4o + tools. See also streamlit_app.py for a web UI.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from agent_orchestrator import (
    CHAT_MODEL,
    REPORTS_DIR,
    build_system_content,
    run_user_turn,
    session_repro_metadata,
)
from session_log import SessionLog
from tools.db_query import DuckDBQuery
from tools.session_state import SessionState


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env file")
        sys.exit(1)
    try:
        client = OpenAI(api_key=api_key)
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
            
            # Add user message to history
            messages.append({
                "role": "user",
                "content": user_input
            })
            session_log.start_turn(user_input)
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = build_system_content(session_state)

            planner_disabled = os.getenv("DISABLE_PLANNER", "").lower() in ("1", "true", "yes")
            result = run_user_turn(
                client,
                messages,
                session_state,
                session_log,
                db,
                user_input,
                planner_disabled=planner_disabled,
            )

            if result.error:
                print(f"\nError: {result.error}")
                continue
            if result.planner_text and not planner_disabled:
                print(f"\n[Planner]\n{result.planner_text}\n")
            if result.assistant_text:
                print(f"\nAssistant: {result.assistant_text}")
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
