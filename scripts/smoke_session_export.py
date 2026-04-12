"""One-off smoke test: generate a session markdown report under outputs/reports/. Run: uv run python scripts/smoke_session_export.py"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent_orchestrator import REPORTS_DIR, session_repro_metadata  # noqa: E402
from session_log import SessionLog  # noqa: E402
from tools.db_query import DuckDBQuery  # noqa: E402


def main() -> None:
    try:
        db = DuckDBQuery()
    except Exception:
        db = None

    log = SessionLog()
    log.set_repro_metadata(session_repro_metadata(db))
    if db:
        db.close()

    log.start_turn("How many distinct patients are in demographics?")
    log.set_planner_phase(
        "1. Confirm demographics table.\n2. Run COUNT DISTINCT PATIENT_NUMBER with LIMIT 1."
    )
    result_json = json.dumps(
        {
            "columns": ["n"],
            "rows": [[42]],
            "total_rows": 1,
            "truncated": False,
        }
    )
    log.add_tool_round(
        1,
        [
            (
                "query_database",
                {"sql": 'SELECT COUNT(DISTINCT "PATIENT_NUMBER") AS n FROM demographics LIMIT 1'},
                result_json,
            )
        ],
        assistant_reasoning=None,
    )
    log.set_assistant(
        "There are **42** distinct patients in `demographics`.\n\n"
        "### Suggested follow-ups\n"
        "- **By state:** Break down counts using `geography`.\n"
        "- **Age:** Join `PATIENT_YOB` from events.\n"
        "- **MX vs RX:** Compare patients with mx vs rx events.\n"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = log.save(REPORTS_DIR)
    text = path.read_text(encoding="utf-8")
    print("OK: report written to:\n ", path.resolve())
    print("\n--- First 1200 chars of report ---\n")
    print(text[:1200])
    if "## Reproducibility" not in text:
        sys.exit("FAIL: missing repro section")
    if "### SQL" not in text:
        sys.exit("FAIL: missing SQL section")
    print("\n--- Smoke test passed ---")


if __name__ == "__main__":
    main()
