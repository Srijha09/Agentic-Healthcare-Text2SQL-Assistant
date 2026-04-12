"""Markdown session export: repro metadata, planner text, SQL log, redacted tool I/O, final reply."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


def _truncate(s: str, max_len: int = 4000) -> str:
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 20] + "\n... [truncated for export]"


# Long numeric tokens in exported SQL (patient-like IDs, NPIs) — redact in markdown only.
_SQL_DIGIT_TOKEN = re.compile(r"\b\d{8,}\b")


def _redact_sql_for_export(sql: str) -> str:
    return _SQL_DIGIT_TOKEN.sub("[id]", sql)


def _redact_tool_result_for_export(result_json: str) -> str:
    """Redact PATIENT_NUMBER cells in JSON tool results for markdown export only."""
    try:
        d = json.loads(result_json)
    except json.JSONDecodeError:
        return result_json

    def redact_rows(cols: list, rows: list) -> None:
        if not isinstance(cols, list) or not isinstance(rows, list):
            return
        idx = [
            i
            for i, c in enumerate(cols)
            if c is not None and str(c).strip().upper() == "PATIENT_NUMBER"
        ]
        for row in rows:
            if not isinstance(row, list):
                continue
            for i in idx:
                if i < len(row):
                    row[i] = "[redacted]"

    if isinstance(d, dict):
        redact_rows(d.get("columns"), d.get("rows"))
        sd = d.get("sample_data")
        if isinstance(sd, dict):
            redact_rows(sd.get("columns"), sd.get("rows"))
        try:
            return json.dumps(d, indent=2, default=str)
        except TypeError:
            return str(d)
    return result_json


def _redact_arguments_for_export(args: dict) -> dict:
    if not isinstance(args, dict):
        return args
    out = dict(args)
    if isinstance(out.get("sql"), str):
        out["sql"] = _redact_sql_for_export(out["sql"])
    return out


def _tool_result_one_line(result_json: str) -> str:
    """Short summary for audit table (error vs ok)."""
    try:
        d = json.loads(result_json)
        if isinstance(d, dict):
            if d.get("retry_limit_reached"):
                return "blocked (SQL retry limit)"
            if "error" in d:
                return f"error: {_truncate(str(d['error']), 200)}"
            if "total_rows" in d:
                return f"ok, rows={d.get('total_rows')}"
            if "chart_path" in d:
                return f"ok, chart={d.get('chart_path')}"
            if "tables" in d:
                return f"ok, {len(d.get('tables', []))} tables"
            if "table" in d and "schema" in d:
                return f"ok, describe {d.get('table')}"
            if "row_count" in d:
                return f"ok, row_count={d.get('row_count')}"
        return "ok"
    except json.JSONDecodeError:
        return _truncate(result_json, 120)


@dataclass
class TurnLog:
    user: str
    planner_phase: str | None = None
    tool_rounds: list[dict] = field(default_factory=list)
    sql_audit_rows: list[dict] = field(default_factory=list)
    assistant: str | None = None


@dataclass
class SessionLog:
    """Accumulates one turn at a time; safe to call export anytime."""

    turns: list[TurnLog] = field(default_factory=list)
    repro: dict = field(default_factory=dict)

    def set_repro_metadata(self, meta: dict) -> None:
        """Merge reproducibility fields (model, git, database fingerprint)."""
        self.repro.update(meta)

    def start_turn(self, user_text: str) -> None:
        self.turns.append(TurnLog(user=user_text))

    def set_planner_phase(self, text: str | None) -> None:
        """Pre-tool plan text, if the planner step ran."""
        if not self.turns:
            return
        self.turns[-1].planner_phase = (text or "").strip() or None

    def add_tool_round(
        self,
        round_num: int,
        calls: list[tuple[str, dict, str]],
        assistant_reasoning: str | None = None,
    ) -> None:
        """calls: (function_name, arguments_dict, result_string)."""
        if not self.turns:
            return
        turn = self.turns[-1]
        entries = []
        for name, args, result in calls:
            entry = {
                "function": name,
                "arguments": args,
                "result": _truncate(result, 6000),
                "result_summary": _tool_result_one_line(result),
            }
            entries.append(entry)

            if name == "query_database" and isinstance(args, dict) and args.get("sql"):
                turn.sql_audit_rows.append(
                    {
                        "step": len(turn.sql_audit_rows) + 1,
                        "tool": name,
                        "sql": str(args["sql"]).strip(),
                        "round": round_num,
                    }
                )
            elif name == "create_chart" and isinstance(args, dict) and args.get("sql"):
                turn.sql_audit_rows.append(
                    {
                        "step": len(turn.sql_audit_rows) + 1,
                        "tool": name,
                        "sql": str(args["sql"]).strip(),
                        "round": round_num,
                        "chart_type": args.get("chart_type"),
                    }
                )

        tr = {
            "round": round_num,
            "calls": entries,
        }
        if assistant_reasoning and assistant_reasoning.strip():
            tr["assistant_reasoning_with_tools"] = assistant_reasoning.strip()
        turn.tool_rounds.append(tr)

    def set_assistant(self, text: str | None) -> None:
        if not self.turns:
            return
        self.turns[-1].assistant = text

    def _repro_markdown(self) -> list[str]:
        if not self.repro:
            return []
        lines = [
            "## Reproducibility",
            "",
            "_LLM outputs can vary; SQL and row counts below are from tools at export time._",
            "",
        ]
        for k in sorted(self.repro.keys()):
            v = self.repro[k]
            lines.append(f"- **{k}:** {v}")
        lines.append("")
        return lines

    def to_markdown(self) -> str:
        lines = [
            "# Session export",
            "",
            f"_Generated {datetime.now().isoformat(timespec='seconds')}. Synthetic data._",
            "",
        ]
        lines.extend(self._repro_markdown())
        for i, turn in enumerate(self.turns, start=1):
            lines.append(f"## Turn {i}")
            lines.append("")
            lines.append("### User")
            lines.append("")
            lines.append(turn.user)
            lines.append("")

            if turn.planner_phase:
                lines.append("### Planner")
                lines.append("")
                lines.append(turn.planner_phase)
                lines.append("")

            if turn.sql_audit_rows:
                lines.append("### SQL")
                lines.append("")
                lines.append("| Step | Tool | Round | SQL |")
                lines.append("|------|------|-------|-----|")
                for row in turn.sql_audit_rows:
                    sql_raw = str(row.get("sql", ""))
                    sql_cell = _redact_sql_for_export(sql_raw).replace("|", "\\|")
                    if len(sql_cell) > 500:
                        sql_cell = sql_cell[:480] + "…"
                    lines.append(
                        f"| {row.get('step')} | `{row.get('tool')}` | {row.get('round')} | `{sql_cell}` |"
                    )
                lines.append("")

            for tr in turn.tool_rounds:
                lines.append(f"### Tool round {tr['round']}")
                lines.append("")
                if tr.get("assistant_reasoning_with_tools"):
                    lines.append("**With tools:**")
                    lines.append("")
                    lines.append(tr["assistant_reasoning_with_tools"])
                    lines.append("")
                for j, call in enumerate(tr["calls"], start=1):
                    lines.append(f"#### {j}. `{call['function']}`")
                    lines.append("")
                    args = call["arguments"]
                    if args:
                        safe_args = _redact_arguments_for_export(args)
                        lines.append("```json")
                        try:
                            lines.append(json.dumps(safe_args, indent=2, default=str))
                        except TypeError:
                            lines.append(str(safe_args))
                        lines.append("```")
                        lines.append("")
                    lines.append(f"Result ({call.get('result_summary', '?')}):")
                    lines.append("")
                    lines.append("```")
                    lines.append(_redact_tool_result_for_export(call["result"]))
                    lines.append("```")
                    lines.append("")

            lines.append("### Answer")
            lines.append("")
            if turn.assistant:
                lines.append(turn.assistant)
            else:
                lines.append("_(none)_")
            lines.append("")
        return "\n".join(lines)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        safe_ts = re.sub(r"[^\dT]", "_", datetime.now().isoformat(timespec="seconds"))
        path = directory / f"session_{safe_ts}.md"
        path.write_text(self.to_markdown(), encoding="utf-8")
        return path
