"""Rolling context for one chat session (last SQL, tables, charts), merged into the system prompt."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionState:
    """Tracks recent analyst actions for context (not persisted across process restarts)."""

    last_user_request: str | None = None
    last_sql: str | None = None
    last_profiled_table: str | None = None
    chart_paths: list[str] = field(default_factory=list)
    recent_tables: list[str] = field(default_factory=list)
    _max_recent: int = 6
    _max_charts: int = 8
    _max_user_snippet: int = 400

    def set_last_user_request(self, text: str | None) -> None:
        """Latest user message (truncated), for grounding suggested follow-ups in the system prompt."""
        if not text or not str(text).strip():
            return
        t = str(text).strip()
        self.last_user_request = t[: self._max_user_snippet]

    def note_sql(self, sql: str | None) -> None:
        if not sql:
            return
        self.last_sql = sql.strip()[:4000]

    def note_table(self, table: str | None) -> None:
        if not table:
            return
        t = table.strip()
        if t and t not in self.recent_tables:
            self.recent_tables.append(t)
        if len(self.recent_tables) > self._max_recent:
            self.recent_tables = self.recent_tables[-self._max_recent :]

    def note_profile(self, table: str | None) -> None:
        if table:
            self.last_profiled_table = table.strip()
            self.note_table(table)

    def note_chart(self, relative_path: str | None) -> None:
        if not relative_path:
            return
        p = relative_path.strip().replace("\\", "/")
        if p and p not in self.chart_paths:
            self.chart_paths.append(p)
        if len(self.chart_paths) > self._max_charts:
            self.chart_paths = self.chart_paths[-self._max_charts :]

    def context_block(self) -> str:
        """Short markdown for appending to the system prompt."""
        lines: list[str] = []
        if self.last_user_request:
            u = self.last_user_request
            tail = "…" if len(self.last_user_request) >= self._max_user_snippet else ""
            lines.append(
                f"- Latest user request (tie follow-ups to this): {u}{tail}"
            )
        if self.last_sql:
            lines.append(f"- Last SQL executed (truncated): `{self.last_sql[:500]}{'…' if len(self.last_sql) > 500 else ''}`")
        if self.recent_tables:
            lines.append(f"- Recently used tables: {', '.join(self.recent_tables)}")
        if self.last_profiled_table:
            lines.append(f"- Last profiled table: `{self.last_profiled_table}`")
        if self.chart_paths:
            lines.append(
                "- Charts saved this session: "
                + ", ".join(f"`{p}`" for p in self.chart_paths[-4:])
            )
        if not lines:
            return ""
        return "\n".join(lines)
