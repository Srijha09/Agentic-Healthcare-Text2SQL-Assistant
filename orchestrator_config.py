"""
Feature flags and model selection for the agent orchestrator (env-driven).

Set to 1 / true / yes to disable optional LLM phases and reduce cost per turn.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def _env_str(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


@dataclass
class OrchestratorFlags:
    """Resolved once per turn from environment (and optional overrides)."""

    planner_disabled: bool
    report_writer_disabled: bool
    viz_agent_disabled: bool
    chart_narrative_disabled: bool
    peer_review_disabled: bool
    followup_ranking_disabled: bool
    peer_review_model: str | None
    chat_model: str

    @classmethod
    def from_env(
        cls,
        *,
        planner_disabled: bool | None = None,
    ) -> OrchestratorFlags:
        return cls(
            planner_disabled=(
                planner_disabled
                if planner_disabled is not None
                else _env_truthy("DISABLE_PLANNER")
            ),
            report_writer_disabled=_env_truthy("DISABLE_REPORT_WRITER"),
            viz_agent_disabled=_env_truthy("DISABLE_VIZ_AGENT"),
            chart_narrative_disabled=_env_truthy("DISABLE_CHART_NARRATIVE"),
            peer_review_disabled=_env_truthy("DISABLE_PEER_REVIEW"),
            followup_ranking_disabled=_env_truthy("DISABLE_FOLLOWUP_RANKING"),
            peer_review_model=_env_str("PEER_REVIEW_MODEL") or None,
            chat_model=_env_str("CHAT_MODEL", "gpt-4o") or "gpt-4o",
        )

    def should_run_viz_gate(self, analyst_intent: dict) -> bool:
        """Skip viz LLM when disabled or analyst did not prioritize charts."""
        if self.viz_agent_disabled:
            return False
        return bool(analyst_intent.get("prioritize_visualization"))
