"""Orchestrator feature flags."""

import os

from orchestrator_config import OrchestratorFlags


def test_viz_gate_skipped_when_not_prioritized(monkeypatch) -> None:
    monkeypatch.delenv("DISABLE_VIZ_AGENT", raising=False)
    flags = OrchestratorFlags.from_env()
    assert flags.should_run_viz_gate({"prioritize_visualization": False}) is False


def test_viz_gate_runs_when_prioritized(monkeypatch) -> None:
    monkeypatch.delenv("DISABLE_VIZ_AGENT", raising=False)
    flags = OrchestratorFlags.from_env()
    assert flags.should_run_viz_gate({"prioritize_visualization": True}) is True


def test_disable_viz_agent_env(monkeypatch) -> None:
    monkeypatch.setenv("DISABLE_VIZ_AGENT", "1")
    flags = OrchestratorFlags.from_env()
    assert flags.should_run_viz_gate({"prioritize_visualization": True}) is False
