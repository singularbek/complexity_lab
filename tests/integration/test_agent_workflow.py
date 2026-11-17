"""Integration tests for the autonomous scientific agent end-to-end workflow."""

from pathlib import Path
from typing import Any

import pytest

from complexity_lab.application.agent import ScientificAgent, ScientificReport


@pytest.mark.integration
def test_scientific_agent_full_investigation(stub_provider: Any) -> None:
    events: list[str] = []

    def observer(event: str, _payload: dict[str, Any]) -> None:
        events.append(event)

    agent = ScientificAgent(provider=stub_provider, observer_callback=observer)  # type: ignore[arg-type]
    goal = "Investigate the impact of boundary conditions on density preservation"
    report = agent.investigate(research_goal=goal)

    assert isinstance(report, ScientificReport)
    assert report.research_goal == goal
    assert report.hypothesis is not None
    assert len(report.experiments_executed) >= 2

    for finding in report.experiments_executed:
        assert finding.status == "COMPLETED"
        assert "mean_density" in finding.key_metrics
        assert "final_entropy" in finding.key_metrics
        assert finding.experiment_id.startswith("exp_")

    assert len(report.synthesis) > 10
    assert "SUPPORTED" in report.conclusion or "consistent" in report.conclusion.lower()
    assert len(report.suggested_next_steps) >= 1

    assert "investigation_started" in events
    assert "plan_formulated" in events
    assert "executing_experiment" in events
    assert "experiment_completed" in events
    assert "investigation_completed" in events


@pytest.mark.integration
def test_scientific_agent_report_saving(
    stub_provider: Any, tmp_path: Path
) -> None:
    report_file = tmp_path / "research_report.md"
    agent = ScientificAgent(provider=stub_provider)  # type: ignore[arg-type]
    goal = "Investigate phase transition across initial density thresholds"
    report = agent.investigate(research_goal=goal, output_path=report_file)

    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")
    assert "# Scientific Investigation Report" in content
    assert report.hypothesis.statement in content
