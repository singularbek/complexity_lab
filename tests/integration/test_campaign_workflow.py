"""Integration tests for the Autonomous Research Campaign and artifact generation."""

from pathlib import Path
from typing import Any

import pytest

from complexity_lab.application.agent import AutonomousResearchCampaign, ResearchCampaignReport
from complexity_lab.application.experiment_service import ExperimentService
from complexity_lab.domain.value_objects import ArtifactType, SimulationType


@pytest.mark.integration
def test_autonomous_research_campaign_full_workflow(tmp_path: Path) -> None:
    events: list[str] = []

    def observer(event: str, _payload: dict[str, Any]) -> None:
        events.append(event)

    campaign_runner = AutonomousResearchCampaign(observer_callback=observer)
    goal = "Investigate boundary effects and perturbation stability"
    journal_path = tmp_path / "campaign_journal.md"
    art_dir = tmp_path / "artifacts"

    campaign_report = campaign_runner.execute_campaign(
        campaign_goal=goal,
        max_rounds=2,
        output_path=journal_path,
        artifacts_dir=art_dir,
    )

    assert isinstance(campaign_report, ResearchCampaignReport)
    assert campaign_report.total_rounds == 2
    assert len(campaign_report.rounds) == 2
    assert journal_path.exists()

    journal_text = journal_path.read_text(encoding="utf-8")
    assert "# Autonomous Scientific Campaign" in journal_text
    assert "Round 1" in journal_text
    assert "Round 2" in journal_text
    assert "Meta-Synthesis" in journal_text

    # Verify event cycle
    assert "campaign_started" in events
    assert "round_started" in events
    assert "round_completed" in events
    assert "campaign_completed" in events


@pytest.mark.integration
def test_experiment_service_visual_artifact_generation(tmp_path: Path) -> None:
    service = ExperimentService()
    art_dir = tmp_path / "exp_artifacts"

    exp = service.run_experiment(
        simulation_type=SimulationType.CELLULAR_AUTOMATON,
        parameters={"rule": "B3/S23", "init_mode": "random", "init_density": 0.25},
        seed=42,
        steps=10,
        grid_size=(16, 16),
        artifacts_dir=art_dir,
        render_gif=True,
        render_plot=True,
    )

    assert exp.status.value == "COMPLETED"
    assert len(exp.artifacts) >= 3

    artifact_types = [a.artifact_type for a in exp.artifacts]
    assert ArtifactType.IMAGE in artifact_types
    assert ArtifactType.FIGURE in artifact_types
    assert ArtifactType.ANIMATION in artifact_types

    for a in exp.artifacts:
        assert a.path.exists()
        assert a.path.stat().st_size > 50

