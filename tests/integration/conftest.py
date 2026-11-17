"""Shared fixtures and stubs for integration tests."""

from typing import Any

import pytest

from complexity_lab.application.agent.models import (
    ExperimentFinding,
    ExperimentPlan,
    Hypothesis,
    PlannedExperiment,
)


class StubProvider:
    """Deterministic stub provider for integration tests — no API key required."""

    def formulate_plan(
        self,
        goal: str,
        available_simulations: list[str],
        tool_schemas: list[dict[str, Any]],
    ) -> ExperimentPlan:
        _ = available_simulations, tool_schemas
        hyp = Hypothesis(
            statement="Periodic boundaries preserve density better than fixed boundaries.",
            reasoning="Periodic boundaries prevent edge dissipation.",
            expected_outcome="Higher final density with periodic boundaries.",
        )
        return ExperimentPlan(
            goal=goal,
            hypothesis=hyp,
            planned_experiments=[
                PlannedExperiment(
                    experiment_name="Periodic_Test",
                    parameters={
                        "rule": "B3/S23",
                        "boundary": "periodic",
                        "init_mode": "random",
                        "init_density": 0.25,
                    },
                    seed=42,
                    steps=15,
                    grid_size=(16, 16),
                ),
                PlannedExperiment(
                    experiment_name="Fixed_Test",
                    parameters={
                        "rule": "B3/S23",
                        "boundary": "fixed",
                        "init_mode": "random",
                        "init_density": 0.25,
                    },
                    seed=42,
                    steps=15,
                    grid_size=(16, 16),
                ),
            ],
        )

    def synthesize(
        self,
        goal: str,
        hypothesis: Hypothesis,
        findings: list[ExperimentFinding],
    ) -> tuple[str, str, list[str]]:
        _ = goal, hypothesis
        density_summary = "; ".join(
            f"{f.experiment_name}: density={f.key_metrics.get('mean_density', 0):.4f}"
            for f in findings
        )
        return (
            f"Synthesis: {density_summary}",
            "Hypothesis is SUPPORTED by empirical simulation data.",
            ["Run more seeds.", "Test higher resolution grids."],
        )


@pytest.fixture
def stub_provider() -> StubProvider:
    return StubProvider()
