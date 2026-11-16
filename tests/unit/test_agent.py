"""Unit tests for the scientific agent, tools, and providers."""

import pytest

from complexity_lab.application.agent.models import ExperimentFinding, Hypothesis, ScientificReport
from complexity_lab.application.agent.providers import DeterministicScientistProvider
from complexity_lab.application.agent.tools import (
    DamageSpreadingTool,
    InspectSimulationTool,
    ListSimulationsTool,
    ParameterSweepTool,
    RunExperimentTool,
    ToolRegistry,
)
from complexity_lab.application.experiment_service import ExperimentService


@pytest.mark.unit
def test_tool_registry_and_schemas() -> None:
    registry = ToolRegistry()
    tools = registry.list_tools()
    assert len(tools) >= 4

    schemas = registry.get_schemas()
    names = [s["function"]["name"] for s in schemas]
    assert "run_experiment" in names
    assert "list_simulations" in names
    assert "inspect_simulation" in names
    assert "compare_experiments" in names


@pytest.mark.unit
def test_list_simulations_tool() -> None:
    tool = ListSimulationsTool()
    res = tool.execute()
    assert "cellular_automaton" in res["available_simulations"]
    assert res["count"] >= 1


@pytest.mark.unit
def test_inspect_simulation_tool() -> None:
    tool = InspectSimulationTool()
    res = tool.execute(simulation_name="cellular_automaton")
    assert res["name"] == "cellular_automaton"
    assert "supported_boundary_conditions" in res
    assert "conway_life" in res["canonical_rules"]
    assert "available_patterns" in res


@pytest.mark.unit
def test_run_experiment_tool() -> None:
    service = ExperimentService()
    tool = RunExperimentTool(service=service)
    res = tool.execute(
        simulation_type="cellular_automaton",
        parameters={"rule": "B3/S23", "init_mode": "random", "init_density": 0.25},
        seed=42,
        steps=20,
        grid_rows=16,
        grid_cols=16,
    )
    assert res["status"] == "COMPLETED"
    assert "observables" in res
    obs = res["observables"]
    assert "mean_density" in obs
    assert "final_entropy" in obs
    assert "time_series_summary" in res


@pytest.mark.unit
def test_deterministic_provider_planning_and_synthesis() -> None:
    provider = DeterministicScientistProvider()
    plan = provider.formulate_plan(
        goal="Compare HighLife B36/S23 vs Conway B3/S23",
        available_simulations=["cellular_automaton"],
        tool_schemas=[],
    )
    assert isinstance(plan.hypothesis, Hypothesis)
    assert len(plan.planned_experiments) >= 2

    # Verify synthesis
    synthesis, conclusion, next_steps = provider.synthesize(
        goal="Compare HighLife B36/S23 vs Conway B3/S23",
        hypothesis=plan.hypothesis,
        findings=[],
    )
    assert len(synthesis) > 0
    assert len(conclusion) > 0
    assert len(next_steps) > 0


@pytest.mark.unit
def test_parameter_sweep_tool() -> None:
    service = ExperimentService()
    tool = ParameterSweepTool(service=service)
    res = tool.execute(
        parameter_name="init_density",
        values=[0.1, 0.3],
        base_parameters={"rule": "B3/S23"},
        seed=42,
        steps=10,
        grid_rows=16,
        grid_cols=16,
    )
    assert res["parameter_name"] == "init_density"
    assert res["values_evaluated"] == 2
    assert len(res["sweep_results"]) == 2
    assert res["sweep_results"][0]["mean_density"] is not None


@pytest.mark.unit
def test_damage_spreading_tool() -> None:
    tool = DamageSpreadingTool()
    res = tool.execute(
        rule="B3/S23",
        density=0.25,
        seed=42,
        steps=15,
        grid_rows=16,
        grid_cols=16,
        perturbation_cells=1,
    )
    assert res["initial_damage"] == 1
    assert "final_damage" in res
    assert "damage_trajectory" in res
    assert len(res["damage_trajectory"]) == 16
    assert "dynamical_classification" in res


@pytest.mark.unit
def test_scientific_report_markdown_generation() -> None:
    hyp = Hypothesis(
        statement="HighLife sustains activity longer than Conway's Life.",
        reasoning="Replicator rule provides an additional birth transition.",
        expected_outcome="Activity > 0.",
    )
    finding = ExperimentFinding(
        experiment_name="Conway_Test",
        experiment_id="exp_001",
        status="COMPLETED",
        key_metrics={"mean_density": 0.22, "final_entropy": 0.75, "is_periodic": True, "period": 2},
        interpretation="Oscillator stabilized.",
    )
    report = ScientificReport(
        research_goal="Compare Life and HighLife",
        hypothesis=hyp,
        experiments_executed=[finding],
        synthesis="The empirical evidence shows stabilization.",
        conclusion="Hypothesis is SUPPORTED.",
        suggested_next_steps=["Test with higher resolution."],
    )
    md = report.to_markdown()
    assert "# Scientific Investigation Report" in md
    assert "HighLife sustains activity longer" in md
    assert "exp_001" in md
    assert "Hypothesis is SUPPORTED." in md
    assert "Test with higher resolution." in md

