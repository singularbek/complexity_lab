"""Autonomous Scientific Agent orchestrating planning, execution, and synthesis."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from complexity_lab.application.agent.models import (
    ExperimentFinding,
    ExperimentPlan,
    ScientificReport,
)
from complexity_lab.application.agent.providers import LLMProvider
from complexity_lab.application.agent.tools import ToolRegistry
from complexity_lab.simulations.interfaces import SimulationRegistry


class ScientificAgent:
    """Agent that formulates hypotheses, executes simulations, and synthesizes findings."""

    def __init__(
        self,
        provider: LLMProvider,
        tool_registry: ToolRegistry | None = None,
        observer_callback: Callable[[str, Any], None] | None = None,
    ) -> None:
        self.provider = provider
        self.tool_registry = tool_registry or ToolRegistry()
        self.observer = observer_callback or (lambda _event, _payload: None)

    def investigate(
        self,
        research_goal: str,
        output_path: Path | str | None = None,
    ) -> ScientificReport:
        """Conduct an autonomous scientific investigation for a given research objective.

        Workflow:
            1. Discover available tools and simulation capabilities.
            2. Formulate hypothesis and experiment plan via LLM provider.
            3. Execute deterministic simulations using tool registry.
            4. Synthesize real numerical metrics into a peer-review-style report.

        Args:
            research_goal: Natural language description of the research target.
            output_path: Optional file path to persist the synthesized Markdown report.

        Returns:
            ScientificReport with hypothesis, empirical data, synthesis, and conclusion.
        """
        self.observer("investigation_started", {"goal": research_goal})

        # Phase 1: Environment Discovery
        available_sims = SimulationRegistry.list_simulations()
        tool_schemas = self.tool_registry.get_schemas()
        self.observer("environment_discovered", {"simulations": available_sims})

        # Phase 2: Formulate Hypothesis & Design Controlled Experiments
        plan: ExperimentPlan = self.provider.formulate_plan(
            goal=research_goal,
            available_simulations=available_sims,
            tool_schemas=tool_schemas,
        )
        self.observer("plan_formulated", {"hypothesis": plan.hypothesis.statement, "plan": plan})

        # Phase 3: Execute Deterministic Simulations via Tools
        run_tool = self.tool_registry.get("run_experiment")
        findings: list[ExperimentFinding] = []

        for planned_exp in plan.planned_experiments:
            self.observer("executing_experiment", {"name": planned_exp.experiment_name})

            tool_result = run_tool.execute(
                simulation_type=planned_exp.simulation_type,
                parameters=planned_exp.parameters,
                seed=planned_exp.seed,
                steps=planned_exp.steps,
                grid_rows=planned_exp.grid_size[0],
                grid_cols=planned_exp.grid_size[1],
            )

            status = tool_result.get("status", "FAILED")
            observables = tool_result.get("observables", {})

            interpretation = (
                f"Completed in {tool_result.get('runtime_seconds', 0.0)}s. "
                f"Final alive cells: {observables.get('final_alive', 0)}, "
                f"Entropy: {observables.get('final_entropy', 0.0):.4f} bits, "
                f"Periodic: {observables.get('is_periodic', False)}."
            )

            finding = ExperimentFinding(
                experiment_name=planned_exp.experiment_name,
                experiment_id=tool_result.get("experiment_id", "unknown"),
                status=status,
                key_metrics=observables,
                interpretation=interpretation,
            )
            findings.append(finding)
            self.observer("experiment_completed", {"finding": finding})

        # Phase 4: Synthesize Numerical Evidence into Scientific Conclusion
        synthesis, conclusion, next_steps = self.provider.synthesize(
            goal=research_goal,
            hypothesis=plan.hypothesis,
            findings=findings,
        )

        report = ScientificReport(
            research_goal=research_goal,
            hypothesis=plan.hypothesis,
            experiments_executed=findings,
            synthesis=synthesis,
            conclusion=conclusion,
            suggested_next_steps=next_steps,
        )

        if output_path is not None:
            dest = Path(output_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(report.to_markdown(), encoding="utf-8")
            self.observer("report_saved", {"path": str(dest)})

        self.observer("investigation_completed", {"report": report})
        return report
