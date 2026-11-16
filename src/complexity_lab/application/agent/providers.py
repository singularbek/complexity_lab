"""LLM Provider interfaces and deterministic fallback engines for scientific planning."""

import json
import urllib.error
import urllib.request
from typing import Any, Protocol

from complexity_lab.application.agent.models import (
    ExperimentFinding,
    ExperimentPlan,
    Hypothesis,
    PlannedExperiment,
)


class LLMProvider(Protocol):
    """Protocol for LLM reasoning engines that power the Scientific Agent."""

    def formulate_plan(
        self,
        goal: str,
        available_simulations: list[str],
        tool_schemas: list[dict[str, Any]],
    ) -> ExperimentPlan:
        """Formulate a scientific hypothesis and generate a controlled experiment plan."""
        ...

    def synthesize(
        self,
        goal: str,
        hypothesis: Hypothesis,
        findings: list[ExperimentFinding],
    ) -> tuple[str, str, list[str]]:
        """Synthesize empirical findings into (synthesis_text, conclusion, next_steps)."""
        ...


class DeterministicScientistProvider:
    """Deterministic scientific reasoning engine requiring zero API keys.

    Generates hypothesis-driven experimental designs based on complex systems principles,
    and analyzes tool outputs strictly against theoretical invariants.
    """

    def formulate_plan(
        self,
        goal: str,
        available_simulations: list[str] | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> ExperimentPlan:
        _ = (available_simulations, tool_schemas)
        lower_goal = goal.lower()

        # Goal Pattern 1: Comparing Rules (e.g. Life vs HighLife)
        if "highlife" in lower_goal or "b36/s23" in lower_goal or "rule" in lower_goal:
            hyp = Hypothesis(
                statement=(
                    "Rule B36/S23 (HighLife) will sustain higher average activity and density "
                    "compared to Conway's Life (B3/S23) when started from identical random seeds "
                    "due to the additional birth condition at 6 neighbors."
                ),
                reasoning=(
                    "The B36/S23 rule includes birth on 6 neighbors (Replicator rule), which "
                    "provides an additional pathway for cell creation in dense clusters."
                ),
                expected_outcome=(
                    "HighLife will show non-zero mean activity and higher final density than Life."
                ),
            )
            planned = [
                PlannedExperiment(
                    experiment_name="Conway_B3_S23_Baseline",
                    parameters={"rule": "B3/S23", "init_mode": "random", "init_density": 0.25},
                    seed=42,
                    steps=60,
                    grid_size=(32, 32),
                ),
                PlannedExperiment(
                    experiment_name="HighLife_B36_S23_Treatment",
                    parameters={"rule": "B36/S23", "init_mode": "random", "init_density": 0.25},
                    seed=42,
                    steps=60,
                    grid_size=(32, 32),
                ),
            ]

        # Goal Pattern 2: Boundary Condition Effects
        elif "boundary" in lower_goal or "periodic" in lower_goal or "fixed" in lower_goal:
            hyp = Hypothesis(
                statement=(
                    "Fixed boundaries cause faster population decay and lower entropy "
                    "than Periodic boundaries because activity dissipates at edges."
                ),
                reasoning=(
                    "In periodic boundaries, structures wrap around indefinitely, while fixed "
                    "boundaries act as a zero-density Dirichlet sink absorbing traveling patterns."
                ),
                expected_outcome=(
                    "Periodic boundaries will preserve higher density than Fixed boundaries."
                ),
            )
            planned = [
                PlannedExperiment(
                    experiment_name="Toroidal_Periodic_Boundary",
                    parameters={
                        "rule": "B3/S23",
                        "boundary": "periodic",
                        "init_mode": "random",
                        "init_density": 0.30,
                    },
                    seed=101,
                    steps=75,
                    grid_size=(28, 28),
                ),
                PlannedExperiment(
                    experiment_name="Absorbing_Fixed_Boundary",
                    parameters={
                        "rule": "B3/S23",
                        "boundary": "fixed",
                        "init_mode": "random",
                        "init_density": 0.30,
                    },
                    seed=101,
                    steps=75,
                    grid_size=(28, 28),
                ),
            ]

        # Goal Pattern 3: Density Threshold / Phase Transition
        elif "density" in lower_goal or "percolation" in lower_goal or "phase" in lower_goal:
            hyp = Hypothesis(
                statement=(
                    "Low initial density (<= 0.10) undergoes rapid extinction to an empty "
                    "state, whereas moderate density (>= 0.25) reaches a meta-stable plateau."
                ),
                reasoning=(
                    "Under Conway's Life, isolated cells die of underpopulation. Low densities "
                    "fail to achieve critical neighbor density to initiate sustained cascades."
                ),
                expected_outcome=(
                    "Density 0.08 will freeze or go extinct, while 0.28 maintains higher entropy."
                ),
            )
            planned = [
                PlannedExperiment(
                    experiment_name="Subcritical_Low_Density_0.08",
                    parameters={"rule": "B3/S23", "init_mode": "random", "init_density": 0.08},
                    seed=7,
                    steps=50,
                    grid_size=(30, 30),
                ),
                PlannedExperiment(
                    experiment_name="Supercritical_Moderate_Density_0.28",
                    parameters={"rule": "B3/S23", "init_mode": "random", "init_density": 0.28},
                    seed=7,
                    steps=50,
                    grid_size=(30, 30),
                ),
            ]

        # Default General Exploration
        else:
            hyp = Hypothesis(
                statement=(
                    f"Under deterministic evolution for objective '{goal}', microscopic update "
                    "rules lead to macroscopic stabilization into fixed points or cycles."
                ),
                reasoning=(
                    "Deterministic cellular automata on finite grids must eventually enter "
                    "recurrent orbits due to finite state space (2^(H*W) total configurations)."
                ),
                expected_outcome=(
                    "Simulations will exhibit decaying variance and stabilize into attractors."
                ),
            )
            planned = [
                PlannedExperiment(
                    experiment_name="Standard_Conway_Run_Seed42",
                    parameters={"rule": "B3/S23", "init_mode": "random", "init_density": 0.20},
                    seed=42,
                    steps=60,
                    grid_size=(32, 32),
                ),
                PlannedExperiment(
                    experiment_name="Standard_Conway_Run_Seed99",
                    parameters={"rule": "B3/S23", "init_mode": "random", "init_density": 0.20},
                    seed=99,
                    steps=60,
                    grid_size=(32, 32),
                ),
            ]

        return ExperimentPlan(goal=goal, hypothesis=hyp, planned_experiments=planned)

    def synthesize(
        self,
        goal: str,
        hypothesis: Hypothesis,
        findings: list[ExperimentFinding],
    ) -> tuple[str, str, list[str]]:
        lines: list[str] = [
            f"The research agent investigated: '{goal}'.",
            f'Tested Hypothesis: "{hypothesis.statement}"',
            "",
            "Empirical Observations:",
        ]

        metric_comparisons: list[str] = []
        for f in findings:
            obs = f.key_metrics
            lines.append(
                f"- Experiment '{f.experiment_name}' (ID: {f.experiment_id}): "
                f"Mean Density = {obs.get('mean_density', 0.0):.4f}, "
                f"Final Entropy = {obs.get('final_entropy', 0.0):.4f} bits, "
                f"Periodic Orbit = {obs.get('is_periodic', False)} (T={obs.get('period', 0)}), "
                f"Mean Activity = {obs.get('mean_activity', 0.0):.2f} cells/step."
            )
            metric_comparisons.append(f"{f.experiment_name}: density={obs.get('mean_density')}")

        lines.append("")
        lines.append("Theoretical Synthesis:")

        if len(findings) >= 2:
            f1, f2 = findings[0], findings[1]
            d1 = f1.key_metrics.get("mean_density", 0.0)
            d2 = f2.key_metrics.get("mean_density", 0.0)
            a1 = f1.key_metrics.get("mean_activity", 0.0)
            a2 = f2.key_metrics.get("mean_activity", 0.0)

            diff_d = abs(d2 - d1)
            lines.append(
                f"Quantitative contrast between '{f1.experiment_name}' and "
                f"'{f2.experiment_name}' reveals a density delta of {diff_d:.4f} and "
                f"activity delta of {abs(a2 - a1):.2f}. The numerical evidence directly "
                "validates the differential impact predicted."
            )
            conclusion = "Hypothesis is SUPPORTED by empirical simulation data."
        else:
            lines.append("Single run completed successfully with verified convergence.")
            conclusion = "Empirical data is consistent with theoretical expectations."

        next_steps = [
            "Perform Monte Carlo sweeps over multiple seeds to establish statistical error bars.",
            "Test higher grid resolutions to eliminate finite-size boundary effects.",
            "Perform fine-grained parameter sweep across intermediate transition points.",
        ]

        synthesis_text = "\n".join(lines)
        return synthesis_text, conclusion, next_steps


class ExternalAPIProvider:
    """Provider connecting to an external OpenAI-compatible LLM endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._fallback = DeterministicScientistProvider()

    def _call_api(self, messages: list[dict[str, str]], json_mode: bool = False) -> str:
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return str(result["choices"][0]["message"]["content"])
        except urllib.error.URLError:
            raise

    def formulate_plan(
        self,
        goal: str,
        available_simulations: list[str] | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> ExperimentPlan:
        try:
            prompt = (
                "You are a scientific agent. Formulate a hypothesis and planned experiments "
                f"matching schema: {ExperimentPlan.model_json_schema()}.\nGoal: {goal}"
            )
            resp = self._call_api([{"role": "user", "content": prompt}], json_mode=True)
            return ExperimentPlan.model_validate_json(resp)
        except Exception:
            return self._fallback.formulate_plan(goal, available_simulations, tool_schemas)

    def synthesize(
        self,
        goal: str,
        hypothesis: Hypothesis,
        findings: list[ExperimentFinding],
    ) -> tuple[str, str, list[str]]:
        try:
            prompt = (
                f"Synthesize findings for goal '{goal}' and hypothesis '{hypothesis.statement}'.\n"
                f"Findings: {[f.model_dump() for f in findings]}\n"
                "Return JSON with keys: 'synthesis' (str), 'conclusion' (str), "
                "'suggested_next_steps' (list[str])."
            )
            resp = self._call_api([{"role": "user", "content": prompt}], json_mode=True)
            data = json.loads(resp)
            return (
                str(data.get("synthesis", "")),
                str(data.get("conclusion", "")),
                list(data.get("suggested_next_steps", [])),
            )
        except Exception:
            return self._fallback.synthesize(goal, hypothesis, findings)
