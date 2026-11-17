"""LLM Provider interfaces and external API engine for scientific planning."""

import json
import urllib.error
import urllib.request
from typing import Any, Protocol

from complexity_lab.application.agent.models import (
    ExperimentFinding,
    ExperimentPlan,
    Hypothesis,
)
from complexity_lab.application.agent.prompts import (
    HYPOTHESIS_PROMPT,
    PLANNING_PROMPT,
    SYNTHESIS_PROMPT,
    SYSTEM_PROMPT,
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


class ExternalAPIProvider:
    """Provider connecting to an external OpenAI-compatible LLM endpoint.

    Uses the structured prompts from prompts.py to guide the model through
    hypothesis formulation, experiment planning, and synthesis.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

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
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return str(result["choices"][0]["message"]["content"])

    def formulate_plan(
        self,
        goal: str,
        available_simulations: list[str],
        tool_schemas: list[dict[str, Any]],
    ) -> ExperimentPlan:
        """Build hypothesis + experiment plan using structured prompts."""
        _ = tool_schemas  # schemas available for future tool-call mode
        sim_list = (
            ", ".join(available_simulations)
            if available_simulations
            else "cellular_automaton"
        )

        # Step 1: formulate hypothesis
        hyp_user = HYPOTHESIS_PROMPT.format(goal=goal, simulations=sim_list)
        hyp_resp = self._call_api(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": hyp_user},
            ]
        )

        # Step 2: design experiments (JSON)
        plan_schema = ExperimentPlan.model_json_schema()
        plan_user = (
            PLANNING_PROMPT.format(statement=hyp_resp.strip())
            + f"\n\nRespond with valid JSON matching this schema:\n{plan_schema}"
            + f"\n\nUse goal: {goal!r}"
        )
        plan_resp = self._call_api(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": plan_user},
            ],
            json_mode=True,
        )
        return ExperimentPlan.model_validate_json(plan_resp)

    def synthesize(
        self,
        goal: str,
        hypothesis: Hypothesis,
        findings: list[ExperimentFinding],
    ) -> tuple[str, str, list[str]]:
        """Synthesize findings using the SYNTHESIS_PROMPT, returning structured JSON."""
        _ = goal
        results_summary = "\n".join(
            f"- {f.experiment_name}: {f.key_metrics}" for f in findings
        )
        synth_user = (
            SYNTHESIS_PROMPT.format(
                results_summary=results_summary,
                statement=hypothesis.statement,
            )
            + "\n\nRespond with JSON: {\"synthesis\": str, \"conclusion\": str, "
            "\"suggested_next_steps\": list[str]}"
        )
        resp = self._call_api(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": synth_user},
            ],
            json_mode=True,
        )
        data: dict[str, Any] = json.loads(resp)
        return (
            str(data.get("synthesis", "")),
            str(data.get("conclusion", "")),
            list(data.get("suggested_next_steps", [])),
        )
