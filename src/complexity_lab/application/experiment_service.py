"""Application service for orchestrating deterministic experiment execution."""

import time
from typing import Any

from complexity_lab.analysis.metrics import summarize_trajectory
from complexity_lab.domain.exceptions import DomainValidationError, SimulationError
from complexity_lab.domain.experiments import Experiment, ExperimentParameters
from complexity_lab.domain.value_objects import SimulationType
from complexity_lab.simulations.interfaces import SimulationRegistry


class ExperimentService:
    """Orchestrates experiment lifecycles, simulation execution, and analysis."""

    def run_experiment(
        self,
        simulation_type: SimulationType = SimulationType.CELLULAR_AUTOMATON,
        parameters: dict[str, Any] | None = None,
        seed: int = 42,
        steps: int = 50,
        grid_size: tuple[int, int] = (32, 32),
    ) -> Experiment:
        """Create, execute, analyze, and complete an experiment deterministically.

        Args:
            simulation_type: Enum classification of the simulation.
            parameters: Raw parameter overrides.
            seed: Deterministic random seed.
            steps: Number of simulation steps.
            grid_size: Dimensions of the grid (rows, cols).

        Returns:
            Completed domain Experiment entity with result observables and time series.
        """
        raw_params = dict(parameters or {})
        raw_params.setdefault("steps", steps)
        raw_params.setdefault("grid_size", grid_size)

        domain_params = ExperimentParameters(
            steps=steps,
            grid_size=grid_size,
            extra_params=raw_params,
        )

        experiment = Experiment(
            simulation_type=simulation_type,
            parameters=domain_params,
            seed=seed,
        )

        experiment.start()
        start_time = time.perf_counter()

        try:
            sim_cls = SimulationRegistry.get(simulation_type.value)
            sim = sim_cls()
            sim.initialize(seed=seed, parameters=raw_params)

            trajectory = sim.run(steps=steps)
            elapsed = time.perf_counter() - start_time

            activity = getattr(sim, "activity_history", None)
            result = summarize_trajectory(
                trajectory=trajectory,
                activity_history=activity,
                runtime_seconds=elapsed,
            )

            experiment.complete(result=result)

        except (SimulationError, DomainValidationError, Exception) as exc:
            experiment.fail(str(exc))

        return experiment

