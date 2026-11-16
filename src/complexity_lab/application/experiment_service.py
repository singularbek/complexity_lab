"""Application service for orchestrating deterministic experiment execution."""

import time
from pathlib import Path
from typing import Any

from complexity_lab.analysis.metrics import summarize_trajectory
from complexity_lab.domain.artifacts import Artifact
from complexity_lab.domain.exceptions import DomainValidationError, SimulationError
from complexity_lab.domain.experiments import Experiment, ExperimentParameters
from complexity_lab.domain.value_objects import ArtifactType, SimulationType
from complexity_lab.simulations.interfaces import SimulationRegistry
from complexity_lab.visualization.plots import plot_time_series
from complexity_lab.visualization.renderers import render_grid_image, render_trajectory_gif


class ExperimentService:
    """Orchestrates experiment lifecycles, simulation execution, and analysis."""

    def run_experiment(
        self,
        simulation_type: SimulationType = SimulationType.CELLULAR_AUTOMATON,
        parameters: dict[str, Any] | None = None,
        seed: int = 42,
        steps: int = 50,
        grid_size: tuple[int, int] = (32, 32),
        artifacts_dir: Path | str | None = None,
        render_gif: bool = False,
        render_plot: bool = False,
    ) -> Experiment:
        """Create, execute, analyze, and complete an experiment deterministically.

        Args:
            simulation_type: Enum classification of the simulation.
            parameters: Raw parameter overrides.
            seed: Deterministic random seed.
            steps: Number of simulation steps.
            grid_size: Dimensions of the grid (rows, cols).
            artifacts_dir: Optional directory to persist visual artifacts.
            render_gif: Whether to render and persist an animated GIF artifact.
            render_plot: Whether to render and persist a dynamics plot figure artifact.

        Returns:
            Completed domain Experiment entity with result observables, time series, and artifacts.
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

            artifacts: list[Artifact] = []
            if artifacts_dir is not None:
                out_dir = Path(artifacts_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                exp_prefix = experiment.experiment_id

                # Render final grid state
                if trajectory:
                    final_path = out_dir / f"{exp_prefix}_final.png"
                    render_grid_image(trajectory[-1], final_path)
                    artifacts.append(
                        Artifact(
                            name="final_state_image",
                            artifact_type=ArtifactType.IMAGE,
                            path=final_path,
                            mime_type="image/png",
                        )
                    )

                # Render dynamics time series plot
                if render_plot and result.time_series:
                    plot_path = out_dir / f"{exp_prefix}_dynamics.png"
                    plot_time_series(result.time_series, plot_path)
                    artifacts.append(
                        Artifact(
                            name="dynamics_plot",
                            artifact_type=ArtifactType.FIGURE,
                            path=plot_path,
                            mime_type="image/png",
                        )
                    )

                # Render trajectory animation GIF
                if render_gif and trajectory:
                    gif_path = out_dir / f"{exp_prefix}_animation.gif"
                    render_trajectory_gif(trajectory, gif_path)
                    artifacts.append(
                        Artifact(
                            name="trajectory_gif",
                            artifact_type=ArtifactType.ANIMATION,
                            path=gif_path,
                            mime_type="image/gif",
                        )
                    )

            experiment.complete(result=result, artifacts=artifacts)

        except (SimulationError, DomainValidationError, Exception) as exc:
            experiment.fail(str(exc))

        return experiment

