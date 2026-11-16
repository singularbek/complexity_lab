"""Scientific tools interface providing explicit, schema-validated execution for LLMs."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from complexity_lab.analysis.metrics import compute_damage_spreading
from complexity_lab.application.experiment_service import ExperimentService
from complexity_lab.domain.value_objects import SimulationType
from complexity_lab.simulations.cellular_automaton import (
    PATTERNS,
    BoundaryCondition,
    CellularAutomatonSimulation,
    InitMode,
    NeighborhoodType,
)
from complexity_lab.simulations.interfaces import SimulationRegistry
from complexity_lab.visualization.plots import (
    plot_parameter_sweep,
    plot_time_series,
)
from complexity_lab.visualization.renderers import render_trajectory_gif


class BaseTool(ABC):
    """Abstract base class for all scientific tools invocable by an agent."""

    name: str
    description: str
    args_schema: type[BaseModel]

    @abstractmethod
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool deterministically and return structured results."""
        ...

    def to_json_schema(self) -> dict[str, Any]:
        """Return JSON Schema compatible with OpenAI/Gemini/Anthropic function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema.model_json_schema(),
            },
        }


# ---------------------------------------------------------------------------
# Tool 1: List Available Simulations
# ---------------------------------------------------------------------------


class ListSimulationsInput(BaseModel):
    """Empty schema for listing simulations."""


class ListSimulationsTool(BaseTool):
    """Discover registered scientific simulation engines."""

    name = "list_simulations"
    description = "List all simulation models currently available in the scientific laboratory."
    args_schema = ListSimulationsInput

    def execute(self, **_kwargs: Any) -> dict[str, Any]:
        simulations = SimulationRegistry.list_simulations()
        return {
            "available_simulations": simulations,
            "count": len(simulations),
        }


# ---------------------------------------------------------------------------
# Tool 2: Inspect Simulation Capabilities & Schema
# ---------------------------------------------------------------------------


class InspectSimulationInput(BaseModel):
    simulation_name: str = Field(
        ...,
        description="Name of the simulation to inspect (e.g. 'cellular_automaton')",
    )


class InspectSimulationTool(BaseTool):
    """Inspect parameter schema, boundaries, and patterns supported by a simulation."""

    name = "inspect_simulation"
    description = (
        "Inspect the parameter requirements, supported boundary conditions, and "
        "available pattern templates for a specific simulation engine."
    )
    args_schema = InspectSimulationInput

    def execute(
        self,
        simulation_name: str = "cellular_automaton",
        **_kwargs: Any,
    ) -> dict[str, Any]:
        try:
            sim_cls = SimulationRegistry.get(simulation_name)
            sim_instance = sim_cls()
            param_schema = getattr(sim_instance, "_param_schema", None)
            schema_json = param_schema.model_json_schema() if param_schema else {}

            details: dict[str, Any] = {
                "name": simulation_name,
                "parameter_schema": schema_json,
            }

            if simulation_name == "cellular_automaton":
                details["supported_boundary_conditions"] = [b.value for b in BoundaryCondition]
                details["supported_neighborhoods"] = [n.value for n in NeighborhoodType]
                details["supported_init_modes"] = [m.value for m in InitMode]
                details["available_patterns"] = sorted(PATTERNS.keys())
                details["canonical_rules"] = {
                    "conway_life": "B3/S23",
                    "highlife": "B36/S23",
                    "seeds": "B2/S",
                    "life_without_death": "B3/S012345678",
                    "day_and_night": "B3678/S34678",
                }

            return details
        except Exception as e:
            return {"error": f"Failed to inspect simulation '{simulation_name}': {e}"}


# ---------------------------------------------------------------------------
# Tool 3: Run Deterministic Experiment
# ---------------------------------------------------------------------------


class RunExperimentInput(BaseModel):
    simulation_type: str = Field(
        "cellular_automaton",
        description="Type of simulation to execute (e.g. 'cellular_automaton')",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the simulation (e.g. rule, boundary, init_mode, init_density)",
    )
    seed: int = Field(42, description="Random seed for deterministic reproducibility")
    steps: int = Field(50, ge=1, le=1000, description="Number of simulation steps to run")
    grid_rows: int = Field(32, ge=4, le=256, description="Grid height (rows)")
    grid_cols: int = Field(32, ge=4, le=256, description="Grid width (cols)")


class RunExperimentTool(BaseTool):
    """Execute a deterministic simulation and compute macroscopic observables."""

    name = "run_experiment"
    description = (
        "Execute a deterministic scientific simulation with specified parameters, "
        "and return computed empirical observables, convergence metrics, and summary data."
    )
    args_schema = RunExperimentInput

    def __init__(self, service: ExperimentService | None = None) -> None:
        self._service = service or ExperimentService()

    def execute(
        self,
        simulation_type: str = "cellular_automaton",
        parameters: dict[str, Any] | None = None,
        seed: int = 42,
        steps: int = 50,
        grid_rows: int = 32,
        grid_cols: int = 32,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        try:
            stype = SimulationType(simulation_type)
        except ValueError:
            return {"error": f"Invalid simulation type '{simulation_type}'"}

        experiment = self._service.run_experiment(
            simulation_type=stype,
            parameters=parameters,
            seed=seed,
            steps=steps,
            grid_size=(grid_rows, grid_cols),
        )

        if experiment.status != "COMPLETED" or experiment.result is None:
            return {
                "experiment_id": experiment.experiment_id,
                "status": experiment.status.value,
                "error": experiment.error_message,
            }

        res = experiment.result
        return {
            "experiment_id": experiment.experiment_id,
            "status": experiment.status.value,
            "seed": seed,
            "steps_run": steps,
            "grid_size": [grid_rows, grid_cols],
            "runtime_seconds": round(res.runtime_seconds, 4),
            "observables": res.observables,
            "time_series_summary": {
                key: {
                    "initial": round(series[0], 4) if series else None,
                    "final": round(series[-1], 4) if series else None,
                    "min": round(min(series), 4) if series else None,
                    "max": round(max(series), 4) if series else None,
                }
                for key, series in res.time_series.items()
            },
        }


# ---------------------------------------------------------------------------
# Tool 4: Compare Experiments
# ---------------------------------------------------------------------------


class CompareExperimentsInput(BaseModel):
    experiments: list[RunExperimentInput] = Field(
        ...,
        min_length=2,
        max_length=10,
        description="List of experiment specifications to run and contrast",
    )


class CompareExperimentsTool(BaseTool):
    """Run multiple parameter variations and contrast empirical observables."""

    name = "compare_experiments"
    description = (
        "Run two or more parameter variations of an experiment and produce "
        "a side-by-side comparative table of empirical metrics."
    )
    args_schema = CompareExperimentsInput

    def __init__(self, service: ExperimentService | None = None) -> None:
        self._runner = RunExperimentTool(service=service)

    def execute(
        self,
        experiments: list[dict[str, Any]] | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        exp_list = experiments or []
        results: list[dict[str, Any]] = []
        for idx, exp_spec in enumerate(exp_list):
            res = self._runner.execute(**exp_spec)
            results.append({"index": idx, "spec": exp_spec, "result": res})

        comparison_table: list[dict[str, Any]] = []
        for r in results:
            res_dict: dict[str, Any] = r["result"]
            obs: dict[str, Any] = res_dict.get("observables", {})
            comparison_table.append(
                {
                    "experiment_id": res_dict.get("experiment_id"),
                    "rule": r["spec"].get("parameters", {}).get("rule", "default"),
                    "boundary": r["spec"].get("parameters", {}).get("boundary", "periodic"),
                    "mean_density": obs.get("mean_density"),
                    "final_entropy": obs.get("final_entropy"),
                    "is_periodic": obs.get("is_periodic"),
                    "period": obs.get("period"),
                    "mean_activity": obs.get("mean_activity"),
                    "is_extinct": obs.get("is_extinct"),
                }
            )

        return {
            "experiments_run": len(results),
            "comparison": comparison_table,
            "raw_results": results,
        }


# ---------------------------------------------------------------------------
# Tool 5: Parameter Sweep
# ---------------------------------------------------------------------------


class ParameterSweepInput(BaseModel):
    parameter_name: str = Field(
        ...,
        description="Name of parameter to vary (e.g. 'init_density', 'boundary', 'rule')",
    )
    values: list[Any] = Field(
        ...,
        min_length=2,
        max_length=20,
        description="Discrete values to sweep over",
    )
    simulation_type: str = Field(
        default="cellular_automaton",
        description="Simulation identifier",
    )
    base_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Base parameters held constant across the sweep",
    )
    seed: int = Field(default=42, description="Random seed")
    steps: int = Field(default=50, ge=1, le=500, description="Steps per simulation run")
    grid_rows: int = Field(default=32, ge=4, le=256, description="Grid rows")
    grid_cols: int = Field(default=32, ge=4, le=256, description="Grid cols")


class ParameterSweepTool(BaseTool):
    """Execute a systematic sweep over a parameter dimension to detect phase transitions."""

    name = "parameter_sweep"
    description = (
        "Run an automated parameter sweep over a simulation parameter and collect "
        "observables to discover critical thresholds, phase transitions, and scaling laws."
    )
    args_schema = ParameterSweepInput

    def __init__(self, service: ExperimentService | None = None) -> None:
        self._runner = RunExperimentTool(service=service)

    def execute(
        self,
        parameter_name: str = "init_density",
        values: list[Any] | None = None,
        simulation_type: str = "cellular_automaton",
        base_parameters: dict[str, Any] | None = None,
        seed: int = 42,
        steps: int = 50,
        grid_rows: int = 32,
        grid_cols: int = 32,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        val_list = values or [0.1, 0.2, 0.3, 0.4, 0.5]
        base_params = dict(base_parameters or {})
        sweep_results: list[dict[str, Any]] = []

        for val in val_list:
            current_params = dict(base_params)
            current_params[parameter_name] = val

            run_res = self._runner.execute(
                simulation_type=simulation_type,
                parameters=current_params,
                seed=seed,
                steps=steps,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
            )

            obs = run_res.get("observables", {})
            sweep_results.append({
                "value": val,
                "status": run_res.get("status"),
                "mean_density": obs.get("mean_density"),
                "final_entropy": obs.get("final_entropy"),
                "is_periodic": obs.get("is_periodic"),
                "period": obs.get("period"),
                "mean_activity": obs.get("mean_activity"),
                "is_extinct": obs.get("is_extinct"),
            })

        return {
            "parameter_name": parameter_name,
            "values_evaluated": len(val_list),
            "sweep_results": sweep_results,
        }


# ---------------------------------------------------------------------------
# Tool 6: Perturbation Sensitivity / Damage Spreading
# ---------------------------------------------------------------------------


class DamageSpreadingInput(BaseModel):
    rule: str = Field(default="B3/S23", description="Cellular automaton rule string")
    density: float = Field(default=0.25, ge=0.0, le=1.0, description="Initial random density")
    seed: int = Field(default=42, description="Random seed")
    steps: int = Field(default=50, ge=1, le=500, description="Simulation steps")
    grid_rows: int = Field(default=32, ge=4, le=128, description="Grid rows")
    grid_cols: int = Field(default=32, ge=4, le=128, description="Grid cols")
    boundary: str = Field(default="periodic", description="Boundary condition")
    perturbation_cells: int = Field(default=1, ge=1, le=10, description="Bits flipped at t=0")


class DamageSpreadingTool(BaseTool):
    """Analyze perturbation growth (butterfly effect / damage spreading) between twin runs."""

    name = "measure_damage_spreading"
    description = (
        "Quantify sensitivity to initial conditions by measuring the Hamming distance "
        "trajectory between an unperturbed run and a 1-bit flipped run."
    )
    args_schema = DamageSpreadingInput

    def execute(
        self,
        rule: str = "B3/S23",
        density: float = 0.25,
        seed: int = 42,
        steps: int = 50,
        grid_rows: int = 32,
        grid_cols: int = 32,
        boundary: str = "periodic",
        perturbation_cells: int = 1,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        sim_a = CellularAutomatonSimulation()
        sim_b = CellularAutomatonSimulation()

        params: dict[str, Any] = {
            "rule": rule,
            "boundary": BoundaryCondition(boundary),
            "init_mode": InitMode.RANDOM,
            "init_density": density,
            "grid_size": (grid_rows, grid_cols),
            "steps": steps,
        }

        sim_a.initialize(seed=seed, parameters=params)
        sim_b.initialize(seed=seed, parameters=params)

        # Perturb sim_b by flipping bit(s) at grid center
        grid_b = sim_b.current_state.copy()  # type: ignore[union-attr]
        cy, cx = grid_rows // 2, grid_cols // 2
        for i in range(perturbation_cells):
            py = (cy + (i // grid_cols)) % grid_rows
            px = (cx + (i % grid_cols)) % grid_cols
            grid_b[py, px] = 1 - grid_b[py, px]
        sim_b._current_state = grid_b

        traj_a = sim_a.run(steps=steps)
        traj_b = sim_b.run(steps=steps)

        analysis = compute_damage_spreading(traj_a, traj_b)
        classification = (
            "ordered / stable (healed)"
            if analysis["is_healed"]
            else (
                "chaotic (spreading damage)"
                if analysis["final_damage"] > perturbation_cells * 5
                else "complex / localized"
            )
        )

        return {
            "rule": rule,
            "initial_damage": analysis["initial_damage"],
            "final_damage": analysis["final_damage"],
            "max_damage": analysis["max_damage"],
            "mean_damage": analysis["mean_damage"],
            "is_healed": analysis["is_healed"],
            "dynamical_classification": classification,
            "damage_trajectory": analysis["hamming_series"],
        }


# ---------------------------------------------------------------------------
# Tool 7: Render Simulation Animation (GIF)
# ---------------------------------------------------------------------------


class RenderAnimationInput(BaseModel):
    rule: str = Field(default="B3/S23", description="Cellular automaton rule string")
    pattern: str | None = Field(
        default=None,
        description="Preset pattern name (e.g. glider, pulsar)",
    )
    density: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Random density if pattern is None",
    )
    seed: int = Field(default=42, description="Random seed")
    steps: int = Field(default=40, ge=5, le=300, description="Animation frames")
    grid_rows: int = Field(default=32, ge=8, le=128, description="Grid rows")
    grid_cols: int = Field(default=32, ge=8, le=128, description="Grid cols")
    boundary: str = Field(default="periodic", description="Boundary condition")
    output_path: str = Field(default="artifacts/animation.gif", description="Output GIF path")
    fps: int = Field(default=10, ge=1, le=30, description="Frames per second")


class RenderAnimationTool(BaseTool):
    """Render a cellular automaton trajectory into an animated GIF file."""

    name = "render_animation"
    description = (
        "Execute a simulation and compile its full 2D grid trajectory into an animated GIF."
    )
    args_schema = RenderAnimationInput

    def execute(
        self,
        rule: str = "B3/S23",
        pattern: str | None = None,
        density: float = 0.25,
        seed: int = 42,
        steps: int = 40,
        grid_rows: int = 32,
        grid_cols: int = 32,
        boundary: str = "periodic",
        output_path: str = "artifacts/animation.gif",
        fps: int = 10,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        sim = CellularAutomatonSimulation()
        init_mode = InitMode.PATTERN if pattern else InitMode.RANDOM
        sim.initialize(
            seed=seed,
            parameters={
                "rule": rule,
                "boundary": BoundaryCondition(boundary),
                "init_mode": init_mode,
                "init_density": density,
                "init_pattern": pattern,
                "grid_size": (grid_rows, grid_cols),
                "steps": steps,
            },
        )
        trajectory = sim.run(steps=steps)
        saved_path = render_trajectory_gif(trajectory, output_path=Path(output_path), fps=fps)
        return {
            "status": "COMPLETED",
            "artifact_type": "animation",
            "file_path": str(saved_path),
            "frames": len(trajectory),
            "fps": fps,
        }


# ---------------------------------------------------------------------------
# Tool 8: Plot Simulation Dynamics (Time Series)
# ---------------------------------------------------------------------------


class PlotDynamicsInput(BaseModel):
    rule: str = Field(default="B3/S23", description="CA rule string")
    density: float = Field(default=0.25, ge=0.0, le=1.0, description="Initial density")
    seed: int = Field(default=42, description="Random seed")
    steps: int = Field(default=60, ge=10, le=500, description="Simulation steps")
    grid_rows: int = Field(default=32, ge=8, le=256, description="Grid rows")
    grid_cols: int = Field(default=32, ge=8, le=256, description="Grid cols")
    boundary: str = Field(default="periodic", description="Boundary condition")
    output_path: str = Field(default="artifacts/dynamics.png", description="Output figure path")


class PlotDynamicsTool(BaseTool):
    """Run an experiment and plot its macroscopic dynamics (density, entropy, activity)."""

    name = "plot_dynamics"
    description = (
        "Execute a simulation run and generate a multi-panel publication plot "
        "tracking density, spatial Shannon entropy, and activity trajectories over time."
    )
    args_schema = PlotDynamicsInput

    def __init__(self, service: ExperimentService | None = None) -> None:
        self._service = service or ExperimentService()

    def execute(
        self,
        rule: str = "B3/S23",
        density: float = 0.25,
        seed: int = 42,
        steps: int = 60,
        grid_rows: int = 32,
        grid_cols: int = 32,
        boundary: str = "periodic",
        output_path: str = "artifacts/dynamics.png",
        **_kwargs: Any,
    ) -> dict[str, Any]:
        exp = self._service.run_experiment(
            simulation_type=SimulationType.CELLULAR_AUTOMATON,
            parameters={
                "rule": rule,
                "boundary": boundary,
                "init_mode": "random",
                "init_density": density,
            },
            seed=seed,
            steps=steps,
            grid_size=(grid_rows, grid_cols),
        )

        if exp.status != "COMPLETED" or exp.result is None:
            return {"error": f"Experiment failed: {exp.error_message}"}

        saved_path = plot_time_series(
            time_series=exp.result.time_series,
            output_path=Path(output_path),
            title=f"Macroscopic Dynamics: Rule {rule} (Seed {seed})",
        )

        return {
            "status": "COMPLETED",
            "artifact_type": "figure",
            "file_path": str(saved_path),
            "observables": exp.result.observables,
        }


# ---------------------------------------------------------------------------
# Tool 9: Plot Parameter Sweep
# ---------------------------------------------------------------------------


class PlotSweepInput(BaseModel):
    parameter_name: str = Field(default="init_density", description="Parameter to sweep")
    values: list[Any] = Field(default=[0.05, 0.15, 0.25, 0.35, 0.45], description="Values to test")
    rule: str = Field(default="B3/S23", description="Base rule")
    steps: int = Field(default=50, ge=10, le=200, description="Steps per run")
    grid_rows: int = Field(default=32, ge=8, le=128, description="Grid rows")
    grid_cols: int = Field(default=32, ge=8, le=128, description="Grid cols")
    output_path: str = Field(default="artifacts/sweep.png", description="Output figure path")


class PlotSweepTool(BaseTool):
    """Execute a parameter sweep and plot the order parameter curves."""

    name = "plot_sweep"
    description = (
        "Execute a parameter sweep and generate an order-parameter phase diagram "
        "plotting mean density and final entropy across the parameter variation."
    )
    args_schema = PlotSweepInput

    def __init__(self, service: ExperimentService | None = None) -> None:
        self._sweep_runner = ParameterSweepTool(service=service)

    def execute(
        self,
        parameter_name: str = "init_density",
        values: list[Any] | None = None,
        rule: str = "B3/S23",
        steps: int = 50,
        grid_rows: int = 32,
        grid_cols: int = 32,
        output_path: str = "artifacts/sweep.png",
        **_kwargs: Any,
    ) -> dict[str, Any]:
        val_list = values or [0.05, 0.15, 0.25, 0.35, 0.45]
        sweep_data = self._sweep_runner.execute(
            parameter_name=parameter_name,
            values=val_list,
            base_parameters={"rule": rule},
            steps=steps,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
        )

        saved_path = plot_parameter_sweep(
            sweep_results=sweep_data["sweep_results"],
            parameter_name=parameter_name,
            output_path=Path(output_path),
            title=f"Phase Transition Analysis: {parameter_name} (Rule {rule})",
        )

        return {
            "status": "COMPLETED",
            "artifact_type": "figure",
            "file_path": str(saved_path),
            "values_evaluated": len(val_list),
            "sweep_results": sweep_data["sweep_results"],
        }


# ---------------------------------------------------------------------------
# Tool Registry Factory
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Manages available scientific tools and provides schema exports."""

    def __init__(self, service: ExperimentService | None = None) -> None:
        self._service = service or ExperimentService()
        self._tools: dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        self.register(ListSimulationsTool())
        self.register(InspectSimulationTool())
        self.register(RunExperimentTool(service=self._service))
        self.register(CompareExperimentsTool(service=self._service))
        self.register(ParameterSweepTool(service=self._service))
        self.register(DamageSpreadingTool())
        self.register(RenderAnimationTool())
        self.register(PlotDynamicsTool(service=self._service))
        self.register(PlotSweepTool(service=self._service))

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry.")
        return self._tools[name]

    def list_tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return function calling schemas for all registered tools."""
        return [tool.to_json_schema() for tool in self._tools.values()]
