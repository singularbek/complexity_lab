"""Cellular Automaton simulation engine and parameter schema."""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from complexity_lab.domain.exceptions import SimulationError
from complexity_lab.simulations.base import BaseSimulation
from complexity_lab.simulations.cellular_automaton.boundary import BoundaryCondition
from complexity_lab.simulations.cellular_automaton.initialization import (
    InitMode,
    initialize_grid,
)
from complexity_lab.simulations.cellular_automaton.neighborhood import (
    NeighborhoodType,
    count_neighbors,
)
from complexity_lab.simulations.cellular_automaton.rules import LifeRule
from complexity_lab.simulations.interfaces import SimulationRegistry


class CellularAutomatonParameters(BaseModel):
    """Configuration parameters for cellular automaton simulation."""

    steps: int = Field(100, ge=1, description="Number of simulation steps")
    grid_size: tuple[int, int] = Field((64, 64), description="Grid dimensions (rows, cols)")
    rule: str = Field("B3/S23", description="Transition rule string (e.g. 'B3/S23')")
    boundary: BoundaryCondition = Field(
        BoundaryCondition.PERIODIC,
        description="Boundary condition type",
    )
    neighborhood: NeighborhoodType = Field(
        NeighborhoodType.MOORE,
        description="Neighborhood topology type",
    )
    init_mode: InitMode = Field(InitMode.RANDOM, description="Grid initialization mode")
    init_density: float = Field(
        0.2,
        ge=0.0,
        le=1.0,
        description="Initial alive probability for random initialization",
    )
    init_pattern: str | None = Field(
        None,
        description="Standard pattern name for pattern initialization",
    )
    pattern_offset: tuple[int, int] | None = Field(
        None,
        description="Offset (row, col) for pattern placement",
    )
    custom_grid: list[list[int]] | None = Field(
        None,
        description="Custom binary grid matrix for custom initialization",
    )

    model_config = {"frozen": True}


@SimulationRegistry.register("cellular_automaton")
class CellularAutomatonSimulation(BaseSimulation[CellularAutomatonParameters]):
    """Deterministic 2D Cellular Automaton simulation engine."""

    def __init__(self) -> None:
        super().__init__(name="cellular_automaton", param_schema=CellularAutomatonParameters)
        self._rule: LifeRule | None = None
        self._alive_counts: list[int] = []
        self._activity_history: list[int] = []

    @property
    def rule(self) -> LifeRule | None:
        """Active parsed transition rule."""
        return self._rule

    @property
    def alive_counts(self) -> list[int]:
        """Trajectory history of alive cell counts per step."""
        return list(self._alive_counts)

    @property
    def activity_history(self) -> list[int]:
        """Trajectory history of cell state change counts per step."""
        return list(self._activity_history)

    def initialize(
        self,
        seed: int,
        parameters: CellularAutomatonParameters | dict[str, Any] | BaseModel,
    ) -> None:
        """Initialize the cellular automaton state deterministically.

        Args:
            seed: Integer seed for reproducible random generation.
            parameters: Simulation parameters dict or CellularAutomatonParameters instance.
        """
        self._seed = seed
        self._params = self.validate_parameters(parameters)

        try:
            self._rule = LifeRule.from_string(self._params.rule)
        except Exception as e:
            raise SimulationError(f"Invalid rule '{self._params.rule}': {e}") from e

        custom_array = (
            np.array(self._params.custom_grid, dtype=np.int8)
            if self._params.custom_grid is not None
            else None
        )

        try:
            self._current_state = initialize_grid(
                grid_size=self._params.grid_size,
                init_mode=self._params.init_mode,
                seed=self._seed,
                density=self._params.init_density,
                pattern_name=self._params.init_pattern,
                pattern_offset=self._params.pattern_offset,
                custom_grid=custom_array,
            )
        except Exception as e:
            raise SimulationError(f"Failed to initialize grid: {e}") from e

        self._step_count = 0
        initial_alive = int(np.sum(self._current_state))
        self._alive_counts = [initial_alive]
        self._activity_history = [0]

    def step(self) -> np.ndarray:
        """Advance the simulation by one timestep and return updated state.

        Returns:
            New 2D numpy array with updated cell states.
        """
        if self._current_state is None or self._params is None or self._rule is None:
            raise SimulationError("Cannot step uninitialized simulation. Call initialize() first.")

        neighbor_counts = count_neighbors(
            self._current_state,
            neighborhood=self._params.neighborhood,
            boundary=self._params.boundary,
        )

        next_state = self._rule.apply(self._current_state, neighbor_counts)
        changed_cells = int(np.sum(next_state != self._current_state))

        self._current_state = next_state
        self._step_count += 1
        self._alive_counts.append(int(np.sum(self._current_state)))
        self._activity_history.append(changed_cells)

        return self._current_state
