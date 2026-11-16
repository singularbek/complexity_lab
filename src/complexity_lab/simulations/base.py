"""Abstract base implementation providing reusable boilerplate for concrete engines."""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import numpy as np
from pydantic import BaseModel

from complexity_lab.domain.exceptions import SimulationError

TParams = TypeVar("TParams", bound=BaseModel)


class BaseSimulation(Generic[TParams], ABC):
    """Base class for simulations implementing standard step and run loops."""

    def __init__(self, name: str, param_schema: type[TParams]) -> None:
        self._name = name
        self._param_schema = param_schema
        self._current_state: np.ndarray | None = None
        self._step_count: int = 0
        self._params: TParams | None = None
        self._seed: int | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def current_state(self) -> np.ndarray | None:
        return self._current_state

    @property
    def current_step(self) -> int:
        return self._step_count

    def validate_parameters(self, parameters: dict[str, Any] | BaseModel) -> TParams:
        if isinstance(parameters, self._param_schema):
            return parameters
        try:
            return self._param_schema.model_validate(parameters)
        except Exception as e:
            raise SimulationError(f"Invalid parameters for {self.name}: {e}") from e

    @abstractmethod
    def initialize(self, seed: int, parameters: TParams | dict[str, Any]) -> None:
        """Initialize the simulation state."""
        ...

    @abstractmethod
    def step(self) -> np.ndarray:
        """Advance the simulation by one timestep and return the new state."""
        ...

    def run(self, steps: int) -> list[np.ndarray]:
        if self._current_state is None:
            raise SimulationError("Cannot run simulation before calling initialize().")
        if steps < 1:
            raise SimulationError("Steps parameter must be at least 1.")

        trajectory = [self._current_state.copy()]
        for _ in range(steps):
            state = self.step()
            trajectory.append(state.copy())
        return trajectory
