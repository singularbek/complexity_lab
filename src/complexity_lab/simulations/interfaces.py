"""Interfaces, base protocols, and registries for simulation engines."""

from collections.abc import Callable
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np
from pydantic import BaseModel

from complexity_lab.domain.exceptions import SimulationError

TState = TypeVar("TState")
TParams = TypeVar("TParams", bound=BaseModel)
TSim = TypeVar("TSim")


@runtime_checkable
class Simulation(Protocol):
    """Generic interface for all scientific simulations in the platform."""

    @property
    def name(self) -> str:
        """Unique identifier name for the simulation."""
        ...

    def validate_parameters(self, parameters: dict[str, Any]) -> BaseModel:
        """Validate raw dictionary parameters against the simulation's schema."""
        ...

    def initialize(self, seed: int, parameters: BaseModel | dict[str, Any]) -> None:
        """Initialize simulation state deterministically using seed and parameters."""
        ...

    def step(self) -> np.ndarray:
        """Advance the simulation by one timestep and return the new state."""
        ...

    def run(self, steps: int) -> list[np.ndarray]:
        """Run simulation for `steps` timesteps and return trajectory states."""
        ...

    @property
    def current_state(self) -> np.ndarray | None:
        """Return current state array or None if uninitialized."""
        ...

    @property
    def current_step(self) -> int:
        """Return the current step counter."""
        ...


class SimulationRegistry:
    """Central registry for discovering and instantiating available simulations."""

    _registry: dict[str, type[Any]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[TSim]], type[TSim]]:
        """Decorator to register a simulation class under a unique key."""

        def decorator(subclass: type[TSim]) -> type[TSim]:
            if name in cls._registry:
                raise SimulationError(f"Simulation '{name}' is already registered.")
            cls._registry[name] = subclass
            return subclass

        return decorator

    @classmethod
    def get(cls, name: str) -> type[Any]:
        """Retrieve a simulation class by name."""
        if name not in cls._registry:
            raise SimulationError(f"Simulation '{name}' not found in registry.")
        return cls._registry[name]

    @classmethod
    def list_simulations(cls) -> list[str]:
        """List all available registered simulation names."""
        return sorted(cls._registry.keys())
