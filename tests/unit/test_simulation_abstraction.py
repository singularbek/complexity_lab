"""Unit tests for the simulation protocol, BaseSimulation, and SimulationRegistry."""

from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel, Field

from complexity_lab.domain.exceptions import SimulationError
from complexity_lab.simulations.base import BaseSimulation
from complexity_lab.simulations.interfaces import Simulation, SimulationRegistry


class DummyParams(BaseModel):
    increment: int = Field(1, ge=1)


class DummySimulation(BaseSimulation[DummyParams]):
    def __init__(self) -> None:
        super().__init__(name="dummy_test", param_schema=DummyParams)

    def initialize(self, seed: int, parameters: DummyParams | dict[str, Any]) -> None:
        self._seed = seed
        self._params = self.validate_parameters(parameters)
        self._step_count = 0
        self._current_state = np.zeros((3, 3), dtype=np.int32)

    def step(self) -> np.ndarray:
        if self._current_state is None or self._params is None:
            raise SimulationError("Simulation uninitialized.")
        self._current_state = self._current_state + self._params.increment
        self._step_count += 1
        return self._current_state


@pytest.mark.unit
def test_simulation_protocol_compliance() -> None:
    sim = DummySimulation()
    assert isinstance(sim, Simulation)
    assert sim.name == "dummy_test"
    assert sim.current_state is None
    assert sim.current_step == 0


@pytest.mark.unit
def test_dummy_simulation_execution() -> None:
    sim = DummySimulation()
    sim.initialize(seed=42, parameters={"increment": 2})

    assert sim.current_step == 0
    np.testing.assert_array_equal(sim.current_state, np.zeros((3, 3)))

    state1 = sim.step()
    assert sim.current_step == 1
    np.testing.assert_array_equal(state1, np.full((3, 3), 2))

    trajectory = sim.run(steps=3)
    # initial state + 3 steps = 4 states total
    assert len(trajectory) == 4
    assert sim.current_step == 4
    np.testing.assert_array_equal(trajectory[-1], np.full((3, 3), 8))


@pytest.mark.unit
def test_simulation_run_uninitialized_error() -> None:
    sim = DummySimulation()
    with pytest.raises(SimulationError, match="Cannot run simulation before calling initialize"):
        sim.run(steps=5)


@pytest.mark.unit
def test_simulation_invalid_parameters_error() -> None:
    sim = DummySimulation()
    with pytest.raises(SimulationError, match="Invalid parameters"):
        sim.validate_parameters({"increment": -5})


@pytest.mark.unit
def test_simulation_registry() -> None:
    reg_name = "test_registered_dummy"

    @SimulationRegistry.register(reg_name)
    class RegisteredDummy(DummySimulation):
        pass

    assert reg_name in SimulationRegistry.list_simulations()
    cls_found = SimulationRegistry.get(reg_name)
    assert cls_found is RegisteredDummy

    # Duplicate registration error
    with pytest.raises(SimulationError, match="already registered"):
        SimulationRegistry.register(reg_name)(RegisteredDummy)

    # Missing lookup error
    with pytest.raises(SimulationError, match="not found in registry"):
        SimulationRegistry.get("non_existent_sim")
