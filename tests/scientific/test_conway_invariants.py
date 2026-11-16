"""Scientific verification tests for Conway's Game of Life invariants."""

import numpy as np
import pytest

from complexity_lab.simulations.cellular_automaton import (
    BoundaryCondition,
    CellularAutomatonSimulation,
)


@pytest.mark.scientific
@pytest.mark.parametrize("pattern", ["block", "beehive", "tub"])
def test_still_life_invariance(pattern: str) -> None:
    """Still life patterns must remain strictly identical at every step."""
    sim = CellularAutomatonSimulation()
    sim.initialize(
        seed=1,
        parameters={
            "grid_size": (16, 16),
            "init_mode": "pattern",
            "init_pattern": pattern,
            "boundary": BoundaryCondition.PERIODIC,
        },
    )
    initial_state = sim.current_state.copy()  # type: ignore[union-attr]

    trajectory = sim.run(steps=10)
    for step_idx, state in enumerate(trajectory):
        np.testing.assert_array_equal(
            state,
            initial_state,
            err_msg=f"Still life '{pattern}' changed state at step {step_idx}",
        )


@pytest.mark.scientific
def test_blinker_period_two_oscillator() -> None:
    """Blinker must oscillate between horizontal and vertical with period 2."""
    sim = CellularAutomatonSimulation()
    sim.initialize(
        seed=1,
        parameters={
            "grid_size": (16, 16),
            "init_mode": "pattern",
            "init_pattern": "blinker",
        },
    )
    s0 = sim.current_state.copy()  # type: ignore[union-attr]

    s1 = sim.step()
    assert not np.array_equal(s0, s1), "Blinker state at step 1 should differ from step 0"

    s2 = sim.step()
    np.testing.assert_array_equal(s2, s0, err_msg="Blinker should return to s0 after 2 steps")

    s3 = sim.step()
    np.testing.assert_array_equal(s3, s1, err_msg="Blinker should match s1 at step 3")


@pytest.mark.scientific
@pytest.mark.parametrize("pattern", ["toad", "beacon"])
def test_period_two_oscillators(pattern: str) -> None:
    """Toad and Beacon must return to their initial state after 2 steps."""
    sim = CellularAutomatonSimulation()
    sim.initialize(
        seed=1,
        parameters={
            "grid_size": (20, 20),
            "init_mode": "pattern",
            "init_pattern": pattern,
        },
    )
    s0 = sim.current_state.copy()  # type: ignore[union-attr]
    s1 = sim.step()
    assert not np.array_equal(s0, s1)

    s2 = sim.step()
    np.testing.assert_array_equal(s2, s0, err_msg=f"{pattern} should have period 2")


@pytest.mark.scientific
def test_pulsar_period_three_oscillator() -> None:
    """Pulsar must return to its initial state after exactly 3 steps."""
    sim = CellularAutomatonSimulation()
    sim.initialize(
        seed=1,
        parameters={
            "grid_size": (25, 25),
            "init_mode": "pattern",
            "init_pattern": "pulsar",
        },
    )
    s0 = sim.current_state.copy()  # type: ignore[union-attr]
    s1 = sim.step()
    s2 = sim.step()
    assert not np.array_equal(s0, s1)
    assert not np.array_equal(s1, s2)

    s3 = sim.step()
    np.testing.assert_array_equal(s3, s0, err_msg="Pulsar should have period 3")


@pytest.mark.scientific
def test_glider_translation() -> None:
    """A glider on a toroidal grid must translate diagonally by (1, 1) every 4 steps."""
    grid_size = (32, 32)
    sim = CellularAutomatonSimulation()
    sim.initialize(
        seed=1,
        parameters={
            "grid_size": grid_size,
            "init_mode": "pattern",
            "init_pattern": "glider",
            "pattern_offset": (5, 5),
            "boundary": BoundaryCondition.PERIODIC,
        },
    )
    s0 = sim.current_state.copy()  # type: ignore[union-attr]

    # Advance 4 steps (one full glider cycle)
    for _ in range(4):
        sim.step()

    s4 = sim.current_state.copy()  # type: ignore[union-attr]

    # Population must be conserved (5 alive cells)
    assert np.sum(s0) == 5
    assert np.sum(s4) == 5

    # Check that s4 is s0 rolled by (1, 1)
    expected_s4 = np.roll(s0, shift=(1, 1), axis=(0, 1))
    np.testing.assert_array_equal(
        s4,
        expected_s4,
        err_msg="Glider did not translate diagonally by (1, 1) after 4 steps",
    )


@pytest.mark.scientific
def test_empty_grid_and_single_cell_extinction() -> None:
    """Empty grid stays empty; single cell dies due to underpopulation."""
    sim = CellularAutomatonSimulation()
    sim.initialize(
        seed=1,
        parameters={
            "grid_size": (10, 10),
            "init_mode": "empty",
        },
    )
    s1 = sim.step()
    assert np.all(s1 == 0)

    # Single cell grid
    single = np.zeros((10, 10), dtype=np.int8)
    single[5, 5] = 1
    sim.initialize(
        seed=1,
        parameters={
            "grid_size": (10, 10),
            "init_mode": "custom",
            "custom_grid": single.tolist(),
        },
    )
    s1 = sim.step()
    assert np.all(s1 == 0), "Single live cell must die at next step"
