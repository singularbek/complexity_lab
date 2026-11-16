"""Unit tests for the cellular automaton engine components."""

import numpy as np
import pytest

from complexity_lab.domain.exceptions import DomainValidationError, SimulationError
from complexity_lab.simulations.cellular_automaton import (
    PATTERNS,
    BoundaryCondition,
    CellularAutomatonSimulation,
    InitMode,
    LifeRule,
    NeighborhoodType,
    apply_boundary,
    count_neighbors,
    embed_pattern,
    initialize_grid,
)
from complexity_lab.simulations.interfaces import Simulation, SimulationRegistry


@pytest.mark.unit
def test_boundary_periodic() -> None:
    grid = np.array(
        [
            [1, 2],
            [3, 4],
        ]
    )
    padded = apply_boundary(grid, BoundaryCondition.PERIODIC, pad_width=1)
    assert padded.shape == (4, 4)
    # Check toroidal corners and edges
    assert padded[0, 0] == 4
    assert padded[0, 3] == 3
    assert padded[3, 0] == 2
    assert padded[3, 3] == 1
    np.testing.assert_array_equal(padded[1:3, 1:3], grid)


@pytest.mark.unit
def test_boundary_fixed() -> None:
    grid = np.array(
        [
            [1, 1],
            [1, 1],
        ]
    )
    padded = apply_boundary(grid, BoundaryCondition.FIXED, pad_width=1, fixed_value=0)
    assert padded.shape == (4, 4)
    assert np.all(padded[0, :] == 0)
    assert np.all(padded[-1, :] == 0)
    assert np.all(padded[:, 0] == 0)
    assert np.all(padded[:, -1] == 0)
    np.testing.assert_array_equal(padded[1:3, 1:3], grid)


@pytest.mark.unit
def test_boundary_reflective() -> None:
    grid = np.array(
        [
            [1, 2],
            [3, 4],
        ]
    )
    padded = apply_boundary(grid, BoundaryCondition.REFLECTIVE, pad_width=1)
    assert padded.shape == (4, 4)
    assert padded[0, 1] == 1
    assert padded[1, 0] == 1
    np.testing.assert_array_equal(padded[1:3, 1:3], grid)


@pytest.mark.unit
def test_neighborhood_counts() -> None:
    # 3x3 grid with central cell alive
    grid = np.zeros((3, 3), dtype=np.int8)
    grid[1, 1] = 1

    moore = count_neighbors(grid, NeighborhoodType.MOORE, BoundaryCondition.FIXED)
    von_neumann = count_neighbors(grid, NeighborhoodType.VON_NEUMANN, BoundaryCondition.FIXED)

    # All 8 neighbors should have count 1
    expected_moore = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
    )
    np.testing.assert_array_equal(moore, expected_moore)

    # 4 orthogonal neighbors have count 1, corners have count 0
    expected_vn = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]
    )
    np.testing.assert_array_equal(von_neumann, expected_vn)


@pytest.mark.unit
def test_life_rule_parsing() -> None:
    conway = LifeRule.from_string("B3/S23")
    assert conway.birth == frozenset({3})
    assert conway.survival == frozenset({2, 3})
    assert conway.rule_string == "B3/S23"

    # Reverse format S23/B3
    conway_rev = LifeRule.from_string("S23/B3")
    assert conway_rev.birth == frozenset({3})
    assert conway_rev.survival == frozenset({2, 3})

    # HighLife B36/S23
    highlife = LifeRule.from_string("B36/S23")
    assert highlife.birth == frozenset({3, 6})
    assert highlife.survival == frozenset({2, 3})

    # Invalid rule format
    with pytest.raises(DomainValidationError):
        LifeRule.from_string("INVALID_RULE")


@pytest.mark.unit
def test_grid_initialization_modes() -> None:
    # Empty
    empty = initialize_grid((10, 10), init_mode=InitMode.EMPTY)
    assert np.all(empty == 0)

    # Random deterministic seed
    rnd1 = initialize_grid((20, 20), init_mode=InitMode.RANDOM, seed=42, density=0.3)
    rnd2 = initialize_grid((20, 20), init_mode=InitMode.RANDOM, seed=42, density=0.3)
    rnd_other = initialize_grid((20, 20), init_mode=InitMode.RANDOM, seed=99, density=0.3)

    np.testing.assert_array_equal(rnd1, rnd2)
    assert not np.array_equal(rnd1, rnd_other)

    # Pattern
    glider_grid = initialize_grid((10, 10), init_mode=InitMode.PATTERN, pattern_name="glider")
    assert np.sum(glider_grid) == np.sum(PATTERNS["glider"])

    # Custom
    custom_in = np.ones((5, 5), dtype=np.int8)
    custom_out = initialize_grid((5, 5), init_mode=InitMode.CUSTOM, custom_grid=custom_in)
    np.testing.assert_array_equal(custom_out, custom_in)


@pytest.mark.unit
def test_embed_pattern_out_of_bounds() -> None:
    grid = np.zeros((5, 5), dtype=np.int8)
    pulsar = PATTERNS["pulsar"]  # 13x13
    with pytest.raises(DomainValidationError, match="exceeds grid dimensions"):
        embed_pattern(grid, pulsar)


@pytest.mark.unit
def test_cellular_automaton_simulation_lifecycle() -> None:
    sim = CellularAutomatonSimulation()
    assert isinstance(sim, Simulation)
    assert sim.name == "cellular_automaton"
    assert sim.current_state is None
    assert sim.current_step == 0

    sim.initialize(
        seed=42,
        parameters={
            "grid_size": (16, 16),
            "init_mode": "pattern",
            "init_pattern": "blinker",
        },
    )
    assert sim.current_state is not None
    assert sim.current_step == 0
    assert len(sim.alive_counts) == 1
    assert sim.alive_counts[0] == 3

    state1 = sim.step()
    assert state1.shape == (16, 16)
    assert sim.current_step == 1
    assert len(sim.alive_counts) == 2
    assert sim.alive_counts[1] == 3
    assert sim.activity_history[-1] > 0

    trajectory = sim.run(steps=4)
    # 1 initial state + 4 steps = 5 states in trajectory
    assert len(trajectory) == 5
    assert sim.current_step == 5


@pytest.mark.unit
def test_simulation_registry_discovery() -> None:
    cls_found = SimulationRegistry.get("cellular_automaton")
    assert cls_found is CellularAutomatonSimulation
    assert "cellular_automaton" in SimulationRegistry.list_simulations()


@pytest.mark.unit
def test_uninitialized_step_error() -> None:
    sim = CellularAutomatonSimulation()
    with pytest.raises(SimulationError, match="Cannot step uninitialized"):
        sim.step()
