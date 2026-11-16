"""Unit tests for the scientific analysis engine and observables."""

import numpy as np
import pytest

from complexity_lab.analysis.metrics import (
    compute_block_entropy,
    compute_damage_spreading,
    compute_density,
    compute_hamming_distance,
    compute_shannon_entropy,
    detect_periodicity,
    summarize_trajectory,
)


@pytest.mark.unit
def test_compute_density() -> None:
    assert compute_density(np.array([])) == 0.0

    zeros = np.zeros((4, 4), dtype=np.int8)
    assert compute_density(zeros) == 0.0

    ones = np.ones((4, 4), dtype=np.int8)
    assert compute_density(ones) == 1.0

    half = np.array([[1, 0], [0, 1]], dtype=np.int8)
    assert compute_density(half) == 0.5


@pytest.mark.unit
def test_compute_shannon_entropy() -> None:
    # Homogeneous states have zero entropy (no uncertainty)
    zeros = np.zeros((5, 5), dtype=np.int8)
    assert compute_shannon_entropy(zeros) == 0.0

    ones = np.ones((5, 5), dtype=np.int8)
    assert compute_shannon_entropy(ones) == 0.0

    # Equal distribution has maximum binary entropy of 1.0 bit
    balanced = np.array([[1, 0], [0, 1]], dtype=np.int8)
    assert pytest.approx(compute_shannon_entropy(balanced), abs=1e-5) == 1.0


@pytest.mark.unit
def test_compute_block_entropy() -> None:
    zeros = np.zeros((4, 4), dtype=np.int8)
    assert compute_block_entropy(zeros, block_size=2) == 0.0

    # Grid with 4 different 2x2 blocks
    grid = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.int8,
    )
    entropy = compute_block_entropy(grid, block_size=2)
    assert entropy > 0.0


@pytest.mark.unit
def test_detect_periodicity_fixed_point() -> None:
    state = np.array([[1, 0], [0, 1]])
    trajectory = [state, state, state]
    is_periodic, period = detect_periodicity(trajectory)
    assert is_periodic is True
    assert period == 1


@pytest.mark.unit
def test_detect_periodicity_oscillator() -> None:
    s0 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    s1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    trajectory = [s0, s1, s0, s1, s0, s1]
    is_periodic, period = detect_periodicity(trajectory)
    assert is_periodic is True
    assert period == 2


@pytest.mark.unit
def test_detect_periodicity_non_periodic() -> None:
    trajectory = [np.full((3, 3), i) for i in range(10)]
    is_periodic, period = detect_periodicity(trajectory)
    assert is_periodic is False
    assert period == 0


@pytest.mark.unit
def test_summarize_trajectory() -> None:
    s0 = np.zeros((4, 4), dtype=np.int8)
    s1 = np.zeros((4, 4), dtype=np.int8)
    s1[0, 0] = 1
    trajectory = [s0, s1]
    activity = [0, 1]

    result = summarize_trajectory(trajectory, activity_history=activity, runtime_seconds=0.05)
    assert result.runtime_seconds == 0.05
    assert "mean_density" in result.observables
    assert "final_entropy" in result.observables
    assert result.observables["initial_alive"] == 0
    assert result.observables["final_alive"] == 1
    assert "density" in result.time_series
    assert len(result.time_series["density"]) == 2


@pytest.mark.unit
def test_compute_hamming_distance() -> None:
    g1 = np.zeros((3, 3), dtype=np.int8)
    g2 = np.zeros((3, 3), dtype=np.int8)
    g2[1, 1] = 1
    assert compute_hamming_distance(g1, g2) == 1

    with pytest.raises(ValueError, match="identical shapes"):
        compute_hamming_distance(np.zeros((2, 2)), np.zeros((3, 3)))


@pytest.mark.unit
def test_compute_damage_spreading() -> None:
    g0 = np.zeros((3, 3), dtype=np.int8)
    g1 = np.ones((3, 3), dtype=np.int8)
    traj_a = [g0, g0]
    traj_b = [g0, g1]

    res = compute_damage_spreading(traj_a, traj_b)
    assert res["initial_damage"] == 0
    assert res["final_damage"] == 9
    assert res["is_healed"] is False
    assert res["hamming_series"] == [0, 9]

