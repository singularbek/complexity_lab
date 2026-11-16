"""Deterministic analysis engine and observables for complex systems simulations."""

import math
from typing import Any

import numpy as np

from complexity_lab.domain.artifacts import ExperimentResult


def compute_density(grid: np.ndarray) -> float:
    """Calculate the fraction of active cells in a binary grid."""
    if grid.size == 0:
        return 0.0
    return float(np.mean(grid != 0))


def compute_shannon_entropy(grid: np.ndarray) -> float:
    """Calculate binary spatial Shannon entropy in bits for a 2D grid.

    H = - p * log2(p) - (1-p) * log2(1-p)
    """
    if grid.size == 0:
        return 0.0
    p = float(np.mean(grid != 0))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p))


def compute_block_entropy(grid: np.ndarray, block_size: int = 2) -> float:
    """Calculate block spatial entropy over 2D blocks of size block_size x block_size.

    Discretizes the grid into non-overlapping blocks and calculates the Shannon entropy
    of the configuration probability distribution.
    """
    h, w = grid.shape
    bh = h // block_size
    bw = w // block_size
    if bh == 0 or bw == 0:
        return 0.0

    cropped = grid[: bh * block_size, : bw * block_size]
    # Reshape into blocks and convert each block to an integer representation
    blocks = cropped.reshape(bh, block_size, bw, block_size).swapaxes(1, 2)
    flat_blocks = blocks.reshape(bh * bw, block_size * block_size)

    # Encode binary configurations as integers
    powers = 2 ** np.arange(block_size * block_size, dtype=np.int64)
    block_ints = flat_blocks.dot(powers)

    _, counts = np.unique(block_ints, return_counts=True)
    probs = counts / (bh * bw)
    return float(-np.sum(probs * np.log2(probs)))


def detect_periodicity(
    trajectory: list[np.ndarray],
    max_period: int = 32,
) -> tuple[bool, int]:
    """Detect whether the tail of a trajectory has settled into a periodic orbit.

    Returns:
        (is_periodic, period) where period is 1 for fixed point, >= 2 for cycle, or 0 if none.
    """
    if len(trajectory) < 2:
        return (False, 0)

    last_state = trajectory[-1]
    # Search backwards for a matching state
    for p in range(1, min(max_period + 1, len(trajectory))):
        candidate_state = trajectory[-1 - p]
        if np.array_equal(last_state, candidate_state):
            # Check if intermediate states also cycle (confirm period p)
            if len(trajectory) >= 2 * p:
                confirmed = True
                for offset in range(p):
                    if not np.array_equal(
                        trajectory[-1 - offset],
                        trajectory[-1 - p - offset],
                    ):
                        confirmed = False
                        break
                if confirmed:
                    return (True, p)
            else:
                return (True, p)

    return (False, 0)


def summarize_trajectory(
    trajectory: list[np.ndarray],
    activity_history: list[int] | None = None,
    runtime_seconds: float = 0.0,
) -> ExperimentResult:
    """Compute comprehensive scientific observables from a simulation trajectory.

    Args:
        trajectory: List of 2D grid states at each timestep t = 0..N.
        activity_history: Count of cell state changes per step.
        runtime_seconds: Execution time.

    Returns:
        ExperimentResult domain value object with metrics and time series.
    """
    if not trajectory:
        return ExperimentResult(runtime_seconds=runtime_seconds)

    densities = [compute_density(s) for s in trajectory]
    entropies = [compute_shannon_entropy(s) for s in trajectory]

    final_grid = trajectory[-1]
    is_periodic, period = detect_periodicity(trajectory)

    initial_alive = int(np.sum(trajectory[0] != 0))
    final_alive = int(np.sum(final_grid != 0))
    total_cells = final_grid.size

    mean_density = float(np.mean(densities))
    density_variance = float(np.var(densities))

    if activity_history and len(activity_history) > 1:
        mean_activity = float(np.mean(activity_history[1:]))
        final_activity = int(activity_history[-1])
    else:
        mean_activity = 0.0
        final_activity = 0

    observables: dict[str, Any] = {
        "steps": len(trajectory) - 1,
        "total_cells": total_cells,
        "initial_alive": initial_alive,
        "final_alive": final_alive,
        "mean_density": mean_density,
        "density_variance": density_variance,
        "final_entropy": entropies[-1],
        "is_periodic": is_periodic,
        "period": period,
        "mean_activity": mean_activity,
        "final_activity": final_activity,
        "is_extinct": final_alive == 0,
        "is_static": final_activity == 0,
    }

    time_series: dict[str, list[float]] = {
        "density": densities,
        "shannon_entropy": entropies,
    }
    if activity_history:
        time_series["activity"] = [float(a) for a in activity_history]

    return ExperimentResult(
        observables=observables,
        time_series=time_series,
        runtime_seconds=runtime_seconds,
    )


def compute_hamming_distance(grid_a: np.ndarray, grid_b: np.ndarray) -> int:
    """Calculate the Hamming distance (number of differing cells) between two grids."""
    if grid_a.shape != grid_b.shape:
        raise ValueError("Grids must possess identical shapes to compute Hamming distance.")
    return int(np.sum(grid_a != grid_b))


def compute_damage_spreading(
    trajectory_a: list[np.ndarray],
    trajectory_b: list[np.ndarray],
) -> dict[str, Any]:
    """Analyze perturbation growth (damage spreading) between twin simulation trajectories.

    Used to characterize dynamical regimes (Wolfram classes, Lyapunov divergence).
    """
    if len(trajectory_a) != len(trajectory_b):
        raise ValueError("Trajectories must have identical lengths.")

    hamming_series = [
        compute_hamming_distance(s1, s2)
        for s1, s2 in zip(trajectory_a, trajectory_b, strict=True)
    ]

    initial_damage = hamming_series[0] if hamming_series else 0
    final_damage = hamming_series[-1] if hamming_series else 0
    max_damage = max(hamming_series) if hamming_series else 0
    mean_damage = float(np.mean(hamming_series)) if hamming_series else 0.0

    return {
        "initial_damage": initial_damage,
        "final_damage": final_damage,
        "max_damage": max_damage,
        "mean_damage": mean_damage,
        "is_healed": final_damage == 0,
        "hamming_series": hamming_series,
    }
