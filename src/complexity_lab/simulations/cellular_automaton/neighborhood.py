"""Neighborhood types and neighbor count computation for cellular automata."""

from enum import StrEnum
from typing import cast

import numpy as np

from complexity_lab.domain.exceptions import DomainValidationError
from complexity_lab.simulations.cellular_automaton.boundary import (
    BoundaryCondition,
    apply_boundary,
)


class NeighborhoodType(StrEnum):
    """Supported 2D neighborhood topologies."""

    MOORE = "moore"
    VON_NEUMANN = "von_neumann"


def count_neighbors(
    grid: np.ndarray,
    neighborhood: NeighborhoodType = NeighborhoodType.MOORE,
    boundary: BoundaryCondition = BoundaryCondition.PERIODIC,
) -> np.ndarray:
    """Compute the count of active neighbors for each cell in the 2D grid.

    Args:
        grid: 2D numpy array of binary cell states (0 or 1).
        neighborhood: Moore (8-neighbors) or Von Neumann (4-neighbors).
        boundary: Boundary condition to use for edge padding.

    Returns:
        2D numpy array of identical shape with neighbor counts.
    """
    if grid.ndim != 2:
        raise DomainValidationError(
            f"Expected a 2D grid, but got array with {grid.ndim} dimensions."
        )

    padded = apply_boundary(grid, boundary=boundary, pad_width=1)

    # Orthogonal neighbors (von Neumann)
    top = padded[:-2, 1:-1]
    bottom = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]

    if neighborhood == NeighborhoodType.VON_NEUMANN:
        return cast(np.ndarray, top + bottom + left + right)
    elif neighborhood == NeighborhoodType.MOORE:
        top_left = padded[:-2, :-2]
        top_right = padded[:-2, 2:]
        bottom_left = padded[2:, :-2]
        bottom_right = padded[2:, 2:]
        return cast(
            np.ndarray,
            top + bottom + left + right + top_left + top_right + bottom_left + bottom_right,
        )
    else:
        raise DomainValidationError(f"Unsupported neighborhood type: {neighborhood}")
