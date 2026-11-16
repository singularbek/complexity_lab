"""Boundary conditions for 2D cellular automata grids."""

from enum import StrEnum

import numpy as np

from complexity_lab.domain.exceptions import DomainValidationError


class BoundaryCondition(StrEnum):
    """Supported spatial boundary conditions."""

    PERIODIC = "periodic"
    FIXED = "fixed"
    REFLECTIVE = "reflective"


def apply_boundary(
    grid: np.ndarray,
    boundary: BoundaryCondition,
    pad_width: int = 1,
    fixed_value: int = 0,
) -> np.ndarray:
    """Pad the 2D grid according to the specified boundary condition.

    Args:
        grid: 2D numpy array of cell states.
        boundary: Boundary condition type.
        pad_width: Number of padding cells around borders.
        fixed_value: Fill value for FIXED boundary condition (default 0).

    Returns:
        Padded 2D numpy array.
    """
    if grid.ndim != 2:
        raise DomainValidationError(
            f"Expected a 2D grid, but got array with {grid.ndim} dimensions."
        )

    if boundary == BoundaryCondition.PERIODIC:
        return np.pad(grid, pad_width=pad_width, mode="wrap")
    elif boundary == BoundaryCondition.FIXED:
        return np.pad(grid, pad_width=pad_width, mode="constant", constant_values=fixed_value)
    elif boundary == BoundaryCondition.REFLECTIVE:
        return np.pad(grid, pad_width=pad_width, mode="symmetric")
    else:
        raise DomainValidationError(f"Unsupported boundary condition: {boundary}")
