"""Initialization modes and canonical pattern templates for cellular automata."""

from enum import StrEnum
from typing import cast

import numpy as np

from complexity_lab.domain.exceptions import DomainValidationError


class InitMode(StrEnum):
    """Supported grid initialization strategies."""

    RANDOM = "random"
    PATTERN = "pattern"
    EMPTY = "empty"
    CUSTOM = "custom"


PATTERNS: dict[str, np.ndarray] = {
    # Still lifes
    "block": np.array(
        [
            [1, 1],
            [1, 1],
        ],
        dtype=np.int8,
    ),
    "beehive": np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=np.int8,
    ),
    "tub": np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=np.int8,
    ),
    # Oscillators
    "blinker": np.array(
        [
            [1, 1, 1],
        ],
        dtype=np.int8,
    ),
    "toad": np.array(
        [
            [0, 1, 1, 1],
            [1, 1, 1, 0],
        ],
        dtype=np.int8,
    ),
    "beacon": np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.int8,
    ),
    "pulsar": np.array(
        [
            [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
        ],
        dtype=np.int8,
    ),
    # Spaceships
    "glider": np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.int8,
    ),
    "lwss": np.array(
        [
            [0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 0],
        ],
        dtype=np.int8,
    ),
    # Guns
    "gosper_glider_gun": np.array(
        [
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
            ],
            [
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        ],
        dtype=np.int8,
    ),
}


def embed_pattern(
    grid: np.ndarray,
    pattern: np.ndarray,
    offset: tuple[int, int] | None = None,
) -> np.ndarray:
    """Embed a pattern into a larger grid.

    Args:
        grid: Target grid array to place pattern onto.
        pattern: 2D binary pattern array.
        offset: (row, col) upper-left coordinates. If None, centers the pattern.

    Returns:
        New grid with embedded pattern.
    """
    gh, gw = grid.shape
    ph, pw = pattern.shape

    if ph > gh or pw > gw:
        raise DomainValidationError(
            f"Pattern size ({ph}, {pw}) exceeds grid dimensions ({gh}, {gw})."
        )

    if offset is None:
        r_offset = (gh - ph) // 2
        c_offset = (gw - pw) // 2
    else:
        r_offset, c_offset = offset
        if r_offset < 0 or c_offset < 0 or r_offset + ph > gh or c_offset + pw > gw:
            raise DomainValidationError(
                f"Offset {offset} places pattern ({ph}, {pw}) out of grid bounds ({gh}, {gw})."
            )

    result = grid.copy()
    result[r_offset : r_offset + ph, c_offset : c_offset + pw] = pattern
    return result


def initialize_grid(
    grid_size: tuple[int, int],
    init_mode: InitMode = InitMode.RANDOM,
    seed: int | None = None,
    density: float = 0.2,
    pattern_name: str | None = None,
    pattern_offset: tuple[int, int] | None = None,
    custom_grid: np.ndarray | None = None,
) -> np.ndarray:
    """Generate an initial state grid for cellular automata.

    Args:
        grid_size: (height, width) tuple of dimensions.
        init_mode: Initialization mode (RANDOM, PATTERN, EMPTY, CUSTOM).
        seed: Random seed for deterministic random generation.
        density: Probability of alive cell in RANDOM mode (0.0 to 1.0).
        pattern_name: Name of standard pattern in PATTERNS dictionary.
        pattern_offset: (row, col) position for pattern top-left corner.
        custom_grid: User-supplied numpy array for CUSTOM mode.

    Returns:
        2D numpy array of dtype int8 with binary cell states.
    """
    height, width = grid_size
    if height < 1 or width < 1:
        raise DomainValidationError(f"Invalid grid size ({height}, {width}). Must be positive.")

    if init_mode == InitMode.EMPTY:
        return np.zeros((height, width), dtype=np.int8)

    elif init_mode == InitMode.RANDOM:
        if not (0.0 <= density <= 1.0):
            raise DomainValidationError(f"Density must be between 0.0 and 1.0, got {density}.")
        rng = np.random.default_rng(seed)
        random_vals = rng.random((height, width))
        return (random_vals < density).astype(np.int8)

    elif init_mode == InitMode.PATTERN:
        if not pattern_name:
            raise DomainValidationError("pattern_name must be specified when init_mode is PATTERN.")
        normalized = pattern_name.strip().lower()
        if normalized not in PATTERNS:
            available = ", ".join(sorted(PATTERNS.keys()))
            raise DomainValidationError(
                f"Unknown pattern '{pattern_name}'. Available patterns: {available}"
            )
        base_grid = np.zeros((height, width), dtype=np.int8)
        return embed_pattern(base_grid, PATTERNS[normalized], offset=pattern_offset)

    elif init_mode == InitMode.CUSTOM:
        if custom_grid is None:
            raise DomainValidationError("custom_grid must be provided when init_mode is CUSTOM.")
        if custom_grid.shape != (height, width):
            raise DomainValidationError(
                f"Custom grid shape {custom_grid.shape} does not match "
                f"grid_size ({height}, {width})."
            )
        return cast(np.ndarray, (custom_grid != 0).astype(np.int8))

    else:
        raise DomainValidationError(f"Unsupported init_mode: {init_mode}")
