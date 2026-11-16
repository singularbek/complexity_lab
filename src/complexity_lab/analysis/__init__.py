"""Scientific analysis engine and observable metrics."""

from complexity_lab.analysis.metrics import (
    compute_block_entropy,
    compute_damage_spreading,
    compute_density,
    compute_hamming_distance,
    compute_shannon_entropy,
    detect_periodicity,
    summarize_trajectory,
)

__all__ = [
    "compute_block_entropy",
    "compute_damage_spreading",
    "compute_density",
    "compute_hamming_distance",
    "compute_shannon_entropy",
    "detect_periodicity",
    "summarize_trajectory",
]

