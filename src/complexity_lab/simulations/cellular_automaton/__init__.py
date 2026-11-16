"""Cellular Automaton module for deterministic complex systems simulations."""

from complexity_lab.simulations.cellular_automaton.boundary import (
    BoundaryCondition,
    apply_boundary,
)
from complexity_lab.simulations.cellular_automaton.initialization import (
    PATTERNS,
    InitMode,
    embed_pattern,
    initialize_grid,
)
from complexity_lab.simulations.cellular_automaton.neighborhood import (
    NeighborhoodType,
    count_neighbors,
)
from complexity_lab.simulations.cellular_automaton.rules import CONWAY_LIFE, LifeRule
from complexity_lab.simulations.cellular_automaton.simulation import (
    CellularAutomatonParameters,
    CellularAutomatonSimulation,
)

__all__ = [
    "CONWAY_LIFE",
    "PATTERNS",
    "BoundaryCondition",
    "CellularAutomatonParameters",
    "CellularAutomatonSimulation",
    "InitMode",
    "LifeRule",
    "NeighborhoodType",
    "apply_boundary",
    "count_neighbors",
    "embed_pattern",
    "initialize_grid",
]
