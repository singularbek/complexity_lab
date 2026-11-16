"""Transition rules for cellular automata, including Conway's Game of Life."""

import re
from dataclasses import dataclass

import numpy as np

from complexity_lab.domain.exceptions import DomainValidationError


@dataclass(frozen=True)
class LifeRule:
    """Transition rule for Life-like totalistic cellular automata.

    Attributes:
        birth: Neighbor counts causing a dead cell to become alive.
        survival: Neighbor counts allowing a live cell to stay alive.
        rule_string: Canonical RLE/Golly format string, e.g. 'B3/S23'.
    """

    birth: frozenset[int]
    survival: frozenset[int]
    rule_string: str

    @classmethod
    def from_string(cls, rule_str: str) -> "LifeRule":
        """Parse rule string in standard 'B.../S...' or 'S.../B...' format.

        Examples:
            - 'B3/S23' (Conway's Game of Life)
            - 'B36/S23' (HighLife)
            - 'B1/S012345678' (Life without Death)
            - '23/3' (S/B Wolfram format)
        """
        cleaned = rule_str.strip().upper()

        # Check standard B.../S... format
        bs_match = re.fullmatch(r"B([0-8]*)/S([0-8]*)", cleaned)
        if bs_match:
            b_digits, s_digits = bs_match.groups()
            birth = frozenset(int(d) for d in b_digits)
            survival = frozenset(int(d) for d in s_digits)
            canonical = f"B{''.join(sorted(b_digits))}/S{''.join(sorted(s_digits))}"
            return cls(birth=birth, survival=survival, rule_string=canonical)

        # Check S.../B... format
        sb_match = re.fullmatch(r"S([0-8]*)/B([0-8]*)", cleaned)
        if sb_match:
            s_digits, b_digits = sb_match.groups()
            birth = frozenset(int(d) for d in b_digits)
            survival = frozenset(int(d) for d in s_digits)
            canonical = f"B{''.join(sorted(b_digits))}/S{''.join(sorted(s_digits))}"
            return cls(birth=birth, survival=survival, rule_string=canonical)

        # Check numeric S/B format like 23/3
        slash_match = re.fullmatch(r"([0-8]*)/([0-8]*)", cleaned)
        if slash_match:
            s_digits, b_digits = slash_match.groups()
            birth = frozenset(int(d) for d in b_digits)
            survival = frozenset(int(d) for d in s_digits)
            canonical = f"B{''.join(sorted(b_digits))}/S{''.join(sorted(s_digits))}"
            return cls(birth=birth, survival=survival, rule_string=canonical)

        raise DomainValidationError(
            f"Invalid cellular automaton rule string '{rule_str}'. Expected format like 'B3/S23'."
        )

    def apply(self, grid: np.ndarray, neighbor_counts: np.ndarray) -> np.ndarray:
        """Apply rule to compute next grid state given neighbor counts.

        Args:
            grid: Current 2D binary grid array.
            neighbor_counts: 2D array of neighbor counts for each cell.

        Returns:
            New 2D numpy array with updated binary cell states.
        """
        is_alive = grid == 1
        is_dead = ~is_alive

        birth_mask = np.isin(neighbor_counts, tuple(self.birth))
        survival_mask = np.isin(neighbor_counts, tuple(self.survival))

        next_grid = np.where((is_dead & birth_mask) | (is_alive & survival_mask), 1, 0)
        return next_grid.astype(np.int8)


CONWAY_LIFE = LifeRule.from_string("B3/S23")
