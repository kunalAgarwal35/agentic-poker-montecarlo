from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from hand_indexing import (
    PLO4_HAND_COMBOS, PLO5_HAND_COMBOS, PLO6_HAND_COMBOS, PLO4_BOARD_COMBOS,
)

# Board combos are always C(5,3) regardless of hole-card count.
_BOARD_COMBOS = PLO4_BOARD_COMBOS  # shape (10, 3)


@dataclass(frozen=True)
class GameSpec:
    name: str
    num_hole: int
    hand_combos: np.ndarray   # (C(num_hole,2), 2)  -- unused for holdem
    board_combos: np.ndarray  # (10, 3)
    eval_kind: str = 'omaha'  # 'omaha' (exactly 2 hole + 3 board) | 'holdem' (best 5 of 7)


_HOLDEM_HAND_COMBOS = np.empty((0, 2), dtype=np.int32)  # unused; holdem uses best5_of_7

_GAMES = {
    "holdem": GameSpec("holdem", 2, _HOLDEM_HAND_COMBOS, _BOARD_COMBOS, eval_kind="holdem"),
    "omahahi": GameSpec("omahahi", 4, PLO4_HAND_COMBOS, _BOARD_COMBOS),
    "omahahi5": GameSpec("omahahi5", 5, PLO5_HAND_COMBOS, _BOARD_COMBOS),
    "omahahi6": GameSpec("omahahi6", 6, PLO6_HAND_COMBOS, _BOARD_COMBOS),
}


def get_game(name: str) -> GameSpec:
    key = name.lower()
    if key not in _GAMES:
        raise ValueError(
            f"Unsupported game '{name}'. Phase 1 supports: {sorted(_GAMES)}"
        )
    return _GAMES[key]
