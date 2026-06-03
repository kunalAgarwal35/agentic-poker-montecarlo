from __future__ import annotations
import numpy as np
from pql.scenario import Scenario


def live_deck(scenario: Scenario) -> np.ndarray:
    """All 52 card ints minus board, dead, and every player's hole cards."""
    used = set(scenario.board.tolist())
    used.update(scenario.dead.tolist())
    for p in scenario.players:
        used.update(p.cards.tolist())
    return np.array([c for c in range(52) if c not in used], dtype=np.int32)
