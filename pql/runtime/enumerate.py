from __future__ import annotations
import numpy as np
from itertools import combinations
from pql.scenario import Scenario
from pql.runtime._deck import live_deck

# Max number of board completions to enumerate exhaustively (else use Monte Carlo).
ENUM_THRESHOLD = 200_000


def enumerate_boards(scenario: Scenario) -> np.ndarray | None:
    """
    Return an (M, 5) array of every legal board completion if M <= ENUM_THRESHOLD,
    else None (caller falls back to Monte Carlo).
    """
    board = scenario.board
    need = 5 - len(board)
    if need < 0:
        raise ValueError("Board already has more than 5 cards")
    if need == 0:
        return np.tile(board, (1, 1)).astype(np.int32)

    deck = live_deck(scenario)
    from math import comb
    m = comb(len(deck), need)
    if m > ENUM_THRESHOLD:
        return None

    combos = np.array(list(combinations(deck.tolist(), need)), dtype=np.int32)  # (M, need)
    fixed = np.tile(board, (combos.shape[0], 1)).astype(np.int32) if len(board) else \
        np.empty((combos.shape[0], 0), dtype=np.int32)
    return np.concatenate([fixed, combos], axis=1).astype(np.int32)
