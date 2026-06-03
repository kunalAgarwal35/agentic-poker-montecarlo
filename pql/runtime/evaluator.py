from __future__ import annotations
import numpy as np

from hand_indexing import BINOMIAL, HOLDEM_7CARD_COMBOS
from optimized_evaluator import (
    get_score_array, get_category_array,
    best_score_numba, best5_of_7_numba,
    best5_category_omaha_numba, best5_category_holdem_numba,
    nut_score_holdem_numba, nut_score_omaha_numba,
)
from pql.scenario import Scenario


_HAND_COMBOS_2 = np.array([[0, 1]], dtype=np.int32)


def compute_nut_scores(scenario, boards) -> np.ndarray:
    """Per-trial nut score: best achievable best-5 over all 2-card holdings given the
    board (live deck = 52 - board - dead). Absolute nuts (does not exclude players' cards)."""
    g = scenario.game
    sa = get_score_array()
    n = boards.shape[0]
    out = np.empty(n, dtype=np.float64)
    boards = boards.astype(np.int32)
    dead = set(scenario.dead.tolist())
    holdem = g.eval_kind == 'holdem'
    for t in range(n):
        b = boards[t]
        used = set(b.tolist()) | dead
        deck = np.array([c for c in range(52) if c not in used], dtype=np.int32)
        if holdem:
            out[t] = nut_score_holdem_numba(b, deck, HOLDEM_7CARD_COMBOS, sa, BINOMIAL)
        else:
            out[t] = nut_score_omaha_numba(b, deck, _HAND_COMBOS_2, g.board_combos, sa, BINOMIAL)
    return out


def player_score_scalar(scenario: Scenario, player_idx: int, board: np.ndarray) -> float:
    """Best-5 score for one player on one completed (5-card) board. Reference impl."""
    g = scenario.game
    hand = scenario.players[player_idx].cards.astype(np.int32)
    board = board.astype(np.int32)
    if g.eval_kind == 'holdem':
        cards7 = np.concatenate((hand, board))
        return best5_of_7_numba(cards7, HOLDEM_7CARD_COMBOS, get_score_array(), BINOMIAL)
    return best_score_numba(
        hand, board, g.hand_combos, g.board_combos, get_score_array(), BINOMIAL,
    )


def player_scores(scenario, player_hands, boards) -> np.ndarray:
    """
    Vectorized best-5 score for every (trial, player). player_hands: (N, P, H) int
    array of per-trial hole cards; boards: (N, 5) int array. Returns (N, P) float.
    Branches once per game on eval_kind.
    """
    g = scenario.game
    sa = get_score_array()
    n = boards.shape[0]
    p = player_hands.shape[1]
    out = np.empty((n, p), dtype=np.float64)
    boards = boards.astype(np.int32)
    holdem = g.eval_kind == 'holdem'
    for t in range(n):
        b = boards[t]
        for j in range(p):
            hand = player_hands[t, j].astype(np.int32)
            if holdem:
                out[t, j] = best5_of_7_numba(np.concatenate((hand, b)), HOLDEM_7CARD_COMBOS, sa, BINOMIAL)
            else:
                out[t, j] = best_score_numba(hand, b, g.hand_combos, g.board_combos, sa, BINOMIAL)
    return out


def player_categories(scenario, player_hands, boards) -> np.ndarray:
    """Per-(trial, player) best-5 category index (0-8). Mirrors player_scores."""
    g = scenario.game
    sa = get_score_array()
    ca = get_category_array()
    n = boards.shape[0]
    p = player_hands.shape[1]
    out = np.empty((n, p), dtype=np.int64)
    boards = boards.astype(np.int32)
    holdem = g.eval_kind == 'holdem'
    for t in range(n):
        b = boards[t]
        for j in range(p):
            hand = player_hands[t, j].astype(np.int32)
            if holdem:
                out[t, j] = best5_category_holdem_numba(np.concatenate((hand, b)), HOLDEM_7CARD_COMBOS, sa, ca, BINOMIAL)
            else:
                out[t, j] = best5_category_omaha_numba(hand, b, g.hand_combos, g.board_combos, sa, ca, BINOMIAL)
    return out
