"""Numba Omaha evaluation on the engine's precomputed 5-card tables.

Card ints are the engine's (card_encoding.CARD_TO_INT): rank*4 + suit with
rank 0..12 = 2..A and suit 0..3 = s,h,d,c -- identical to trainer's own
encoding, so arrays pass straight through.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from numba import njit

from hand_indexing import BINOMIAL
from optimized_evaluator import get_category_array, get_score_array

SCORE = get_score_array()
CATEGORY = get_category_array()


def hole_combos(n_hole: int) -> np.ndarray:
    return np.array(list(combinations(range(n_hole), 2)), dtype=np.int32)


def board_combos(n_board: int) -> np.ndarray:
    return np.array(list(combinations(range(n_board), 3)), dtype=np.int32)


@njit(cache=True)
def _idx(a, b, c, d, e, binomial):
    # insertion sort 5 values ascending
    v = np.empty(5, dtype=np.int64)
    v[0], v[1], v[2], v[3], v[4] = a, b, c, d, e
    for i in range(1, 5):
        x = v[i]
        j = i - 1
        while j >= 0 and v[j] > x:
            v[j + 1] = v[j]
            j -= 1
        v[j + 1] = x
    return binomial[v[0], 1] + binomial[v[1], 2] + binomial[v[2], 3] + binomial[v[3], 4] + binomial[v[4], 5]


@njit(cache=True)
def best_score_cat(hands, boards, hc, bc, score, category, binomial):
    """hands (N,H), boards (N,B) -> (best score (N,), category of best (N,))."""
    n = hands.shape[0]
    out_s = np.empty(n, dtype=np.float64)
    out_c = np.empty(n, dtype=np.int64)
    for k in range(n):
        best = -1.0
        cat = 0
        for p in range(hc.shape[0]):
            h0 = hands[k, hc[p, 0]]
            h1 = hands[k, hc[p, 1]]
            for q in range(bc.shape[0]):
                i = _idx(h0, h1, boards[k, bc[q, 0]], boards[k, bc[q, 1]], boards[k, bc[q, 2]], binomial)
                s = score[i]
                if s > best:
                    best = s
                    cat = category[i]
        out_s[k] = best
        out_c[k] = cat
    return out_s, out_c


def evaluate(hands: np.ndarray, boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hands = np.ascontiguousarray(hands, dtype=np.int64)
    boards = np.ascontiguousarray(boards, dtype=np.int64)
    if boards.ndim == 1:
        boards = np.ascontiguousarray(np.broadcast_to(boards, (len(hands), len(boards))))
    return best_score_cat(hands, boards, hole_combos(hands.shape[1]), board_combos(boards.shape[1]),
                          SCORE, CATEGORY, BINOMIAL)
