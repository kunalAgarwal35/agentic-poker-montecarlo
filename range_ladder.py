"""Postflop range ladder: hero equity vs board-strength percentile slices.

See docs/superpowers/plans/2026-08-13-postflop-range-ladder.md
"""
from itertools import combinations

import numpy as np

from card_encoding import generate_deck_ints, hand_str_to_ints, ints_to_hand_str
from hand_indexing import BINOMIAL
from hand_rank_evaluator import (
    _BOARD_COMBOS,
    _HAND_COMBOS,
    _batch_best_score,
    detect_game_type,
    get_score_array,
)


def sample_villains(deck, num_cards, n, rng):
    """Distinct legal villain hands drawn from `deck`, as sorted rows.

    Falls back to exhaustive enumeration when the space is smaller than `n`,
    so tiny decks return every hand exactly once instead of looping forever.
    """
    deck = np.asarray(deck, dtype=np.int32)
    total = len(deck)
    if total < num_cards:
        return np.empty((0, num_cards), dtype=np.int32)

    # Exhaustive when the space is small enough to enumerate cheaply.
    space = int(BINOMIAL[total, num_cards]) if total <= 52 and num_cards <= 6 else n + 1
    if space <= max(n, 1):
        rows = [sorted(c) for c in combinations(deck.tolist(), num_cards)]
        return np.array(rows, dtype=np.int32)

    seen = set()
    rows = []
    while len(rows) < n:
        need = n - len(rows)
        draw = rng.random((need * 2, total)).argsort(axis=1)[:, :num_cards]
        cand = np.sort(deck[draw], axis=1)
        for row in cand:
            key = tuple(row.tolist())
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            if len(rows) == n:
                break
    return np.array(rows, dtype=np.int32)


def sample_runouts(deck, board_len, r, rng):
    """`r` random completions of the board. River -> a single empty runout."""
    deck = np.asarray(deck, dtype=np.int32)
    need = 5 - board_len
    if need <= 0:
        return np.empty((1, 0), dtype=np.int32)
    draw = rng.random((r, len(deck))).argsort(axis=1)[:, :need]
    return deck[draw].astype(np.int32)


def eligible_mask(villains, runout):
    """False for villains holding a card that the runout also uses.

    Those pairings are impossible and must never be scored. Skipping them
    evaluates each villain over exactly the runouts compatible with it, which
    is the correct conditional distribution -- unbiased. See spec 4.3.
    """
    if runout.size == 0:
        return np.ones(villains.shape[0], dtype=bool)
    return ~np.isin(villains, runout).any(axis=1)
