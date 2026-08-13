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


def score_hands(hands, board5, game, score_array, chunk=2000):
    """Best 5-card score for each hand on one complete board. Higher = better."""
    hand_combos = _HAND_COMBOS[game]
    n = hands.shape[0]
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk):
        block = hands[start:start + chunk]
        boards = np.broadcast_to(board5, (block.shape[0], 5))
        out[start:start + chunk] = _batch_best_score(
            block, np.ascontiguousarray(boards),
            hand_combos, _BOARD_COMBOS, score_array, BINOMIAL,
        )
    return out


def evaluate_population(villains, heroes, board_ints, runouts, game, score_array):
    """The single O((N + H) x R) pass. Returns (strength, hero_equity)."""
    n = villains.shape[0]
    h = heroes.shape[0]

    beat_sum = np.zeros(n, dtype=np.float64)     # share of field beaten, summed
    beat_cnt = np.zeros(n, dtype=np.float64)     # runouts this villain was eligible for
    hero_sum = np.zeros((h, n), dtype=np.float64)

    for runout in runouts:
        board5 = np.concatenate([board_ints, runout]).astype(np.int32)
        mask = eligible_mask(villains, runout)
        idx = np.flatnonzero(mask)
        if idx.size < 2:
            continue

        vs = score_hands(villains[idx], board5, game, score_array)
        hs = score_hands(heroes, board5, game, score_array)

        # Share of the eligible field each villain beats: win + 1/2 tie.
        order = np.sort(vs)
        lower = np.searchsorted(order, vs, side="left")          # strictly worse
        upper = np.searchsorted(order, vs, side="right")
        ties = upper - lower - 1                                  # excluding self
        share = (lower + 0.5 * ties) / (idx.size - 1)
        beat_sum[idx] += share
        beat_cnt[idx] += 1.0

        # Hero vs each eligible villain, heads-up.
        for j in range(h):
            hero_sum[j, idx] += np.where(hs[j] > vs, 1.0, np.where(hs[j] == vs, 0.5, 0.0))

    seen = beat_cnt > 0
    strength = np.zeros(n, dtype=np.float64)
    strength[seen] = beat_sum[seen] / beat_cnt[seen]
    hero_equity = np.zeros((h, n), dtype=np.float64)
    hero_equity[:, seen] = hero_sum[:, seen] / beat_cnt[seen]
    return strength, hero_equity
