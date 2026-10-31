"""Exact per-combo flop features for Omaha hands (4 or 5 hole cards).

Definitions (question wording must match):
  * set            -- a hole pair matching a board rank (unpaired boards only)
  * straight       -- villain's best hand is a straight or straight flush
  * flush draw     -- two-tone flop; >= 2 hole cards of the board suit, no made flush
  * nut flush draw -- as above, one of them the highest card of that suit not on the board
  * straight outs  -- unseen turn cards (not in villain's hand or on the flop) that give
                      a straight villain doesn't already have
  * nuts           -- ties the best hand possible on this flop
  * air            -- no pair or better, no flush draw, fewer than 8 straight outs
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from trainer.fasteval import evaluate

RANKS = "23456789TJQKA"
SUITS = "shdc"
HIGH, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)

_STRAIGHTS = [(0b11111 << lo) for lo in range(9)] + [(1 << 12) | 0b1111]


def card(s: str) -> int:
    return RANKS.index(s[0].upper()) * 4 + SUITS.index(s[1].lower())


def card_str(c: int) -> str:
    return RANKS[c // 4] + SUITS[c % 4]


def cards(s: str) -> list[int]:
    return [card(s[i:i + 2]) for i in range(0, len(s), 2)]


def texture(flop: list[int]) -> dict:
    fr = sorted((c // 4 for c in flop), reverse=True)
    ns = len({c % 4 for c in flop})
    paired = len(set(fr)) < 3
    bits = 0
    for r in fr:
        bits |= 1 << r
    return {
        "suits": {1: "monotone", 2: "two-tone", 3: "rainbow"}[ns],
        "paired": paired,
        "straight_possible": not paired and any((m & bits) == bits for m in _STRAIGHTS),
    }


def _makes_straight(hole_ranks: tuple, board_ranks: list[int]) -> bool:
    """Omaha: some 2 distinct hole ranks + 3 distinct board ranks form a straight."""
    for a, b in combinations(set(hole_ranks), 2):
        for tri in combinations(board_ranks, 3):
            if len(set(tri)) < 3 or a in tri or b in tri:
                continue
            m = (1 << a) | (1 << b) | (1 << tri[0]) | (1 << tri[1]) | (1 << tri[2])
            if m in _STRAIGHTS:
                return True
    return False


def straight_info(hands: np.ndarray, flop: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """(has a straight on the flop, straight outs) per combo. Rank-only, so
    computed once per distinct hole-rank multiset."""
    fr = [c // 4 for c in flop]
    hr = np.sort(hands // 4, axis=1)
    uniq, inv = np.unique(hr, axis=0, return_inverse=True)
    made_u = np.zeros(len(uniq), dtype=bool)
    outs_u = np.zeros(len(uniq), dtype=np.int64)
    for k, row in enumerate(uniq):
        row = tuple(int(x) for x in row)
        made = _makes_straight(row, fr)
        made_u[k] = made
        if made:
            continue
        o = 0
        for t in range(13):
            if _makes_straight(row, fr + [t]):
                o += max(0, 4 - row.count(t) - fr.count(t))
        outs_u[k] = o
    inv = inv.reshape(-1)
    return made_u[inv], outs_u[inv]


def nut_score(flop: list[int]) -> float:
    deck = np.array([c for c in range(52) if c not in flop], dtype=np.int64)
    pairs = np.array(list(combinations(deck, 2)), dtype=np.int64)
    s, _ = evaluate(pairs, np.array(flop))
    return float(s.max())


def flop_features(hands: np.ndarray, flop: list[int], nuts: float | None = None) -> dict[str, np.ndarray]:
    """Feature arrays for every combo (hands must not conflict with the flop)."""
    hr, hs = hands // 4, hands % 4
    fr, fs = [c // 4 for c in flop], [c % 4 for c in flop]
    best, cat = evaluate(hands, np.array(flop))

    has_set = np.zeros(len(hands), dtype=bool)
    top_set = np.zeros(len(hands), dtype=bool)
    if len(set(fr)) == 3:
        for r in fr:
            n = (hr == r).sum(axis=1)
            has_set |= n >= 2
            if r == max(fr):
                top_set |= n >= 2

    fd = np.zeros(len(hands), dtype=bool)
    nfd = np.zeros(len(hands), dtype=bool)
    for s in set(fs):
        if fs.count(s) != 2:
            continue
        held = hs == s
        fd |= held.sum(axis=1) >= 2
        on_board = {fr[k] for k in range(3) if fs[k] == s}
        top = max(r for r in range(13) if r not in on_board)
        nfd |= (held & (hr == top)).any(axis=1) & (held.sum(axis=1) >= 2)

    has_straight, outs = straight_info(hands, flop)
    if nuts is None:
        nuts = nut_score(flop)
    no_flush = cat < FLUSH
    return {
        "category": cat,
        "set": has_set,
        "top_set": top_set,
        "straight": (cat == STRAIGHT) | (cat == STRAIGHT_FLUSH),
        "two_pair_plus": cat >= TWO_PAIR,
        "flush": cat >= FLUSH,
        "flush_draw": fd & no_flush,
        "nut_flush_draw": nfd & no_flush,
        "straight_outs": outs,
        "nuts": best == nuts,
        "air": (cat == HIGH) & ~fd & (outs < 8),
        "combo_draw": fd & (outs >= 8) & (cat < STRAIGHT),
    }
