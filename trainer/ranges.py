"""Villain opening ranges for the trainer.

PLO4: the GTO raise-first-in ranges (PLO50, 100BB, 6-max) from the trainer's
solver data, with each hand's raise frequency as its weight.

PLO5: we have no PLO5 solves, so the ranges are ASSUMED from PLO4:
  * each position opens the same share of hands as in PLO4 GTO
    (UTG 16.8%, MP 23.5%, CO 32.9%, BTN 53.5%, SB 37.4%), and
  * PLO5 hands are ranked by the average PLO4 GTO strength of their five
    4-card sub-hands, where a sub-hand's strength is the percentile rank of
    its BTN raise EV in the solve (sub-hands the BTN folds count as 0).
  * hands with trips (quads) are discounted, matching how the PLO4 solve
    opens trips at about half the random-deal rate.
So a PLO5 hand opens when its 4-card pieces are hands the solver likes --
suits, connectedness and nut potential carry over, and ranges nest
(UTG within MP within CO within BTN). Every PLO5 hand in range has weight 1.
"""
from __future__ import annotations

from functools import lru_cache
from math import comb
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data" / "gto_rfi_plo50_100bb.npz"

# Solver position code -> label shown to players.
POSITIONS = {"EP": "UTG", "MP": "MP", "CO": "CO", "BU": "BTN", "SB": "SB"}
SPOT = {
    "PLO4": "PLO50 100BB 6-max GTO, raise first in",
    "PLO5": "Assumed from PLO4 GTO, raise first in",
}


def colex_index(sorted_cards: np.ndarray) -> np.ndarray:
    """Combinatorial-number-system index of each ascending row (N,k)."""
    idx = np.zeros(len(sorted_cards), dtype=np.int64)
    for j in range(sorted_cards.shape[1]):
        c = sorted_cards[:, j].astype(np.int64)
        # C(c, j+1) vectorized via a lookup table
        table = np.array([comb(x, j + 1) for x in range(52)], dtype=np.int64)
        idx += table[c]
    return idx


@lru_cache(maxsize=1)
def plo4() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    z = np.load(DATA)
    out = {}
    for p in POSITIONS:
        cards = np.sort(z[f"{p}_cards"].astype(np.int64), axis=1)
        out[p] = (cards, z[f"{p}_weights"].astype(np.float64))
    return out


@lru_cache(maxsize=1)
def _plo4_strength() -> np.ndarray:
    """Strength per PLO4 hand (colex index): BTN raise-EV percentile, 0 if folded."""
    z = np.load(DATA)
    cards = np.sort(z["BU_cards"].astype(np.int64), axis=1)
    ev = z["BU_ev"].astype(np.float64)
    # Percentile rank (0..1] of each raised hand's EV; folded hands = 0. Ranks,
    # not raw EV: EV is extremely skewed (AAxx is worth ~10x a typical open),
    # and averaging raw EVs let one AA sub-hand carry hands like AAATT that
    # are poor in PLO5.
    pct = np.empty(len(ev))
    pct[np.argsort(ev, kind="stable")] = np.arange(1, len(ev) + 1) / len(ev)
    strength = np.zeros(comb(52, 4))
    strength[colex_index(cards)] = pct
    return strength


@lru_cache(maxsize=1)
def plo5() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from itertools import combinations
    all5 = np.array(list(combinations(range(52), 5)), dtype=np.int64)  # ascending rows
    strength4 = _plo4_strength()
    score = np.zeros(len(all5))
    for drop in range(5):
        sub = np.delete(all5, drop, axis=1)
        score += strength4[colex_index(sub)]
    score /= 5
    # Third/fourth cards of a rank are dead weight in Omaha. The PLO4 solve opens
    # trips at about half the random-deal rate (0.4% vs 0.9% of UTG opens);
    # without a penalty the sub-hand average keeps PLO5 trips at the random
    # rate, so apply the same lesson explicitly.
    rank_counts = np.stack([(all5 // 4 == r).sum(axis=1) for r in range(13)], axis=1).max(axis=1)
    score *= np.where(rank_counts >= 4, 0.6, np.where(rank_counts == 3, 0.85, 1.0))
    order = np.argsort(-score, kind="stable")
    total = comb(52, 4)
    out = {}
    for p, (_, w4) in plo4().items():
        share = w4.sum() / total
        n = int(round(share * len(all5)))
        pick = order[:n]
        out[p] = (all5[pick], np.ones(n))
    return out


def ranges(game: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return plo4() if game == "PLO4" else plo5()


def subsample(cards: np.ndarray, weights: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform subsample (weights kept) for very large ranges; exact if smaller."""
    if len(cards) <= n:
        return cards, weights
    idx = np.random.default_rng(seed).choice(len(cards), size=n, replace=False)
    return cards[idx], weights[idx]
