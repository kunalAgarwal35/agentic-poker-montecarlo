"""Per-trial single-card outs kernel shared by the draw/outs value functions.

For a player standing at `street` (board[:street] visible), counts the cards in the
live deck that bring the player's best-5 to a target category. Two match modes:
  GE     -> result category >= target            (PPT outsToHandType / minOutsToHandType)
  MEMBER -> result category in {cat_a, cat_b}     (named draw helpers; keeps a flush-making
                                                   card from being miscounted as a straight out)
The live deck excludes only the player's hole cards, board[:street], and dead cards, so
board[street:] (the sampled turn/river) and opponents' cards remain countable outs --
this reproduces PPT's averaged out counts over Monte Carlo runouts.

Note on future streets: when the requested street equals the given board length, the out
count is fully determined by the visible board (deterministic, and exactly matches the
PPT oracle). When the requested street is BEYOND the given board (e.g. street=turn with
only a flop), the value is an expectation over the unseen card(s); under exact enumeration
the turn/river labeling is not uniform, so such future-street outs are best queried in
Monte Carlo mode.
"""
from __future__ import annotations
import itertools
import numpy as np
from numba import njit

from hand_indexing import BINOMIAL
from optimized_evaluator import (
    get_score_array, get_category_array, sort5, hand_to_index_numba,
)

MODE_GE = 0
MODE_MEMBER = 1

_STREET = {"flop": 3, "turn": 4, "river": 5}


def street_index(token: str, allow_river: bool = False) -> int:
    t = token.lower()
    if t not in _STREET:
        raise ValueError(f"Unknown street '{token}'. Expected flop/turn" + ("/river" if allow_river else ""))
    s = _STREET[t]
    if s == 5 and not allow_river:
        raise ValueError("outs are only defined for 'flop' or 'turn' (the river has no next card)")
    return s


def _combos(n: int, k: int) -> np.ndarray:
    if n < k:
        return np.zeros((0, k), dtype=np.int32)
    return np.array(list(itertools.combinations(range(n), k)), dtype=np.int32)


# best-5-of-N combos for holdem (N = 2 hole + board-so-far)
_HOLDEM_COMBOS = {5: _combos(5, 5), 6: _combos(6, 5), 7: _combos(7, 5)}
# 3-of-N board combos for omaha (N = board-so-far)
_OMAHA_BCOMBOS = {3: _combos(3, 3), 4: _combos(4, 3), 5: _combos(5, 3)}


@njit(cache=True)
def _cat_combos(cards, combos, sa, ca, binom):
    """Category index (0-8) of the best 5-card hand over the given index combos."""
    best = 0.0
    best_idx = 0
    for i in range(combos.shape[0]):
        a = cards[combos[i, 0]]; b = cards[combos[i, 1]]; c = cards[combos[i, 2]]
        d = cards[combos[i, 3]]; e = cards[combos[i, 4]]
        c0, c1, c2, c3, c4 = sort5(a, b, c, d, e)
        idx = hand_to_index_numba(c0, c1, c2, c3, c4, binom)
        s = sa[idx]
        if s > best:
            best = s
            best_idx = idx
    return ca[best_idx]


@njit(cache=True)
def _cat_omaha(hole, board, hand_combos, board_combos, sa, ca, binom):
    """Category index of the best omaha hand (exactly 2 hole + 3 board)."""
    best = 0.0
    best_idx = 0
    for i in range(hand_combos.shape[0]):
        h0 = hole[hand_combos[i, 0]]; h1 = hole[hand_combos[i, 1]]
        for j in range(board_combos.shape[0]):
            b0 = board[board_combos[j, 0]]; b1 = board[board_combos[j, 1]]; b2 = board[board_combos[j, 2]]
            c0, c1, c2, c3, c4 = sort5(h0, h1, b0, b1, b2)
            idx = hand_to_index_numba(c0, c1, c2, c3, c4, binom)
            s = sa[idx]
            if s > best:
                best = s
                best_idx = idx
    return ca[best_idx]


@njit(cache=True)
def _match(cat, mode, a, b):
    if mode == 0:        # GE
        return cat >= a
    return cat == a or cat == b   # MEMBER


@njit(cache=True)
def _outs_trial_holdem(hole, partial, live, mode, a, b, combos_made, combos_out, sa, ca, binom):
    nmade = hole.shape[0] + partial.shape[0]
    made = np.empty(nmade, dtype=np.int32)
    for i in range(hole.shape[0]):
        made[i] = hole[i]
    for i in range(partial.shape[0]):
        made[hole.shape[0] + i] = partial[i]
    if _cat_combos(made, combos_made, sa, ca, binom) >= a:   # already at/above target -> no outs
        return 0
    cnt = 0
    buf = np.empty(nmade + 1, dtype=np.int32)
    for i in range(nmade):
        buf[i] = made[i]
    for k in range(live.shape[0]):
        buf[nmade] = live[k]
        if _match(_cat_combos(buf, combos_out, sa, ca, binom), mode, a, b):
            cnt += 1
    return cnt


@njit(cache=True)
def _outs_trial_omaha(hole, partial, live, mode, a, b, hand_combos, bcombos_made, bcombos_out, sa, ca, binom):
    if _cat_omaha(hole, partial, hand_combos, bcombos_made, sa, ca, binom) >= a:
        return 0
    cnt = 0
    nb = partial.shape[0]
    buf = np.empty(nb + 1, dtype=np.int32)
    for i in range(nb):
        buf[i] = partial[i]
    for k in range(live.shape[0]):
        buf[nb] = live[k]
        if _match(_cat_omaha(hole, buf, hand_combos, bcombos_out, sa, ca, binom), mode, a, b):
            cnt += 1
    return cnt


def outs_count(ctx, player_idx, street, mode, a, b) -> np.ndarray:
    """Per-trial out counts (N,). `a`/`b` are category indices; see module docstring for modes."""
    scenario = ctx.scenario
    g = scenario.game
    holdem = g.eval_kind == "holdem"
    sa = get_score_array()
    ca = get_category_array()
    boards = ctx.boards.astype(np.int32)
    hands = ctx.player_hands.astype(np.int32)   # (N, P, H)
    n = boards.shape[0]
    dead = set(scenario.dead.tolist())
    out = np.empty(n, dtype=np.float64)
    if holdem:
        combos_made = _HOLDEM_COMBOS[2 + street]
        combos_out = _HOLDEM_COMBOS[2 + street + 1]
    else:
        bcombos_made = _OMAHA_BCOMBOS[street]
        bcombos_out = _OMAHA_BCOMBOS[street + 1]
        hand_combos = g.hand_combos
    for t in range(n):
        hole = hands[t, player_idx]
        partial = boards[t, :street]
        used = set(hole.tolist()) | set(partial.tolist()) | dead
        live = np.array([c for c in range(52) if c not in used], dtype=np.int32)
        if holdem:
            out[t] = _outs_trial_holdem(hole, partial, live, mode, a, b, combos_made, combos_out, sa, ca, BINOMIAL)
        else:
            out[t] = _outs_trial_omaha(hole, partial, live, mode, a, b, hand_combos, bcombos_made, bcombos_out, sa, ca, BINOMIAL)
    return out


def infer_street(ctx, args) -> int:
    """Street for a named draw helper: an explicit token if given, else the live board length."""
    if args:
        return street_index(args[0])
    s = len(ctx.scenario.board)
    if s not in (3, 4):
        raise ValueError("draw helpers need a flop (3) or turn (4) board, or an explicit street token")
    return s


@njit(cache=True)
def _score_cat_holdem(hole, partial, combos, sa, ca, binom):
    best = 0.0
    best_idx = 0
    n = hole.shape[0] + partial.shape[0]
    cards = np.empty(n, dtype=np.int32)
    for i in range(hole.shape[0]):
        cards[i] = hole[i]
    for i in range(partial.shape[0]):
        cards[hole.shape[0] + i] = partial[i]
    for i in range(combos.shape[0]):
        a = cards[combos[i, 0]]; b = cards[combos[i, 1]]; c = cards[combos[i, 2]]
        d = cards[combos[i, 3]]; e = cards[combos[i, 4]]
        c0, c1, c2, c3, c4 = sort5(a, b, c, d, e)
        idx = hand_to_index_numba(c0, c1, c2, c3, c4, binom)
        s = sa[idx]
        if s > best:
            best = s
            best_idx = idx
    return best, ca[best_idx]


@njit(cache=True)
def _nut_cat_score_holdem(partial, deck, combos, target, sa, ca, binom):
    """Max best-5 score over all 2-card holdings whose best-5 category == target. 0.0 if none."""
    best = 0.0
    nd = deck.shape[0]
    hole = np.empty(2, dtype=np.int32)
    for i in range(nd):
        for j in range(i + 1, nd):
            hole[0] = deck[i]; hole[1] = deck[j]
            s, cat = _score_cat_holdem(hole, partial, combos, sa, ca, binom)
            if cat == target and s > best:
                best = s
    return best


@njit(cache=True)
def _score_cat_omaha(hole, partial, hand_combos, board_combos, sa, ca, binom):
    best = 0.0
    best_idx = 0
    for i in range(hand_combos.shape[0]):
        h0 = hole[hand_combos[i, 0]]; h1 = hole[hand_combos[i, 1]]
        for j in range(board_combos.shape[0]):
            b0 = partial[board_combos[j, 0]]; b1 = partial[board_combos[j, 1]]; b2 = partial[board_combos[j, 2]]
            c0, c1, c2, c3, c4 = sort5(h0, h1, b0, b1, b2)
            idx = hand_to_index_numba(c0, c1, c2, c3, c4, binom)
            s = sa[idx]
            if s > best:
                best = s
                best_idx = idx
    return best, ca[best_idx]


@njit(cache=True)
def _nut_cat_score_omaha(partial, deck, hand_combos2, board_combos, target, sa, ca, binom):
    best = 0.0
    nd = deck.shape[0]
    hole = np.empty(2, dtype=np.int32)
    for i in range(nd):
        for j in range(i + 1, nd):
            hole[0] = deck[i]; hole[1] = deck[j]
            s, cat = _score_cat_omaha(hole, partial, hand_combos2, board_combos, sa, ca, binom)
            if cat == target and s > best:
                best = s
    return best


_HAND_COMBOS_2 = np.array([[0, 1]], dtype=np.int32)


def nut_hi_for_own_category(ctx, player_idx, street) -> np.ndarray:
    """1.0 iff, on board[:street], the player holds the nut hand OF THEIR OWN made category
    (e.g. the nut flush when they have a flush). Matches PPT's nutHiForHandType(player, street).
    Uses the 2-card-holding nut convention, like nutHi/compute_nut_scores."""
    scenario = ctx.scenario
    g = scenario.game
    holdem = g.eval_kind == "holdem"
    sa = get_score_array()
    ca = get_category_array()
    boards = ctx.boards.astype(np.int32)
    hands = ctx.player_hands.astype(np.int32)
    n = boards.shape[0]
    dead = set(scenario.dead.tolist())
    out = np.zeros(n, dtype=np.float64)
    if holdem:
        combos = _HOLDEM_COMBOS[2 + street]
    else:
        bcombos = _OMAHA_BCOMBOS[street]
    for t in range(n):
        hole = hands[t, player_idx]
        partial = boards[t, :street]
        if holdem:
            pscore, pcat = _score_cat_holdem(hole, partial, combos, sa, ca, BINOMIAL)
        else:
            pscore, pcat = _score_cat_omaha(hole, partial, g.hand_combos, bcombos, sa, ca, BINOMIAL)
        # Nut of the player's own category: best score over ALL 2-card holdings on
        # board[:street] whose best-5 category == pcat (deck excludes only board + dead,
        # so the player's own holding is itself a candidate -- matching compute_nut_scores).
        used = set(partial.tolist()) | dead
        deck = np.array([c for c in range(52) if c not in used], dtype=np.int32)
        if holdem:
            nut = _nut_cat_score_holdem(partial, deck, combos, pcat, sa, ca, BINOMIAL)
        else:
            nut = _nut_cat_score_omaha(partial, deck, _HAND_COMBOS_2, bcombos, pcat, sa, ca, BINOMIAL)
        if nut > 0.0 and pscore == nut:
            out[t] = 1.0
    return out


def category_at_street(ctx, player_idx, street) -> np.ndarray:
    """Best-5 category index (0-8) per trial for the player, evaluated on board[:street].
    street == 5 returns the cached full-board categories (fast, unchanged); 3/4 compute
    the category on the partial board, reusing the per-street category kernels."""
    if street == 5:
        return ctx.categories()[:, player_idx]
    scenario = ctx.scenario
    g = scenario.game
    holdem = g.eval_kind == "holdem"
    sa = get_score_array()
    ca = get_category_array()
    boards = ctx.boards.astype(np.int32)
    hands = ctx.player_hands.astype(np.int32)
    n = boards.shape[0]
    out = np.empty(n, dtype=np.int64)
    if holdem:
        combos = _HOLDEM_COMBOS[2 + street]
    else:
        bcombos = _OMAHA_BCOMBOS[street]
    for t in range(n):
        hole = hands[t, player_idx]
        partial = boards[t, :street]
        if holdem:
            cards = np.concatenate((hole, partial)).astype(np.int32)
            out[t] = _cat_combos(cards, combos, sa, ca, BINOMIAL)
        else:
            out[t] = _cat_omaha(hole, partial, g.hand_combos, bcombos, sa, ca, BINOMIAL)
    return out
