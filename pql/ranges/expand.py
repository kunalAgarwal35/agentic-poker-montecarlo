from __future__ import annotations
from collections import Counter
from itertools import combinations
from itertools import product as _product
import numpy as np
from card_encoding import CARD_TO_INT
from pql.ranges.ast import (Pair, PairPlus, PairRange, Suited, Offsuit, Both, Combo,
                            Percentile, Any, Pattern, PercentileBand, Exclude)

RANK_ORDER = '23456789TJQKA'           # idx 0..12
RIDX = {r: i for i, r in enumerate(RANK_ORDER)}


def _card(rank_idx: int, suit: int) -> int:
    return rank_idx * 4 + suit


def _pair_combos(r: int):
    return [frozenset((_card(r, s1), _card(r, s2))) for s1, s2 in combinations(range(4), 2)]


def _two_rank_combos(r1: int, r2: int, mode: str):
    out = []
    for s1 in range(4):
        for s2 in range(4):
            if mode == 'suited' and s1 != s2:
                continue
            if mode == 'offsuit' and s1 == s2:
                continue
            out.append(frozenset((_card(r1, s1), _card(r2, s2))))
    return out


def _holdem_term(term) -> list:
    if isinstance(term, Pair):
        return _pair_combos(RIDX[term.rank])
    if isinstance(term, PairPlus):
        return [c for r in range(RIDX[term.rank], 13) for c in _pair_combos(r)]
    if isinstance(term, PairRange):
        lo, hi = RIDX[term.low], RIDX[term.high]
        return [c for r in range(min(lo, hi), max(lo, hi) + 1) for c in _pair_combos(r)]
    if isinstance(term, Suited):
        return _two_rank_combos(RIDX[term.r1], RIDX[term.r2], 'suited')
    if isinstance(term, Offsuit):
        return _two_rank_combos(RIDX[term.r1], RIDX[term.r2], 'offsuit')
    if isinstance(term, Both):
        return _two_rank_combos(RIDX[term.r1], RIDX[term.r2], 'both')
    if isinstance(term, Any):
        return [frozenset(c) for c in combinations(range(52), 2)]
    raise ValueError(f"Unsupported holdem term {term}")


def _combo_term(term: Combo) -> list:
    cards = [CARD_TO_INT[term.text[i:i+2]] for i in range(0, len(term.text), 2)]
    return [frozenset(cards)]


def _omaha_pattern(term, num_hole: int) -> list:
    """Expand an Omaha rank pattern (required ranks + wildcards + optional suitedness)
    into frozenset card-int combos of size num_hole."""
    fixed_ranks = [RIDX[r] for r in term.ranks]
    rest = num_hole - len(fixed_ranks)
    if rest < 0:
        raise ValueError(f"pattern has more ranks than the {num_hole}-card game allows")
    need = Counter(fixed_ranks)
    fixed_choice_lists = []
    for rank, k in need.items():
        cards_of_rank = [_card(rank, s) for s in range(4)]
        fixed_choice_lists.append(list(combinations(cards_of_rank, k)))
    out = set()
    for fixed_combo in _product(*fixed_choice_lists):
        flat = [c for grp in fixed_combo for c in grp]
        baseset = set(flat)
        if len(baseset) != len(flat):
            continue
        others = [c for c in range(52) if c not in baseset]
        for extra in combinations(others, rest):
            hand = frozenset(baseset | set(extra))
            if len(hand) == num_hole and _suitedness_ok(hand, term.suitedness):
                out.add(hand)
    return list(out)


def _suitedness_ok(hand: frozenset, suitedness) -> bool:
    if suitedness is None:
        return True
    suit_counts = Counter(c % 4 for c in hand)
    if suitedness == 'ds':
        return sum(1 for v in suit_counts.values() if v == 2) >= 2 and max(suit_counts.values()) <= 2
    if suitedness == 'ss':
        twos = [v for v in suit_counts.values() if v >= 2]
        return len(twos) == 1 and twos[0] == 2
    if suitedness == 'ns':
        return all(v == 1 for v in suit_counts.values())
    return True


def _dead_ints(dead) -> set:
    if isinstance(dead, str):
        return {CARD_TO_INT[dead[i:i+2]] for i in range(0, len(dead), 2)}
    return set(int(c) for c in dead)


def _expand_terms(terms, game: str, num_hole: int, dead_set: set) -> set:
    seen: set = set()
    for term in terms:
        if isinstance(term, Exclude):
            base = _expand_terms(term.base, game, num_hole, dead_set)
            excl = _expand_terms(term.excl, game, num_hole, dead_set)
            seen |= (base - excl)
            continue
        if isinstance(term, Combo):
            combos = _combo_term(term)
        elif isinstance(term, PercentileBand):
            from pql.ranges.percentile import top_pct_combos
            hi = {frozenset(r.tolist()) for r in top_pct_combos(game, term.hi, dead_set)}
            lo = {frozenset(r.tolist()) for r in top_pct_combos(game, term.lo, dead_set)}
            combos = hi - lo
        elif isinstance(term, Percentile):
            from pql.ranges.percentile import top_pct_combos
            combos = [frozenset(row.tolist()) for row in top_pct_combos(game, term.pct, dead_set)]
        elif game == 'holdem':
            combos = _holdem_term(term)
        else:
            combos = _omaha_pattern(term, num_hole)
        for c in combos:
            if c & dead_set:
                continue
            if len(c) != num_hole:
                continue
            seen.add(c)
    return seen


def expand(terms: list, game: str, dead='') -> np.ndarray:
    """Expand a parsed range (list of terms) into an (M, num_hole) array of sorted card ints,
    deduplicated and dead-aware. Supports unions, Exclude (set difference), and PercentileBand."""
    from pql.games import get_game
    num_hole = get_game(game).num_hole
    dead_set = _dead_ints(dead)
    seen = _expand_terms(terms, game, num_hole, dead_set)
    if not seen:
        return np.empty((0, num_hole), dtype=np.int32)
    return np.array([sorted(c) for c in seen], dtype=np.int32)
