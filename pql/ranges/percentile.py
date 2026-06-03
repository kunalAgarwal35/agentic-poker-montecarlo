from __future__ import annotations
import os
import json
import functools
import numpy as np
from itertools import combinations
from math import comb
from card_encoding import RANKS
from pql.ranges.plo_classes import expand_class


@functools.lru_cache(maxsize=1)
def _holdem_class_order():
    """Premium-first ordering of the 169 holdem starting-hand classes, loaded from a
    committed precompute (generate_holdem_ranking.py)."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'holdem_class_order.json')
    with open(path) as f:
        order = json.load(f)
    assert len(order) == 169 and len(set(order)) == 169, "holdem_class_order.json must list 169 unique classes"
    return order


def _holdem_class_combos(cls: str, dead: set):
    r1, r2 = cls[0], cls[1]
    a, b = RANKS.index(r1), RANKS.index(r2)
    out = []
    if r1 == r2:
        for s1, s2 in combinations(range(4), 2):
            c = frozenset((a * 4 + s1, a * 4 + s2))
            if not (c & dead):
                out.append(c)
    else:
        suited = cls.endswith('s')
        for s1 in range(4):
            for s2 in range(4):
                if suited and s1 != s2:
                    continue
                if (not suited) and s1 == s2:
                    continue
                c = frozenset((a * 4 + s1, b * 4 + s2))
                if not (c & dead):
                    out.append(c)
    return out


_PLO_FILE = {'omahahi': 'plo4_class_order.json', 'omahahi5': 'plo5_class_order.json',
             'omahahi6': 'plo6_class_order.json'}
_PLO_TOTAL = {'omahahi': comb(52, 4), 'omahahi5': comb(52, 5), 'omahahi6': comb(52, 6)}


@functools.lru_cache(maxsize=4)
def _plo_class_order(game: str):
    fname = _PLO_FILE[game]
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), fname)
    if not os.path.exists(path):
        raise NotImplementedError(
            f"Percentile ranges for game={game} need {fname}, which hasn't been generated yet. "
            f"Run `python generate_plo_ranking.py --variant {fname.split('_')[0]}`, or use an "
            f"explicit range (e.g. 'AAxx')."
        )
    with open(path) as f:
        return json.load(f)


def top_pct_combos(game: str, pct: float, dead: set) -> np.ndarray:
    if game == 'holdem':
        target = int(round(1326 * pct / 100.0))
        seen = []
        for cls in _holdem_class_order():
            for combo in _holdem_class_combos(cls, dead):
                seen.append(combo)
            if len(seen) >= target:
                break
        return np.array([sorted(x) for x in list(dict.fromkeys(seen))[:target]], dtype=np.int32)
    if game in _PLO_TOTAL:
        order = _plo_class_order(game)                  # raises NotImplementedError if no artifact
        target = int(round(_PLO_TOTAL[game] * pct / 100.0))
        seen = []
        for rep in order:
            for combo in expand_class(rep, dead):
                seen.append(combo)
            if len(seen) >= target:
                break
        return np.array([sorted(c) for c in list(dict.fromkeys(seen))[:target]], dtype=np.int32)
    raise NotImplementedError(
        f"Percentile ranges for game={game} aren't supported. Use an explicit range or a supported game."
    )
