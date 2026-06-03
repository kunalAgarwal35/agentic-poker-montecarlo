from __future__ import annotations
import re
from typing import List
from pql.ranges.ast import (Pair, PairPlus, PairRange, Suited, Offsuit, Both, Combo,
                            Percentile, Any, Pattern, PercentileBand, Exclude)

RANKS = 'AKQJT98765432'          # high->low for parsing convenience
RANK_SET = set('23456789TJQKA')
SUIT_SET = set('shdc')
_PCT = re.compile(r'^(\d+(?:\.\d+)?)%$')
_BAND = re.compile(r'^(\d+(?:\.\d+)?)%-(\d+(?:\.\d+)?)%$')
_COMBO = re.compile(r'^([2-9TJQKA][shdc])+$')


def parse_range(text: str, game: str) -> List:
    if '!' in text:
        base_str, excl_str = text.split('!', 1)
        return [Exclude(tuple(parse_range(base_str, game)), tuple(parse_range(excl_str, game)))]
    return [_parse_term(t.strip(), game) for t in text.split(',') if t.strip()]


def _parse_term(t: str, game: str):
    if t == '*' or t == '100%':
        return Any()
    mb = _BAND.match(t)
    if mb:
        a, b = float(mb.group(1)), float(mb.group(2))
        return PercentileBand(min(a, b), max(a, b))
    m = _PCT.match(t)
    if m:
        return Percentile(float(m.group(1)))
    if _COMBO.match(t):
        return Combo(t)
    if game == 'holdem':
        return _parse_holdem(t)
    return _parse_omaha(t)


def _parse_holdem(t: str):
    if len(t) >= 2 and t[0] == t[1] and t[0] in RANK_SET:
        if t.endswith('+'):
            return PairPlus(t[0])
        if '-' in t:  # 22-99
            lo, hi = t.split('-')
            return PairRange(lo[0], hi[0])
        return Pair(t[0])
    base = t.rstrip('+')
    if len(base) == 3 and base[2] in 'so':
        r1, r2, so = base[0], base[1], base[2]
        return Suited(r1, r2) if so == 's' else Offsuit(r1, r2)
    if len(base) == 2:
        return Both(base[0], base[1])
    raise ValueError(f"Unrecognized holdem range term: '{t}'")


def _parse_omaha(t: str):
    suitedness = None
    for suf in ('ds', 'ss', 'ns'):
        if t.endswith(suf):
            suitedness = suf
            t = t[: -len(suf)]
            break
    ranks = [c for c in t if c in RANK_SET]
    num_wild = 4 - len(ranks)  # wildcards are implied for omaha (4-card hand)
    if not ranks and num_wild == 4:
        raise ValueError(f"Unrecognized omaha range term: '{t}'")
    return Pattern(ranks, num_wild, suitedness)
