"""Parse ProPokerTools' ordering-file notation into our canonical class keys.

A class string is a sequence of suit groups: `(XY..)` is a parenthesized group whose
ranks share one suit; each bare rank character is its own (singleton) suit group.
  holdem:  (AK)->AKs, AK->AKo, AA->pair
  omaha:   (AJ)(AJ), AATT (rainbow), AA(JT), (A9)(A9)T
"""
from __future__ import annotations
from card_encoding import RANKS, int_to_card
from pql.ranges.plo_classes import canon_key

_RIDX = {r: i for i, r in enumerate(RANKS)}   # '2'->0 .. 'A'->12


def parse_ppt_ordering_line(line: str) -> tuple:
    """Parse one PPT ordering class string into a canonical sorted tuple of card ints."""
    line = line.strip()
    groups = []
    i = 0
    while i < len(line):
        if line[i] == '(':
            j = line.index(')', i)
            groups.append(list(line[i + 1:j]))
            i = j + 1
        else:
            groups.append([line[i]])
            i += 1
    cards = []
    for suit, grp in enumerate(groups):
        for r in grp:
            cards.append(_RIDX[r] * 4 + suit)
    return canon_key(cards)


def class_key_to_string(key: tuple, holdem: bool = False) -> str:
    """Render a class key as the JSON entry its consumer expects:
    holdem -> 'AKs'/'AKo'/'AA'; omaha -> the canonical concrete hand string."""
    if holdem:
        a, b = key
        ra, sa = divmod(a, 4)
        rb, sb = divmod(b, 4)
        if ra == rb:
            return RANKS[ra] + RANKS[ra]
        # higher rank first
        (hr, hs), (lr, ls) = ((ra, sa), (rb, sb)) if ra > rb else ((rb, sb), (ra, sa))
        suited = 's' if hs == ls else 'o'
        return RANKS[hr] + RANKS[lr] + suited
    return ''.join(int_to_card(c) for c in key)
