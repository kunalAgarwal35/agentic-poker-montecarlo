"""Suit-isomorphic PLO hand classes: canonicalize a hand under suit permutation,
and expand a canonical representative back to its concrete combos (the suit orbit)."""
from __future__ import annotations
from itertools import permutations
from card_encoding import CARD_TO_INT

_SUIT_PERMS = tuple(permutations(range(4)))   # all 24 relabelings of the 4 suits


def canon_key(cards) -> tuple:
    """Canonical (lexicographically minimal) representative of a hand under suit relabeling.
    cards: iterable of card ints (rank*4 + suit). Returns a sorted tuple of card ints."""
    best = None
    for p in _SUIT_PERMS:
        mapped = tuple(sorted((c // 4) * 4 + p[c % 4] for c in cards))
        if best is None or mapped < best:
            best = mapped
    return best


def _rep_to_ints(rep):
    if isinstance(rep, str):
        return [CARD_TO_INT[rep[i:i + 2]] for i in range(0, len(rep), 2)]
    return list(rep)


def expand_class(rep, dead_set: set) -> list:
    """All concrete combos of the class (the suit orbit of `rep`), dead-aware.
    rep: a representative as a card string ('AsAhKsKh') or an iterable of ints."""
    cards = _rep_to_ints(rep)
    out = set()
    for p in _SUIT_PERMS:
        mapped = frozenset((c // 4) * 4 + p[c % 4] for c in cards)
        if len(mapped) == len(cards) and not (mapped & dead_set):
            out.add(mapped)
    return list(out)
