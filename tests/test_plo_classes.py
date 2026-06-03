from math import comb
from itertools import combinations
from pql.ranges.plo_classes import canon_key, expand_class
from card_encoding import CARD_TO_INT


def test_canon_key_groups_suit_isomorphic_hands():
    # AsAhKsKh and AcAdKcKd are the same suit-iso class.
    a = [CARD_TO_INT[x] for x in ('As', 'Ah', 'Ks', 'Kh')]
    b = [CARD_TO_INT[x] for x in ('Ac', 'Ad', 'Kc', 'Kd')]
    assert canon_key(a) == canon_key(b)
    # AsAhKsKh (double-suited) is NOT the same class as AsAhKcKd (single-suited As/Ah... )
    c = [CARD_TO_INT[x] for x in ('As', 'Ah', 'Kc', 'Kd')]
    assert canon_key(a) != canon_key(c)


def test_canon_key_is_a_valid_representative():
    a = [CARD_TO_INT[x] for x in ('Kh', 'As', 'Ah', 'Ks')]
    k = canon_key(a)
    assert len(set(k)) == 4 and all(0 <= x < 52 for x in k)


def test_expand_class_is_the_suit_orbit():
    rep = canon_key([CARD_TO_INT[x] for x in ('As', 'Ah', 'Ks', 'Kh')])
    combos = expand_class(rep, set())
    # every expanded combo canonicalizes back to the same class
    assert all(canon_key(tuple(c)) == rep for c in combos)
    # dead-aware: removing a card drops the combos that use it
    dead = {CARD_TO_INT['As']}
    assert all(CARD_TO_INT['As'] not in c for c in expand_class(rep, dead))


def test_classes_partition_the_plo4_hand_space():
    # The union of all PLO4 classes' orbits covers C(52,4) hands exactly once.
    seen_keys = {}
    for combo in combinations(range(52), 4):
        seen_keys[canon_key(combo)] = True
    total = sum(len(expand_class(rep, set())) for rep in seen_keys)
    assert total == comb(52, 4)
