import numpy as np
from pql.ranges.parse import parse_range
from pql.ranges.expand import expand


def n(text, game, dead=''):
    return len(expand(parse_range(text, game), game, dead))


def test_holdem_counts():
    assert n('AA', 'holdem') == 6
    assert n('AKs', 'holdem') == 4
    assert n('AKo', 'holdem') == 12
    assert n('AK', 'holdem') == 16
    assert n('QQ+', 'holdem') == 18          # QQ,KK,AA
    assert n('22-99', 'holdem') == 48        # 8 pairs * 6
    assert n('*', 'holdem') == 1326          # C(52,2)
    assert n('AsKh', 'holdem') == 1
    assert n('AA,KK', 'holdem') == 12


def test_holdem_dead_aware():
    assert n('AA', 'holdem', dead='As') == 3


def test_expand_returns_sorted_int_combos():
    combos = expand(parse_range('AKs', 'holdem'), 'holdem', '')
    assert combos.shape == (4, 2)
    assert (combos[:, 0] < combos[:, 1]).all()


def test_omaha_pattern_counts():
    # AAxx PLO4 = hands with >= 2 aces among 4 cards = 6*C(48,2)+4*48+1 = 6961
    assert n('AAxx', 'omahahi') == 6961
    assert n('AsAhKsKh', 'omahahi') == 1
    # AKQJ double-suited (PLO4): 3 partitions * 6 suit-pairs * 2 = 36
    assert n('AKQJds', 'omahahi') == 36


def test_omaha_dead_aware():
    base = n('AAxx', 'omahahi')
    with_dead = n('AAxx', 'omahahi', dead='Ac')   # removing one ace shrinks the set
    assert with_dead < base
