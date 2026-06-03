import os
import json
from math import comb
import pytest
from pql.ranges.plo_classes import expand_class

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(variant):
    path = os.path.join(_ROOT, f'{variant}_class_order.json')
    if not os.path.exists(path):
        pytest.skip(f'{variant}_class_order.json not generated')
    with open(path) as f:
        return json.load(f)


def test_plo4_order_covers_every_hand_once():
    order = _load('plo4')
    total = sum(len(expand_class(rep, set())) for rep in order)
    assert total == comb(52, 4)


def test_plo4_order_is_premium_first():
    order = _load('plo4')
    # the strongest class is double-suited aces (AAKK ds-ish); at minimum the top hand is all aces-high.
    top = order[0]
    ranks = ''.join(top[i] for i in range(0, len(top), 2))
    assert ranks.count('A') >= 2   # top class is aces-heavy


def test_plo4_order_has_no_duplicates():
    order = _load('plo4')
    assert len(order) == len(set(order))


def test_holdem_order_matches_ppt():
    import json
    path = os.path.join(_ROOT, 'holdem_class_order.json')
    order = json.load(open(path))
    assert len(order) == 169 and len(set(order)) == 169
    # PPT's exact order: AKs ranks above TT, AKo above 99 (not equity-vs-random).
    assert order[:9] == ['AA', 'KK', 'QQ', 'JJ', 'AKs', 'TT', 'AKo', 'AQs', '99']
