import numpy as np
from pql.ranges.percentile import top_pct_combos


def test_holdem_top_pct_is_premium_pairs():
    combos = top_pct_combos('holdem', 5.0, set())
    from card_encoding import CARD_TO_INT
    aa = frozenset({CARD_TO_INT['As'], CARD_TO_INT['Ah']})
    flat = [frozenset(c.tolist()) for c in combos]
    assert aa in flat
    assert frozenset({CARD_TO_INT['7c'], CARD_TO_INT['2h']}) not in flat


def test_plo5_top25_matches_loh_count():
    import os
    import pytest
    plo5 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plo5_class_order.json')
    if not os.path.exists(plo5):
        pytest.skip('plo5_class_order.json not generated yet')
    combos = top_pct_combos('omahahi5', 25.0, set())
    assert combos.shape[1] == 5
    assert combos.shape[0] > 1000


def test_plo4_top25_supported():
    combos = top_pct_combos('omahahi', 25.0, set())
    assert combos.shape[1] == 4
    assert combos.shape[0] > 1000


def test_plo6_percentile_raises():
    import pytest
    with pytest.raises(NotImplementedError):
        top_pct_combos('omahahi6', 25.0, set())


import os as _os
import pytest as _pytest

_PLO5_JSON = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'plo5_class_order.json')


def test_plo4_percentile_30_contains_25():
    from pql.ranges.percentile import top_pct_combos
    s25 = {frozenset(r.tolist()) for r in top_pct_combos('omahahi', 25.0, set())}
    s30 = {frozenset(r.tolist()) for r in top_pct_combos('omahahi', 30.0, set())}
    assert s25 and s25 < s30   # top 25% is a strict subset of top 30%


def test_plo5_percentile_beyond_25_works():
    if not _os.path.exists(_PLO5_JSON):
        _pytest.skip('plo5_class_order.json not generated yet')
    from pql.ranges.percentile import top_pct_combos
    combos = top_pct_combos('omahahi5', 30.0, set())   # was NotImplementedError
    assert combos.shape[0] > 0 and combos.shape[1] == 5


def test_holdem_percentile_truncates_to_exact_target():
    from pql.ranges.percentile import top_pct_combos
    # 5% of 1326 = 66.3 -> 66 combos exactly (no whole-class overshoot).
    combos = top_pct_combos('holdem', 5.0, set())
    assert combos.shape[0] == round(1326 * 5 / 100.0) == 66


def test_plo4_percentile_truncates_to_exact_target():
    from math import comb
    from pql.ranges.percentile import top_pct_combos
    combos = top_pct_combos('omahahi', 10.0, set())
    assert combos.shape[0] == round(comb(52, 4) * 10 / 100.0)
