from pql.ranges.parse import parse_range
from pql.ranges.expand import expand
from pql.ranges.ast import Exclude, PercentileBand, Percentile


def _set(arr):
    return {frozenset(row.tolist()) for row in arr}


def test_exclusion_parses_and_subtracts():
    terms = parse_range('QQ+!AA', 'holdem')
    assert len(terms) == 1 and isinstance(terms[0], Exclude)
    got = _set(expand(terms, 'holdem'))
    want = _set(expand(parse_range('QQ,KK', 'holdem'), 'holdem'))
    assert got == want and len(got) == 12   # QQ(6) + KK(6)


def test_percentile_exclusion_holdem():
    # 30% minus top 5% is non-empty and a strict subset of top 30%.
    got = _set(expand(parse_range('30%!5%', 'holdem'), 'holdem'))
    full30 = _set(expand(parse_range('30%', 'holdem'), 'holdem'))
    top5 = _set(expand(parse_range('5%', 'holdem'), 'holdem'))
    assert got and got < full30
    assert got.isdisjoint(top5)


def test_band_token_parses_and_equals_exclusion():
    terms = parse_range('5%-30%', 'holdem')
    assert len(terms) == 1 and isinstance(terms[0], PercentileBand)
    assert terms[0].lo == 5.0 and terms[0].hi == 30.0
    band = _set(expand(terms, 'holdem'))
    excl = _set(expand(parse_range('30%!5%', 'holdem'), 'holdem'))
    assert band == excl


def test_exclusion_is_dead_aware():
    # QQ+!QQ = (QQ,KK,AA) minus QQ = KK + AA; with Ah dead, no combo may contain Ah.
    got = _set(expand(parse_range('QQ+!QQ', 'holdem'), 'holdem', dead='Ah'))
    assert got                                       # KK + AA combos remain
    assert all(49 not in combo for combo in got)     # Ah int = 12*4 + 1 = 49
