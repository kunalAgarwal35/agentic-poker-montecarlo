from pql.ranges.parse import parse_range
from pql.ranges.ast import Pair, PairPlus, Suited, Offsuit, Both, Combo, Pattern, Percentile, Any


def test_holdem_terms():
    assert parse_range('AA', 'holdem') == [Pair('A')]
    assert parse_range('QQ+', 'holdem') == [PairPlus('Q')]
    assert parse_range('AKs', 'holdem') == [Suited('A', 'K')]
    assert parse_range('AKo', 'holdem') == [Offsuit('A', 'K')]
    assert parse_range('AK', 'holdem') == [Both('A', 'K')]
    assert parse_range('*', 'holdem') == [Any()]
    assert parse_range('25%', 'holdem') == [Percentile(25.0)]


def test_union_and_combo():
    assert parse_range('AA,KK,AKs', 'holdem') == [Pair('A'), Pair('K'), Suited('A', 'K')]
    assert parse_range('AsKh', 'holdem') == [Combo('AsKh')]


def test_omaha_pattern():
    assert parse_range('AAxx', 'omahahi') == [Pattern(['A', 'A'], num_wild=2, suitedness=None)]
    assert parse_range('AAds', 'omahahi') == [Pattern(['A', 'A'], num_wild=2, suitedness='ds')]
    assert parse_range('AsAhKsKh', 'omahahi') == [Combo('AsAhKsKh')]
