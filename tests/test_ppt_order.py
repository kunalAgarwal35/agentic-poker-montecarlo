import os
import zipfile
import pytest
from pql.ranges.ppt_order import parse_ppt_ordering_line, class_key_to_string
from pql.ranges.plo_classes import canon_key
from card_encoding import CARD_TO_INT

_JAR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'java_files', 'ppt_base.jar')


def test_holdem_tokens_round_trip():
    assert class_key_to_string(parse_ppt_ordering_line('(AK)'), holdem=True) == 'AKs'
    assert class_key_to_string(parse_ppt_ordering_line('AK'), holdem=True) == 'AKo'
    assert class_key_to_string(parse_ppt_ordering_line('AA'), holdem=True) == 'AA'
    assert class_key_to_string(parse_ppt_ordering_line('(72)'), holdem=True) == '72s'


def test_omaha_double_suited_matches_canon():
    # (AJ)(AJ) == AsJsAhJh canonical class
    key = parse_ppt_ordering_line('(AJ)(AJ)')
    assert key == canon_key([CARD_TO_INT[x] for x in ('As', 'Js', 'Ah', 'Jh')])
    assert len(set(key)) == 4


def test_omaha_rainbow_differs_from_single_suited():
    rainbow = parse_ppt_ordering_line('AATT')      # no two cards share a suit
    one_suited = parse_ppt_ordering_line('AA(JT)')  # J,T suited; different ranks anyway
    assert rainbow != one_suited
    # rainbow AATT has no suited pair; verify all four suits distinct in canon
    assert len({k % 4 for k in rainbow}) == 4


def test_full_heordering_parses_to_169_unique():
    if not os.path.exists(_JAR):
        pytest.skip('p2.jar/ppt_base.jar not available')
    with zipfile.ZipFile(_JAR) as z:
        text = z.read('propokertools/core/orderings/heordering.txt').decode()
    keys = [parse_ppt_ordering_line(l.strip()) for l in text.splitlines() if l.strip()]
    assert len(keys) == 169 and len(set(keys)) == 169
