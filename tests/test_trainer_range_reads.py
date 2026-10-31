"""Range Reads question bank: exact flop features over GTO ranges."""
import numpy as np
import pytest

from trainer.range_reads import (
    TEMPLATES, answer, card, card_str, flop_features, load_ranges,
    questions_for_flop, sample_flops, texture,
)


def feats(board: str, hand: str):
    flop = [card(board[i:i + 2]) for i in range(0, 6, 2)]
    h = np.array([[card(hand[i:i + 2]) for i in range(0, 8, 2)]], dtype=np.int64)
    return {k: v[0] for k, v in flop_features(h, flop).items()}


@pytest.mark.parametrize("board,hand,expected", [
    ("9s8h2d", "TcJd7h7s", 16),   # 6,7,T,J,Q -> 4+2+3+3+4 live cards
    ("9s8h2d", "AcKd4h3s", 0),
    ("Th9c4d", "QsJh8c7d", 20),   # 20-out wrap
    ("2d7h6d", "4sAd8d2h", 4),    # only the 5s; flush outs are NOT straight outs
])
def test_straight_outs(board, hand, expected):
    assert feats(board, hand)["straight_outs"] == expected


@pytest.mark.parametrize("board,hand,nuts", [
    ("7dAcKh", "AsAd2c3c", True),
    ("7dAcKh", "KsKd2c3c", False),
    ("9s8h7d", "JcTd2s2h", True),
    ("Ks9s4s", "AsQs2d3c", True),
    ("Ks9s4s", "QsJs2d3c", False),
    ("8h8d3c", "8s8c4h5h", True),
    ("8h8d3c", "8s3s4h5h", False),
])
def test_nuts(board, hand, nuts):
    assert bool(feats(board, hand)["nuts"]) is nuts


def test_flush_draws_and_sets():
    f = feats("Kh9h4c", "Ah2h7s7d")
    assert f["flush_draw"] and f["nut_flush_draw"] and not f["set"]
    f = feats("Kh9h4c", "QhJh9s9d")
    assert f["flush_draw"] and not f["nut_flush_draw"] and f["set"] and not f["top_set"]
    f = feats("Ah9h4c", "KhQh2s3d")          # A on board -> K is the nut flush card
    assert f["nut_flush_draw"]
    f = feats("Kh9h4c", "KsKd2c3c")
    assert f["set"] and f["top_set"] and f["two_pair_plus"]
    f = feats("Kh9h4h", "Ah2h7s7d")          # monotone: a made flush, not a draw
    assert f["flush"] and not f["flush_draw"]


def test_air():
    assert feats("Kh9h4c", "2s3d6c7c")["air"]
    assert not feats("Kh9h4c", "2h3h6c7c")["air"]   # flush draw
    assert not feats("Kh9h4c", "Ks3d6c7c")["air"]   # pair


def test_answer_is_weighted_fraction():
    feat = np.array([True, False, True])
    w = np.array([1.0, 2.0, 0.5])
    assert answer(feat, w, None) == pytest.approx(1.5 / 3.5)
    assert answer(np.array([3, 9, 13]), w, 8) == pytest.approx(2.5 / 3.5)


def test_bank_questions_are_well_formed():
    ranges = load_ranges()
    flop = [card(c) for c in ("Js", "8h", "3h")]
    qs = questions_for_flop(ranges, flop)
    assert {q["position"] for q in qs} == {"UTG", "MP", "CO", "BTN", "SB"}
    assert len({q["id"] for q in qs}) == len(qs)
    for q in qs:
        assert 0 <= q["answer"] <= 100
    by = {(q["position"], q["template"]): q["answer"] for q in qs}
    # J-8-3 can't make a straight, and there is no straight template here.
    assert ("BTN", "straight") not in by
    # A tighter range has fewer total combos but these stay sane fractions.
    assert 0 < by[("UTG", "flush_draw")] < 100


def test_ranges_match_published_rfi_frequencies():
    # Weighted share of all 270,725 PLO4 combos == the frequencies the
    # Hand Ranges page shows for PLO50 100BB.
    expected = {"EP": 16.8, "MP": 23.5, "CO": 32.9, "BU": 53.5, "SB": 37.4}
    for pos, (_, w) in load_ranges().items():
        assert w.sum() / 270725 * 100 == pytest.approx(expected[pos], abs=0.05)


def test_sample_flops_is_deterministic_and_balanced():
    a, b = sample_flops(20, seed=1), sample_flops(20, seed=1)
    assert a == b
    kinds = {("paired" if texture(f)["paired"] else texture(f)["suits"]) for f in a}
    assert kinds == {"rainbow", "two-tone", "monotone", "paired"}


def test_templates_have_unique_ids():
    assert len({t.id for t in TEMPLATES}) == len(TEMPLATES)


def test_card_round_trip():
    assert all(card(card_str(c)) == c for c in range(52))
