"""PLO Trainer question generation: features, ranges, modes."""
from math import comb

import numpy as np
import pytest

from trainer.build import sample_flops
from trainer.features import cards, flop_features, texture
from trainer.questions import (
    TEMPLATE_BY_ID, FlopRanges, card_removal, equity_vs_range, range_reads, range_vs_range,
)
from trainer.ranges import plo4, plo5


def feats(board: str, hand: str):
    h = np.array([cards(hand)], dtype=np.int64)
    return {k: v[0] for k, v in flop_features(h, cards(board)).items()}


@pytest.mark.parametrize("board,hand,expected", [
    ("9s8h2d", "TcJd7h7s", 16),      # 6,7,T,J,Q -> 4+2+3+3+4 live cards
    ("9s8h2d", "AcKd4h3s", 0),
    ("Th9c4d", "QsJh8c7d", 20),      # 20-out wrap
    ("2d7h6d", "4sAd8d2h", 4),       # only the 5s; flush outs are not straight outs
    ("9s8h2d", "TcJd7h7s6c", 19),    # PLO5: Q4 + 5x4 + T3 + J3 + 6x3 + 7x2
])
def test_straight_outs(board, hand, expected):
    assert feats(board, hand)["straight_outs"] == expected


@pytest.mark.parametrize("board,hand,nuts", [
    ("7dAcKh", "AsAd2c3c", True), ("7dAcKh", "KsKd2c3c", False),
    ("9s8h7d", "JcTd2s2h", True), ("Ks9s4s", "AsQs2d3c", True), ("Ks9s4s", "QsJs2d3c", False),
    ("8h8d3c", "8s8c4h5h", True), ("8h8d3c", "8s3s4h5h", False),
    ("Ks9s4s", "AsQs2d3c4h", True),
])
def test_nuts(board, hand, nuts):
    assert bool(feats(board, hand)["nuts"]) is nuts


def test_draws_sets_air():
    f = feats("Kh9h4c", "Ah2h7s7d")
    assert f["flush_draw"] and f["nut_flush_draw"] and not f["set"]
    f = feats("Kh9h4c", "QhJh9s9d")
    assert f["flush_draw"] and not f["nut_flush_draw"] and f["set"] and not f["top_set"]
    assert feats("Ah9h4c", "KhQh2s3d")["nut_flush_draw"]         # A on board -> K is the nut card
    f = feats("Kh9h4h", "Ah2h7s7d")
    assert f["flush"] and not f["flush_draw"]
    assert feats("Kh9h4c", "2s3d6c7c")["air"]
    assert not feats("Kh9h4c", "2h3h6c7c")["air"]
    assert not feats("Kh9h4c", "Ks3d6c7c")["air"]


def test_categories_match_engine():
    from pql import run_pql
    rng = np.random.default_rng(5)
    for _ in range(30):
        flop = rng.choice(52, 3, replace=False).tolist()
        deck = [c for c in range(52) if c not in flop]
        h = np.array([rng.choice(deck, 4, replace=False)], dtype=np.int64)
        mine = int(flop_features(h, flop)["category"][0])
        from trainer.features import card_str
        b, hs = "".join(card_str(c) for c in flop), "".join(card_str(int(c)) for c in h[0])
        hist = run_pql(f"select histogram(handType(PLAYER_1, flop)) as c from game='omahahi', board='{b}', "
                       f"PLAYER_1='{hs}'", trials=20).histograms["c"]
        assert max(hist["pairs"], key=lambda p: p["count"])["value"] == mine


def test_plo4_ranges_match_published_frequencies():
    expected = {"EP": 16.8, "MP": 23.5, "CO": 32.9, "BU": 53.5, "SB": 37.4}
    for pos, (_, w) in plo4().items():
        assert w.sum() / comb(52, 4) * 100 == pytest.approx(expected[pos], abs=0.05)


def test_plo5_ranges_assumed_from_plo4():
    r5, r4 = plo5(), plo4()
    for pos in r4:
        share4 = r4[pos][1].sum() / comb(52, 4)
        assert len(r5[pos][0]) / comb(52, 5) == pytest.approx(share4, abs=1e-4)
    # Nested: every tighter range is a prefix of the BTN range.
    btn = r5["BU"][0]
    for pos in ("EP", "MP", "CO", "SB"):
        assert np.array_equal(r5[pos][0], btn[:len(r5[pos][0])])
    # Trips are discounted like in the PLO4 solve (random-deal rate is ~2.3%).
    ranks = r5["EP"][0] // 4
    trips = (np.stack([(ranks == k).sum(1) for k in range(13)], 1).max(1) >= 3).mean()
    assert trips < 0.02


def test_equity_matches_engine_on_a_uniform_range():
    """Hero-vs-range Monte Carlo agrees with the engine's riverEquity."""
    from pql import run_pql
    from pql.ranges.percentile import top_pct_combos
    flop, hero = cards("Js8h3h"), np.array(cards("AhKhQs9d"))
    # Same pool the engine samples: top 25% first, conflicting combos dropped after.
    pool = top_pct_combos("omahahi", 25, set()).astype(np.int64)
    mine = equity_vs_range(hero, pool, np.ones(len(pool)), flop, 60000, np.random.default_rng(0))
    eng = run_pql("select avg(riverEquity(PLAYER_1)) as e from game='omahahi', board='Js8h3h', "
                  "PLAYER_1='AhKhQs9d', PLAYER_2='25%'", trials=30000, seed=1).values["e"] * 100
    assert mine == pytest.approx(eng, abs=1.5)


@pytest.fixture(scope="module")
def fr():
    return FlopRanges("PLO4", cards("Th9h4c"))


def test_range_reads_questions(fr):
    qs = range_reads(fr)
    assert {q["position"] for q in qs} == {"UTG", "MP", "CO", "BTN", "SB"}
    assert len({q["id"] for q in qs}) == len(qs)
    assert all(0 < q["answer"] <= 100 and q["kind"] == "estimate" for q in qs)


def test_card_removal_answers_are_exact(fr):
    qs = card_removal(fr, np.random.default_rng(3))
    assert qs
    for q in qs:
        pos = {v: k for k, v in __import__("trainer.ranges", fromlist=["POSITIONS"]).POSITIONS.items()}[q["position"]]
        tpl = TEMPLATE_BY_ID[q["template"]]
        assert q["baseline"] == pytest.approx(fr.freq(pos, tpl), abs=0.01)
        assert q["answer"] == pytest.approx(fr.freq(pos, tpl, cards("".join(q["hero"]))), abs=0.01)
        assert abs(q["answer"] - q["baseline"]) >= 1.5
        assert not set(q["hero"]) & set(q["board"])


def test_range_vs_range_correct_matches_values(fr):
    qs = range_vs_range(fr, np.random.default_rng(3))
    assert qs
    for q in qs:
        assert q["kind"] == "choice" and q["choices"] == q["positions"]
        assert q["values"][q["correct"]] > q["values"][1 - q["correct"]]


def test_sample_flops_deterministic_and_balanced():
    a = sample_flops(20, seed=1)
    assert a == sample_flops(20, seed=1)
    kinds = {"paired" if texture(f)["paired"] else texture(f)["suits"] for f in a}
    assert kinds == {"rainbow", "two-tone", "monotone", "paired"}
