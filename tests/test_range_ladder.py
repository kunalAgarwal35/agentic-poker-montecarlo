from itertools import combinations

import numpy as np
import pytest
from card_encoding import hand_str_to_ints, generate_deck_ints
from hand_rank_evaluator import get_score_array
from range_ladder import (
    sample_villains,
    sample_runouts,
    eligible_mask,
    score_hands,
    evaluate_population,
    build_rungs,
    describe_category,
    compute_range_ladder,
)

def test_sample_villains_returns_distinct_legal_hands():
    deck = np.arange(20, 52, dtype=np.int32)      # 32 cards available
    rng = np.random.default_rng(7)
    out = sample_villains(deck, num_cards=6, n=500, rng=rng)

    assert out.shape == (500, 6)
    assert out.dtype == np.int32
    # every card comes from the deck
    assert np.isin(out, deck).all()
    # no card repeats inside a hand
    assert all(len(set(row.tolist())) == 6 for row in out)
    # rows are sorted, so identical hands are byte-identical
    assert (np.diff(out, axis=1) > 0).all()
    # no duplicate hands
    assert len({tuple(r) for r in out.tolist()}) == 500

def test_sample_villains_is_seed_deterministic():
    deck = np.arange(20, 52, dtype=np.int32)
    a = sample_villains(deck, 6, 100, np.random.default_rng(1))
    b = sample_villains(deck, 6, 100, np.random.default_rng(1))
    assert np.array_equal(a, b)

def test_sample_villains_caps_at_the_available_space():
    deck = np.arange(0, 7, dtype=np.int32)        # C(7,6) = 7 possible hands
    out = sample_villains(deck, 6, 500, np.random.default_rng(3))
    assert out.shape[0] == 7
    # every row sorted ascending
    assert (np.diff(out, axis=1) > 0).all()
    # seven distinct rows, not seven copies of the same hand
    assert len({tuple(r) for r in out.tolist()}) == 7
    # exactly the seven 6-card combinations of the deck
    expected = {tuple(sorted(c)) for c in combinations(deck.tolist(), 6)}
    assert {tuple(r) for r in out.tolist()} == expected

def test_sample_villains_exercises_rejection_sampling_branch():
    # C(10,4) = 210, only modestly larger than n=200: the dedup loop must
    # work hard to fill the request without hanging or looping forever.
    deck = np.arange(0, 10, dtype=np.int32)
    out = sample_villains(deck, 4, 200, np.random.default_rng(11))
    assert out.shape == (200, 4)
    assert len({tuple(r) for r in out.tolist()}) == 200

def test_sample_runouts_draws_the_missing_board_cards():
    deck = np.arange(10, 52, dtype=np.int32)
    out = sample_runouts(deck, board_len=3, r=200, rng=np.random.default_rng(5))
    assert out.shape == (200, 2)
    assert np.isin(out, deck).all()
    assert (out[:, 0] != out[:, 1]).all()

def test_river_yields_one_empty_runout():
    deck = np.arange(10, 52, dtype=np.int32)
    out = sample_runouts(deck, board_len=5, r=200, rng=np.random.default_rng(5))
    assert out.shape == (1, 0)

def test_eligible_mask_excludes_villains_holding_a_runout_card():
    villains = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    runout = np.array([4, 9], dtype=np.int32)     # card 4 sits in villain 0
    mask = eligible_mask(villains, runout)
    assert mask.tolist() == [False, True]

def test_empty_runout_leaves_every_villain_eligible():
    villains = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    mask = eligible_mask(villains, np.array([], dtype=np.int32))
    assert mask.tolist() == [True, True]

def test_score_hands_ranks_a_flush_above_a_pair():
    board5 = hand_str_to_ints("6s7s4s2h9d")
    hands = np.stack([
        hand_str_to_ints("AsKs9h2c"),   # nut flush
        hand_str_to_ints("6h6d3c2d"),   # trips/pair
    ])
    out = score_hands(hands, board5, "plo4", get_score_array())
    assert out.shape == (2,)
    assert out[0] > out[1]

def test_score_hands_chunking_matches_unchunked():
    rng = np.random.default_rng(11)
    deck = generate_deck_ints(["6s", "7s", "4s", "2h", "9d"])
    hands = sample_villains(deck, 4, 300, rng)
    board5 = hand_str_to_ints("6s7s4s2h9d")
    sa = get_score_array()
    a = score_hands(hands, board5, "plo4", sa, chunk=10_000)
    b = score_hands(hands, board5, "plo4", sa, chunk=7)
    assert np.array_equal(a, b)

def test_strength_orders_a_made_flush_above_air_on_the_river():
    board = hand_str_to_ints("6s7s4s2h9d")          # river: exact, no sampling
    villains = np.stack([
        hand_str_to_ints("AsKs9h2c"),               # nut flush
        hand_str_to_ints("Jd3d2d4d"),               # air
        hand_str_to_ints("6h6d3c5d"),               # pair
    ])
    heroes = np.stack([hand_str_to_ints("QsJs8c3h")])   # queen-high flush
    runouts = np.empty((1, 0), dtype=np.int32)
    strength, hero_equity, counts = evaluate_population(
        villains, heroes, board, runouts, "plo4", get_score_array()
    )
    assert strength[0] > strength[2] > strength[1]
    assert hero_equity.shape == (1, 3)
    assert hero_equity[0, 0] == 0.0      # loses to the nut flush
    assert hero_equity[0, 1] == 1.0      # beats air

def test_a_villain_holding_a_runout_card_is_skipped_not_scored():
    # Board 6s7s4s; the only runout is 2h9d. Villain 0 holds 2h, so the pairing
    # is impossible -- it must be skipped, not counted as a loss. Villains 1
    # and 2 don't collide, so the runout is still genuinely scored (idx.size
    # == 2 after masking out villain 0) -- an implementation that skipped the
    # whole runout unconditionally would fail the counts/strength assertions
    # below for villains 1 and 2.
    board = hand_str_to_ints("6s7s4s")
    villains = np.stack([
        hand_str_to_ints("As2hKd3c"),
        hand_str_to_ints("AhKh9c8c"),
        hand_str_to_ints("QdJcTd5h"),
    ])
    heroes = np.stack([hand_str_to_ints("QsJs8d3d")])
    runouts = np.stack([hand_str_to_ints("2h9d")])
    strength, hero_equity, counts = evaluate_population(
        villains, heroes, board, runouts, "plo4", get_score_array()
    )
    assert strength[0] == 0.0            # never eligible -> no data
    assert hero_equity[0, 0] == 0.0
    assert counts[0] == 0.0              # the collider contributed to nothing
    # villains 1 and 2 WERE scored against each other on this runout: counts
    # confirms two runout-contributions happened, and villain 1's strength is
    # non-zero, proving the runout was actually evaluated, not skipped.
    assert counts[1] == 1.0
    assert counts[2] == 1.0
    assert strength[1] == 1.0            # beats villain 2 outright
    assert hero_equity[0, 1] == 1.0      # hero also beats villain 1

def test_rungs_slice_by_strength_and_report_the_weakest_hand_in_each():
    # 10 villains with strengths 0.0 .. 0.9; hero beats exactly the weak half,
    # so index i (strength 0.1*i) has hero equity 1.0 for i < 5 and 0.0 above.
    villains = np.stack([hand_str_to_ints("AsKs9h2c")] * 10)
    strength = np.linspace(0.0, 0.9, 10)
    hero = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float64)

    rungs = build_rungs(strength, hero, villains, [20, 50, 100],
                        hand_str_to_ints("6s7s4s2h9d"))

    assert [r["bucket"] for r in rungs] == [20, 50, 100]
    # top 20% = the 2 strongest, which hero loses to
    assert rungs[0]["equity"] == 0.0
    # top 100% = everyone; hero beats 5 of 10
    assert rungs[2]["equity"] == 0.5
    # equity never decreases as the bucket widens
    eq = [r["equity"] for r in rungs]
    assert eq == sorted(eq)

def test_every_bucket_holds_at_least_one_hand():
    villains = np.stack([hand_str_to_ints("AsKs9h2c")] * 3)
    strength = np.array([0.1, 0.5, 0.9])
    hero = np.array([1.0, 1.0, 0.0])
    rungs = build_rungs(strength, hero, villains, [5, 100],
                        hand_str_to_ints("6s7s4s2h9d"))
    assert rungs[0]["equity"] == 0.0        # top 5% rounds up to 1 hand
    assert rungs[0]["edge"]["cards"] != ""


def test_river_ladder_is_exact_and_monotonic():
    # Board/hero fixture deliberately differs from the task-6 brief's literal
    # example (board="6s7s4s2h9d", dead=["AsKs9h2c", "QsJs8c3h"]). Verified
    # by direct computation: that board's 3 spades (4,6,7) are close enough
    # together that TWO straight-flush windows exist (3-4-5-6-7 and
    # 4-5-6-7-8), both sharing card 5s, which makes straight flushes ~1.6%
    # of the hand space instead of the ~0.1% a single window would give --
    # enough of them land in the top-5% bucket (about 30/100) that they
    # dominate it. Worse, the brief's dead cards (AsKs + QsJs) remove all
    # four spade honor-cards, so *every* ordinary villain flush is capped
    # below Ten kicker -- meaning both "nuts" (A-K flush) and "air" (Q-J
    # flush, not actually air: see
    # test_score_hands_ranks_a_flush_above_a_pair's comment on this same
    # string) are equally invincible against every non-straight-flush
    # villain and equally beaten by every straight flush. Their top-5%
    # equities come out identically 0.70 -- provably, for any seed/hand
    # count, not just this one -- so nuts > 0.85 and air < 0.15 can never
    # both hold. Confirmed via score_hands() directly: nuts=10000015.38,
    # air=10000013.18, both dwarfed by straight-flush scores ~1e10, and
    # nuts > air > every reachable ordinary-flush score.
    #
    # A first replacement fixture (board="2s8sKs7d4h",
    # nuts="AsQs9h2c", air="3d6c9hJd") was itself illegal -- both hands
    # contain 9h, which two players can never legally hold at once. That
    # collision is exactly what Finding 1's dead/board/hero validation now
    # catches (compute_range_ladder raises "duplicate card(s) in `dead`:
    # ['9h']" for it), so it can no longer be constructed at all, let alone
    # silently draw villains from a 45-card deck.
    #
    # The fixture below was found empirically (by calling
    # compute_range_ladder directly with several board/hand candidates and
    # printing the resulting ladders) to satisfy all of: no card shared
    # between board/nuts/air; nuts far exceeds the 0.85 floor at bucket 5;
    # air stays under the 0.15 ceiling at bucket 5; and at least one ladder
    # is strictly non-constant across the six buckets so `eq == sorted(eq)`
    # has more than a flat or two-point line to validate. It reuses the
    # brief's board (6s7s4s2h9d, still real nut flush territory for "nuts")
    # but keeps "air" a genuinely disconnected hand -- no pair with the
    # board, no flush cards, no rank/suit overlap with "nuts" -- instead of
    # a second, merely-lower flush. At hands=2000, seed=42, the actual
    # ladders are:
    #   nuts: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]                     (flat: this
    #         hand really is the nuts here -- no flush, straight, full
    #         house or better is reachable by anyone else on this board
    #         once As/Ks are dead, so it wins every single matchup)
    #   air:  [0.0, 0.0183, 0.32, 0.575, 0.7167, 0.83]           (strictly
    #         increasing -- this is the ladder that gives monotonicity
    #         something real to check)
    out = compute_range_ladder(
        board="6s7s4s2h9d",
        dead=["AsKs9h2c", "3dTc5s8h"],
        heroes=[{"id": "nuts", "cards": "AsKs9h2c"},
                {"id": "air",  "cards": "3dTc5s8h"}],
        hands=2000, seed=42,
    )
    assert out["exact"] is True
    assert out["runouts"] == 0
    assert out["population"] > 0

    nuts = next(l for l in out["ladders"] if l["id"] == "nuts")["rungs"]
    air = next(l for l in out["ladders"] if l["id"] == "air")["rungs"]

    # monotonic: equity never falls as the bucket widens
    for rungs in (nuts, air):
        eq = [r["equity"] for r in rungs]
        assert eq == sorted(eq), eq
    # at least one ladder must be strictly non-constant, or the monotonic
    # check above would pass vacuously on flat lines and could no longer
    # catch a regression that scrambles bucket ordering.
    assert len({r["equity"] for r in nuts}) > 1 or len({r["equity"] for r in air}) > 1

    # the whole point: nuts and air must look different vs the top 5%
    assert nuts[0]["equity"] > 0.85
    assert air[0]["equity"] < 0.15

    # boundary hands are shared -- ranking is hero-independent
    assert [r["edge"]["cards"] for r in nuts] == [r["edge"]["cards"] for r in air]


def test_hero_missing_from_dead_is_rejected():
    with pytest.raises(ValueError):
        compute_range_ladder(
            board="6s7s4s",
            dead=["AsKs9h2c"],
            heroes=[{"id": "x", "cards": "QsJs8c3h"}],   # not in dead
            hands=100, seed=1,
        )


def test_hero_card_on_the_board_is_rejected():
    # 6s is both on the board and in the hero's (dead) hand -- a card
    # cannot be simultaneously dead and live on the board.
    with pytest.raises(ValueError):
        compute_range_ladder(
            board="6s7s4s2h9d",
            dead=["6sKs9c2c"],
            heroes=[{"id": "x", "cards": "6sKs9c2c"}],
            hands=100, seed=1,
        )


def test_duplicate_card_within_one_hand_is_rejected():
    # "As" appears twice inside a single dead entry.
    with pytest.raises(ValueError):
        compute_range_ladder(
            board="6s7s4s",
            dead=["AsAs9c2c"],
            heroes=[{"id": "x", "cards": "AsAs9c2c"}],
            hands=100, seed=1,
        )


def test_duplicate_card_across_two_dead_entries_is_rejected():
    # "As" appears once in each of two different dead entries -- two
    # players can never legally hold the same card.
    with pytest.raises(ValueError):
        compute_range_ladder(
            board="6s7s4s",
            dead=["AsKc9c2c", "AsQd8c3c"],
            heroes=[{"id": "x", "cards": "AsKc9c2c"}],
            hands=100, seed=1,
        )


def test_describe_category_names_a_flush():
    assert describe_category(hand_str_to_ints("AsKs9h2c"),
                             hand_str_to_ints("6s7s4s2h9d")) == "flush"
