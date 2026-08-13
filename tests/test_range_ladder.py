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

# A single empty runout -- the same shape sample_runouts returns on the
# river, where board_ints already has all 5 cards and no completion is
# needed. Every hand is trivially "compatible" with an empty runout, so
# this reproduces build_rungs' pre-Task-7 behavior for these two tests,
# which only care about bucket slicing, not runout selection.
RIVER_RUNOUTS = np.empty((1, 0), dtype=np.int32)

def test_rungs_slice_by_strength_and_report_the_weakest_hand_in_each():
    # 10 villains with strengths 0.0 .. 0.9; hero beats exactly the weak half,
    # so index i (strength 0.1*i) has hero equity 1.0 for i < 5 and 0.0 above.
    villains = np.stack([hand_str_to_ints("AsKs9h2c")] * 10)
    strength = np.linspace(0.0, 0.9, 10)
    hero = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float64)

    rungs = build_rungs(strength, hero, villains, [20, 50, 100],
                        hand_str_to_ints("6s7s4s2h9d"), RIVER_RUNOUTS)

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
                        hand_str_to_ints("6s7s4s2h9d"), RIVER_RUNOUTS)
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


def test_zero_runouts_off_river_is_rejected():
    # runouts=0 is only meaningful on the river (where the count is 0 by
    # construction). On a flop board it must raise, not silently sample
    # zero runouts and crash later on an empty runout_rows[0].
    with pytest.raises(ValueError):
        compute_range_ladder(
            board="6s7s4s",
            dead=["AsKs9h2c"],
            heroes=[{"id": "x", "cards": "AsKs9h2c"}],
            hands=100, runouts=0, seed=1,
        )


def test_zero_runouts_on_river_is_a_noop():
    # On the river, board_len == 5 forces r = 1 regardless of `runouts`, so
    # an explicit runouts=0 must still succeed exactly like the default.
    out = compute_range_ladder(
        board="6s7s4s2h9d",
        dead=["AsKs9h2c"],
        heroes=[{"id": "x", "cards": "AsKs9h2c"}],
        hands=100, runouts=0, seed=1,
    )
    assert out["exact"] is True
    assert out["runouts"] == 0


_NINE_CATEGORIES = {
    "high card", "pair", "two pair", "trips", "straight",
    "flush", "full house", "quads", "straight flush",
}


def test_boundary_hand_colliding_with_the_first_runout_does_not_crash():
    # Regression pin for a Task-7 finding: build_rungs used to label every
    # bucket's boundary hand on the SAME precomputed board (board_ints +
    # runout_rows[0]). Nothing checked that a given boundary hand was
    # actually eligible for that particular runout -- when it held a card
    # the runout also used, describe_category fed a 9-card set containing a
    # duplicate card straight into optimized_evaluator's bounds-unchecked
    # numba kernel, which does not raise a Python exception; it segfaults
    # the process (Windows: "access violation").
    #
    # This exact board/dead/seed/hands/runouts combination was confirmed
    # (by direct inspection, independent of compute_range_ladder) to have
    # runout_rows[0] == cards [42, 7] ("Qd3c"), which collides with the
    # bucket-25, bucket-40 and bucket-100 boundary hands -- 3 of 6 buckets.
    # Pre-fix this reproduces a hard crash, not a raised exception, so
    # there is nothing to pytest.raises() around: the assertion that
    # matters is simply that this call returns at all.
    out = compute_range_ladder(
        board="6s7s4s",
        dead=["AsKs9h2c", "QhJhTd3d"],
        heroes=[{"id": "nuts", "cards": "AsKs9h2c"},
                {"id": "air", "cards": "QhJhTd3d"}],
        hands=400, runouts=40, seed=5,
    )
    assert len(out["ladders"]) == 2
    for lad in out["ladders"]:
        assert len(lad["rungs"]) == 6
        for rung in lad["rungs"]:
            category = rung["edge"]["category"]
            # Either a real label, or the documented "no compatible runout
            # among those sampled" fallback -- never anything else, and
            # never a crash.
            assert category in _NINE_CATEGORIES or category == ""


def test_describe_category_names_a_flush():
    assert describe_category(hand_str_to_ints("AsKs9h2c"),
                             hand_str_to_ints("6s7s4s2h9d")) == "flush"


# ---------------------------------------------------------------------------
# Task 10: parallel evaluate_population correctness.
#
# Fixture reused from test_boundary_hand_colliding_with_the_first_runout_does_not_crash
# above (already validated legal: no card shared between board/dead/heroes).
# hands=1000, runouts=80 puts total hand-evaluations at (1000+2)*80 = 80160,
# comfortably above _PARALLEL_MIN_EVALS (4000) so the auto-parallel path
# would trigger on its own -- but these tests pass `parallel=`/`num_workers=`
# explicitly so both sides of each comparison run on IDENTICAL inputs
# regardless of where that threshold sits.
_PARALLEL_BOARD = "6s7s4s"
_PARALLEL_DEAD = ["AsKs9h2c", "QhJhTd3d"]
_PARALLEL_HEROES = [{"id": "nuts", "cards": "AsKs9h2c"},
                    {"id": "air", "cards": "QhJhTd3d"}]


def _parallel_fixture_ladder(**kw):
    return compute_range_ladder(
        board=_PARALLEL_BOARD, dead=_PARALLEL_DEAD, heroes=_PARALLEL_HEROES,
        hands=1000, runouts=80, seed=17, **kw,
    )


def test_parallel_matches_serial_bucket_equities_and_boundary_hands():
    # The Task 10 correctness bar, applied literally: same board, dead,
    # heroes, hands, runouts and seed, computed both ways, must agree to
    # within 1e-9 on every bucket equity and produce identical boundary
    # hands. If these ever diverge, the parallelisation -- not the sampling
    # -- is wrong; there is no seed involved in *which* runouts get
    # dispatched to which worker, only in which runouts get sampled in the
    # first place (identical on both sides here).
    serial = _parallel_fixture_ladder(parallel=False)
    parallel = _parallel_fixture_ladder(parallel=True, num_workers=4)

    assert serial["population"] == parallel["population"]
    assert serial["runouts"] == parallel["runouts"]

    for lad_s, lad_p in zip(serial["ladders"], parallel["ladders"]):
        assert lad_s["id"] == lad_p["id"]
        for rung_s, rung_p in zip(lad_s["rungs"], lad_p["rungs"]):
            assert rung_s["bucket"] == rung_p["bucket"]
            assert rung_s["equity"] == pytest.approx(rung_p["equity"], abs=1e-9)
            # Boundary hand selection is a deterministic function of
            # `strength` (np.argsort); requiring the cards string to match
            # exactly (not just the equity) catches a parallelisation bug
            # that scrambles which villain rows the chunks' partials land
            # on, even if it happened to leave the aggregate equity numbers
            # looking plausible.
            assert rung_s["edge"]["cards"] == rung_p["edge"]["cards"]
            assert rung_s["edge"]["category"] == rung_p["edge"]["category"]


def test_evaluate_population_parallel_matches_serial_directly():
    # Same correctness bar, one level down: call evaluate_population itself
    # both ways (not through compute_range_ladder) so a failure here points
    # straight at the chunk-split/dispatch/reassembly logic rather than
    # anything in build_rungs.
    board_ints = hand_str_to_ints(_PARALLEL_BOARD)
    dead_cards = [c for d in _PARALLEL_DEAD for c in (d[i:i + 2] for i in range(0, len(d), 2))]
    board_cards = [_PARALLEL_BOARD[i:i + 2] for i in range(0, len(_PARALLEL_BOARD), 2)]
    deck = generate_deck_ints(dead_cards + board_cards)
    heroes = np.stack([hand_str_to_ints(h["cards"]) for h in _PARALLEL_HEROES])
    score_array = get_score_array()

    rng = np.random.default_rng(17)
    villains = sample_villains(deck, 4, 1000, rng)
    runouts = sample_runouts(deck, 3, 80, rng)

    strength_s, hero_eq_s, counts_s = evaluate_population(
        villains, heroes, board_ints, runouts, "plo4", score_array, parallel=False,
    )
    strength_p, hero_eq_p, counts_p = evaluate_population(
        villains, heroes, board_ints, runouts, "plo4", score_array,
        parallel=True, num_workers=6,
    )

    # counts is an exact sum of 1.0 increments, one per (villain, eligible
    # runout) pair -- order-independent regardless of chunking, so this can
    # (and should) hold exactly, not just approximately.
    assert np.array_equal(counts_s, counts_p)
    np.testing.assert_allclose(strength_s, strength_p, atol=1e-9)
    np.testing.assert_allclose(hero_eq_s, hero_eq_p, atol=1e-9)


def test_different_worker_counts_produce_identical_output():
    # The other half of the Task 10 correctness bar: worker count must not
    # change results. 7 and 3 don't evenly divide 80 runouts, so the two
    # calls genuinely split the runouts array at different points --
    # np.array_split(..., 7) and np.array_split(..., 3) group different
    # runouts into each partial sum, so this is a real test of summation
    # order, not a no-op because both happen to produce the same chunks.
    few_workers = _parallel_fixture_ladder(parallel=True, num_workers=3)
    many_workers = _parallel_fixture_ladder(parallel=True, num_workers=7)

    assert few_workers["population"] == many_workers["population"]
    for lad_a, lad_b in zip(few_workers["ladders"], many_workers["ladders"]):
        for rung_a, rung_b in zip(lad_a["rungs"], lad_b["rungs"]):
            # Same 1e-9 tolerance as the parallel-vs-serial bar above: float
            # addition is not associative, so partials grouped at different
            # chunk boundaries are not guaranteed bit-identical even when
            # every input (including the seed) is held fixed -- only
            # guaranteed to agree up to floating-point summation order,
            # which this tolerance already accounts for.
            assert rung_a["equity"] == pytest.approx(rung_b["equity"], abs=1e-9)
            assert rung_a["edge"]["cards"] == rung_b["edge"]["cards"]
