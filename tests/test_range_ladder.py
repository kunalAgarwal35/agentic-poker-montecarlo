from itertools import combinations

import numpy as np
import pytest
from card_encoding import hand_str_to_ints, generate_deck_ints
from hand_rank_evaluator import get_score_array
from range_ladder import (
    sample_villains,
    sample_trials,
    score_hands,
    evaluate_trials,
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


# ---------------------------------------------------------------------------
# Task 11: sample_trials -- joint villain-hand + board-completion draws.
# ---------------------------------------------------------------------------

def test_sample_trials_draws_hole_count_plus_need_distinct_cards():
    # Flop: need = 5 - 3 = 2, hole_count = 4 (PLO4) -> 6 distinct cards/row.
    deck = np.arange(10, 52, dtype=np.int32)
    out = sample_trials(deck, hole_count=4, board_len=3, trials=500,
                        rng=np.random.default_rng(5))
    assert out.shape == (500, 6)
    assert out.dtype == np.int32
    assert np.isin(out, deck).all()
    # every card in a row is distinct -- this is the collision-impossible
    # guarantee the whole Task 11 redesign rests on.
    assert all(len(set(row.tolist())) == 6 for row in out)

def test_sample_trials_rows_are_not_deduplicated_across_trials():
    # Unlike sample_villains, repeated trial rows are expected: these are
    # `trials` independent Monte Carlo draws, not a fixed distinct
    # population. A tiny deck forces the birthday-paradox collision that
    # would prove a (wrongly) deduplicating implementation is NOT in use --
    # C(6,4) x [runout choices] leaves few enough combinations that 2000
    # independent trials will almost certainly repeat one.
    deck = np.arange(0, 6, dtype=np.int32)
    out = sample_trials(deck, hole_count=4, board_len=5, trials=2000,
                        rng=np.random.default_rng(1))     # river: need=0
    assert out.shape == (2000, 4)
    assert len({tuple(r) for r in out.tolist()}) < 2000

def test_sample_trials_river_has_zero_runout_columns():
    deck = np.arange(10, 52, dtype=np.int32)
    out = sample_trials(deck, hole_count=6, board_len=5, trials=50,
                        rng=np.random.default_rng(2))
    assert out.shape == (50, 6)      # hole_count + need, need = 0

def test_sample_trials_is_seed_deterministic():
    deck = np.arange(10, 52, dtype=np.int32)
    a = sample_trials(deck, 5, 3, 300, np.random.default_rng(9))
    b = sample_trials(deck, 5, 3, 300, np.random.default_rng(9))
    assert np.array_equal(a, b)

def test_sample_trials_too_small_a_deck_returns_empty():
    deck = np.arange(0, 5, dtype=np.int32)     # only 5 cards, need 4+2=6
    out = sample_trials(deck, hole_count=4, board_len=3, trials=10,
                        rng=np.random.default_rng(1))
    assert out.shape == (0, 6)


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

def test_score_hands_accepts_a_per_hand_board():
    # Task 11: evaluate_trials needs one board PER HAND (every trial
    # completes the board differently), not one board broadcast to every
    # hand. Two hands, two different boards -- same hand string ("AsKs9h2c")
    # is a flush on board 0 and just ace-high on board 1 (no flush, no
    # pair), so a shared-board bug (ignoring board index 1 and always using
    # board 0, or vice versa) would produce the WRONG relative ordering.
    hands = np.stack([hand_str_to_ints("AsKs9h2c")] * 2)
    boards = np.stack([
        hand_str_to_ints("6s7s4s2h9d"),   # flush board -- hand makes the nut flush
        hand_str_to_ints("6h7d4c2d9c"),   # no flush possible -- hand is ace-high
    ])
    out = score_hands(hands, boards, "plo4", get_score_array())
    assert out.shape == (2,)
    assert out[0] > out[1]
    # cross-check against the single-board path called once per row
    expected0 = score_hands(hands[:1], boards[0], "plo4", get_score_array())[0]
    expected1 = score_hands(hands[1:], boards[1], "plo4", get_score_array())[0]
    assert out[0] == expected0
    assert out[1] == expected1


# ---------------------------------------------------------------------------
# Task 11: evaluate_trials -- the single joint-sampling pass.
# ---------------------------------------------------------------------------

def test_evaluate_trials_scores_every_row_on_its_own_completed_board():
    # River (need=0): trials are just villain hands, board is fixed and
    # exact for all of them -- this is the same case Task 4's
    # test_strength_orders_a_made_flush_above_air_on_the_river exercised
    # against evaluate_population.
    board = hand_str_to_ints("6s7s4s2h9d")
    villains = np.stack([
        hand_str_to_ints("AsKs9h2c"),               # nut flush
        hand_str_to_ints("Jd3d2d4d"),               # air
        hand_str_to_ints("6h6d3c5d"),               # pair
    ])
    heroes = np.stack([hand_str_to_ints("QsJs8c3h")])   # queen-high flush
    trials_arr = villains     # need=0 -> trials_arr is just the villain hands
    villain_scores, hero_results = evaluate_trials(
        trials_arr, heroes, board, hole_count=4, game="plo4",
        score_array=get_score_array(),
    )
    assert villain_scores.shape == (3,)
    assert hero_results.shape == (1, 3)
    assert villain_scores[0] > villain_scores[2] > villain_scores[1]
    assert hero_results[0, 0] == 0.0      # hero loses to the nut flush
    assert hero_results[0, 1] == 1.0      # hero beats air
    assert hero_results[0, 2] == 1.0      # hero (flush) beats a made pair

def test_evaluate_trials_completes_the_board_per_trial_off_the_river():
    # Flop board 6s7s4s; two trials, each supplying its OWN 2-card runout
    # (positional: first hole_count cols = villain hand, rest = runout).
    # Trial 0's runout pairs the board (6s7s4s + 9d2h has no extra pairing,
    # so use 6h6d style board completion instead): choose runouts that make
    # each trial's board unambiguous to hand-verify.
    board = hand_str_to_ints("6s7s4s")
    villain0 = hand_str_to_ints("AsKs9h2c")
    villain1 = hand_str_to_ints("QdJcTd5h")
    runout0 = hand_str_to_ints("9d2h")
    runout1 = hand_str_to_ints("8c3c")
    trials_arr = np.stack([
        np.concatenate([villain0, runout0]),
        np.concatenate([villain1, runout1]),
    ])
    heroes = np.stack([hand_str_to_ints("QsJs8d3d")])
    villain_scores, hero_results = evaluate_trials(
        trials_arr, heroes, board, hole_count=4, game="plo4",
        score_array=get_score_array(),
    )
    # Each trial's villain hand must be scored on ITS OWN completed board,
    # not a board shared across trials -- verify directly against score_hands
    # called once per trial with that trial's exact 5-card board.
    board0 = np.concatenate([board, runout0])
    board1 = np.concatenate([board, runout1])
    expected0 = score_hands(villain0[None, :], board0, "plo4", get_score_array())[0]
    expected1 = score_hands(villain1[None, :], board1, "plo4", get_score_array())[0]
    assert villain_scores[0] == expected0
    assert villain_scores[1] == expected1

def test_evaluate_trials_river_and_non_river_both_reachable_via_sample_trials():
    # End-to-end sanity: sample_trials -> evaluate_trials must not crash and
    # must produce shapes consistent with `trials` and the number of heroes,
    # on both the river (need=0) and the flop (need=2).
    deck = generate_deck_ints(["6s", "7s", "4s", "As", "Ks", "9h", "2c"])
    heroes = np.stack([hand_str_to_ints("AsKs9h2c")])
    rng = np.random.default_rng(3)
    trials_arr = sample_trials(deck, hole_count=4, board_len=3, trials=1000, rng=rng)
    villain_scores, hero_results = evaluate_trials(
        trials_arr, heroes, hand_str_to_ints("6s7s4s"), hole_count=4,
        game="plo4", score_array=get_score_array(),
    )
    assert villain_scores.shape == (1000,)
    assert hero_results.shape == (1, 1000)
    assert set(np.unique(hero_results)) <= {0.0, 0.5, 1.0}


def test_rungs_slice_by_strength_and_report_the_weakest_hand_in_each():
    # 10 trials with villain scores 0.0 .. 0.9 (river: runouts column is
    # empty); hero beats exactly the weak half, so index i (score 0.1*i)
    # has hero_result 1.0 for i < 5 and 0.0 above.
    villain_hands = np.stack([hand_str_to_ints("AsKs9h2c")] * 10)
    villain_scores = np.linspace(0.0, 0.9, 10)
    hero_results_row = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float64)
    runouts = np.empty((10, 0), dtype=np.int32)

    rungs = build_rungs(villain_scores, hero_results_row, villain_hands, runouts,
                        [20, 50, 100], hand_str_to_ints("6s7s4s2h9d"))

    assert [r["bucket"] for r in rungs] == [20, 50, 100]
    # top 20% = the 2 strongest, which hero loses to
    assert rungs[0]["equity"] == 0.0
    # top 100% = everyone; hero beats 5 of 10
    assert rungs[2]["equity"] == 0.5
    # equity never decreases as the bucket widens
    eq = [r["equity"] for r in rungs]
    assert eq == sorted(eq)

def test_every_bucket_holds_at_least_one_hand():
    villain_hands = np.stack([hand_str_to_ints("AsKs9h2c")] * 3)
    villain_scores = np.array([0.1, 0.5, 0.9])
    hero_results_row = np.array([1.0, 1.0, 0.0])
    runouts = np.empty((3, 0), dtype=np.int32)
    rungs = build_rungs(villain_scores, hero_results_row, villain_hands, runouts,
                        [5, 100], hand_str_to_ints("6s7s4s2h9d"))
    assert rungs[0]["equity"] == 0.0        # top 5% rounds up to 1 hand
    assert rungs[0]["edge"]["cards"] != ""

def test_build_rungs_labels_the_edge_hand_on_its_own_trials_runout():
    # Off the river, the edge hand's category must come from THAT TRIAL's
    # own completed board (board + that trial's runout), not the shared
    # flop board alone. Self-verifying: compares build_rungs' reported
    # category against describe_category computed directly on the trial's
    # real completed board (board + its own runout), so this catches a
    # regression that labels the edge hand on the bare flop (or any other
    # board) instead, regardless of what that hand's true category is.
    board = hand_str_to_ints("6s7s4s")
    villain_hands = np.stack([hand_str_to_ints("QdJcTd5h")])
    runouts = np.stack([hand_str_to_ints("8c9c")])
    villain_scores = np.array([1.0])
    hero_results_row = np.array([0.0])

    rungs = build_rungs(villain_scores, hero_results_row, villain_hands, runouts,
                        [100], board)
    expected_board = np.concatenate([board, runouts[0]])
    expected_category = describe_category(villain_hands[0], expected_board)
    assert rungs[0]["edge"]["category"] == expected_category
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
    # villain and equally beaten by every straight flush.
    #
    # The fixture below was found empirically (by calling
    # compute_range_ladder directly with several board/hand candidates and
    # printing the resulting ladders) to satisfy all of: no card shared
    # between board/nuts/air; nuts far exceeds the 0.85 floor at bucket 5;
    # air stays under the 0.15 ceiling at bucket 5; and at least one ladder
    # is strictly non-constant across the six buckets so `eq == sorted(eq)`
    # has more than a flat or two-point line to validate.
    out = compute_range_ladder(
        board="6s7s4s2h9d",
        dead=["AsKs9h2c", "3dTc5s8h"],
        heroes=[{"id": "nuts", "cards": "AsKs9h2c"},
                {"id": "air",  "cards": "3dTc5s8h"}],
        trials=2000, seed=42,
    )
    assert out["exact"] is True
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
            trials=100, seed=1,
        )


def test_hero_card_on_the_board_is_rejected():
    # 6s is both on the board and in the hero's (dead) hand -- a card
    # cannot be simultaneously dead and live on the board.
    with pytest.raises(ValueError):
        compute_range_ladder(
            board="6s7s4s2h9d",
            dead=["6sKs9c2c"],
            heroes=[{"id": "x", "cards": "6sKs9c2c"}],
            trials=100, seed=1,
        )


def test_duplicate_card_within_one_hand_is_rejected():
    # "As" appears twice inside a single dead entry.
    with pytest.raises(ValueError):
        compute_range_ladder(
            board="6s7s4s",
            dead=["AsAs9c2c"],
            heroes=[{"id": "x", "cards": "AsAs9c2c"}],
            trials=100, seed=1,
        )


def test_duplicate_card_across_two_dead_entries_is_rejected():
    # "As" appears once in each of two different dead entries -- two
    # players can never legally hold the same card.
    with pytest.raises(ValueError):
        compute_range_ladder(
            board="6s7s4s",
            dead=["AsKc9c2c", "AsQd8c3c"],
            heroes=[{"id": "x", "cards": "AsKc9c2c"}],
            trials=100, seed=1,
        )


_NINE_CATEGORIES = {
    "high card", "pair", "two pair", "trips", "straight",
    "flush", "full house", "quads", "straight flush",
}


def test_no_collision_is_ever_possible_by_construction():
    # Task 11's headline correctness bar: within any trial row every card is
    # distinct, and no card in a row appears on the board or in any hero
    # hand. This is the property that makes eligible_mask (and the
    # duplicate-card crash class it existed to avoid) structurally
    # unnecessary -- checked directly against compute_range_ladder's output
    # path (every rung must resolve to a real, non-empty category, on
    # every bucket, for every hero), across several seeds and both a flop
    # and a turn board so the runout width (2 vs 1) is exercised.
    for board, need_desc in (("6s7s4s", "flop"), ("6s7s4s2h", "turn")):
        for seed in range(5):
            out = compute_range_ladder(
                board=board,
                dead=["AsKs9h2c", "QhJhTd3d"],
                heroes=[{"id": "nuts", "cards": "AsKs9h2c"},
                        {"id": "air", "cards": "QhJhTd3d"}],
                trials=400, seed=seed,
            )
            for lad in out["ladders"]:
                assert len(lad["rungs"]) == 6
                for rung in lad["rungs"]:
                    assert rung["edge"]["category"] in _NINE_CATEGORIES, (
                        f"{need_desc} seed={seed}: got {rung['edge']['category']!r}"
                    )
                    assert len(rung["edge"]["cards"]) in (8, 10, 12)


def test_describe_category_names_a_flush():
    assert describe_category(hand_str_to_ints("AsKs9h2c"),
                             hand_str_to_ints("6s7s4s2h9d")) == "flush"


# ---------------------------------------------------------------------------
# Task 10/11: parallel evaluate_trials correctness.
#
# Fixture reused from test_no_collision_is_ever_possible_by_construction
# above (already validated legal: no card shared between board/dead/heroes).
# trials=80000 with hole_count=4, 2 heroes puts total hand-evaluations at
# (1+2)*80000 = 240000, comfortably above _PARALLEL_MIN_EVALS (4000) so the
# auto-parallel path would trigger on its own -- but these tests pass
# `parallel=`/`num_workers=` explicitly so both sides of each comparison run
# on IDENTICAL inputs regardless of where that threshold sits.
_PARALLEL_BOARD = "6s7s4s"
_PARALLEL_DEAD = ["AsKs9h2c", "QhJhTd3d"]
_PARALLEL_HEROES = [{"id": "nuts", "cards": "AsKs9h2c"},
                    {"id": "air", "cards": "QhJhTd3d"}]


def _parallel_fixture_ladder(**kw):
    return compute_range_ladder(
        board=_PARALLEL_BOARD, dead=_PARALLEL_DEAD, heroes=_PARALLEL_HEROES,
        trials=80000, seed=17, **kw,
    )


def test_parallel_matches_serial_bucket_equities_and_boundary_hands():
    # The Task 10 correctness bar, applied to Task 11's design: same board,
    # dead, heroes, trials and seed, computed both ways, must agree
    # EXACTLY (not just to a tolerance) on every bucket equity and produce
    # identical boundary hands. Unlike the predecessor design (which
    # additively summed beat_sum/beat_cnt/hero_sum across runout chunks, so
    # float non-associativity meant only approximate agreement was
    # guaranteed), joint trial sampling never sums partial results across
    # chunks -- it only concatenates independently-computed per-trial rows
    # -- so parallel and serial output is bit-identical by construction.
    serial = _parallel_fixture_ladder(parallel=False)
    parallel = _parallel_fixture_ladder(parallel=True, num_workers=4)

    assert serial["population"] == parallel["population"]

    for lad_s, lad_p in zip(serial["ladders"], parallel["ladders"]):
        assert lad_s["id"] == lad_p["id"]
        for rung_s, rung_p in zip(lad_s["rungs"], lad_p["rungs"]):
            assert rung_s["bucket"] == rung_p["bucket"]
            assert rung_s["equity"] == rung_p["equity"]
            # Boundary hand selection is a deterministic function of
            # `villain_scores` (np.argsort); requiring the cards string to
            # match exactly (not just the equity) catches a parallelisation
            # bug that scrambles which trial rows the chunks' partials land
            # on, even if it happened to leave the aggregate equity numbers
            # looking plausible.
            assert rung_s["edge"]["cards"] == rung_p["edge"]["cards"]
            assert rung_s["edge"]["category"] == rung_p["edge"]["category"]


def test_evaluate_trials_parallel_matches_serial_directly():
    # Same correctness bar, one level down: call evaluate_trials itself both
    # ways (not through compute_range_ladder) so a failure here points
    # straight at the chunk-split/dispatch/reassembly logic rather than
    # anything in build_rungs.
    board_ints = hand_str_to_ints(_PARALLEL_BOARD)
    dead_cards = [c for d in _PARALLEL_DEAD for c in (d[i:i + 2] for i in range(0, len(d), 2))]
    board_cards = [_PARALLEL_BOARD[i:i + 2] for i in range(0, len(_PARALLEL_BOARD), 2)]
    deck = generate_deck_ints(dead_cards + board_cards)
    heroes = np.stack([hand_str_to_ints(h["cards"]) for h in _PARALLEL_HEROES])
    score_array = get_score_array()

    rng = np.random.default_rng(17)
    trials_arr = sample_trials(deck, hole_count=4, board_len=3, trials=80000, rng=rng)

    villain_scores_s, hero_results_s = evaluate_trials(
        trials_arr, heroes, board_ints, hole_count=4, game="plo4",
        score_array=score_array, parallel=False,
    )
    villain_scores_p, hero_results_p = evaluate_trials(
        trials_arr, heroes, board_ints, hole_count=4, game="plo4",
        score_array=score_array, parallel=True, num_workers=6,
    )

    # Bit-exact: no summation crosses a chunk boundary, only concatenation
    # of independently-computed rows.
    assert np.array_equal(villain_scores_s, villain_scores_p)
    assert np.array_equal(hero_results_s, hero_results_p)


def test_different_worker_counts_produce_identical_output():
    # The other half of the Task 10/11 correctness bar: worker count must
    # not change results. 7 and 3 don't evenly divide 80000 trials, so the
    # two calls genuinely split the trials array at different points --
    # np.array_split(..., 7) and np.array_split(..., 3) group different
    # trials into each chunk -- but since chunks are only concatenated
    # (never summed), the reassembled per-trial numbers are identical
    # regardless of where the splits fell.
    few_workers = _parallel_fixture_ladder(parallel=True, num_workers=3)
    many_workers = _parallel_fixture_ladder(parallel=True, num_workers=7)

    assert few_workers["population"] == many_workers["population"]
    for lad_a, lad_b in zip(few_workers["ladders"], many_workers["ladders"]):
        for rung_a, rung_b in zip(lad_a["rungs"], lad_b["rungs"]):
            assert rung_a["equity"] == rung_b["equity"]
            assert rung_a["edge"]["cards"] == rung_b["edge"]["cards"]
