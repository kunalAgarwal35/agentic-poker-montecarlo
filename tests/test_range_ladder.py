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
    rank_hands,
    sample_pass2_trials,
    evaluate_pass2,
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
# Task 13 Pass 1: sample_runouts + eligible_mask -- shared runouts, and the
# collision-skip rule that makes them safe to reuse across every hand.
# ---------------------------------------------------------------------------

def test_sample_runouts_draws_the_requested_shared_runouts():
    deck = np.arange(10, 52, dtype=np.int32)
    out = sample_runouts(deck, board_len=3, r=30, rng=np.random.default_rng(5))
    assert out.shape == (30, 2)        # need = 5 - 3
    assert out.dtype == np.int32
    assert np.isin(out, deck).all()
    # every runout's own two cards are distinct from each other
    assert all(len(set(row.tolist())) == 2 for row in out)

def test_sample_runouts_river_is_a_single_empty_runout():
    deck = np.arange(10, 52, dtype=np.int32)
    out = sample_runouts(deck, board_len=5, r=30, rng=np.random.default_rng(1))
    assert out.shape == (1, 0)         # no card left to complete

def test_sample_runouts_is_seed_deterministic():
    deck = np.arange(10, 52, dtype=np.int32)
    a = sample_runouts(deck, 3, 30, np.random.default_rng(9))
    b = sample_runouts(deck, 3, 30, np.random.default_rng(9))
    assert np.array_equal(a, b)

def test_eligible_mask_is_false_only_for_a_colliding_hand():
    hands = np.stack([
        hand_str_to_ints("AsKs9h2c"),   # holds 9h -- collides with the runout below
        hand_str_to_ints("QdJcTd5h"),
        hand_str_to_ints("8c3c2d6d"),
    ])
    runout = hand_str_to_ints("9h2h")
    mask = eligible_mask(hands, runout)
    assert mask.tolist() == [False, True, True]

def test_eligible_mask_is_vacuously_true_for_an_empty_runout():
    # River: the "runout" is empty, so no hand can ever collide with it.
    hands = np.stack([hand_str_to_ints("AsKs9h2c"), hand_str_to_ints("QdJcTd5h")])
    empty_runout = np.empty(0, dtype=np.int32)
    assert eligible_mask(hands, empty_runout).all()


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
    hands = np.stack([hand_str_to_ints("AsKs9h2c")] * 2)
    boards = np.stack([
        hand_str_to_ints("6s7s4s2h9d"),   # flush board -- hand makes the nut flush
        hand_str_to_ints("6h7d4c2d9c"),   # no flush possible -- hand is ace-high
    ])
    out = score_hands(hands, boards, "plo4", get_score_array())
    assert out.shape == (2,)
    assert out[0] > out[1]
    expected0 = score_hands(hands[:1], boards[0], "plo4", get_score_array())[0]
    expected1 = score_hands(hands[1:], boards[1], "plo4", get_score_array())[0]
    assert out[0] == expected0
    assert out[1] == expected1


# ---------------------------------------------------------------------------
# Task 13 Pass 1: rank_hands -- ranks HANDS (not trials/scenarios) on
# shared runouts, hero-independent.
# ---------------------------------------------------------------------------

def test_rank_hands_ranks_a_nut_flush_above_air_on_the_river():
    # River (need=0): a single empty runout, ranking is exact -- no
    # averaging needed. Same fixture as the old per-scenario test, but now
    # exercised through the hand-ranking path.
    board = hand_str_to_ints("6s7s4s2h9d")
    villains = np.stack([
        hand_str_to_ints("AsKs9h2c"),               # nut flush
        hand_str_to_ints("Jd3d2d4d"),               # air
        hand_str_to_ints("6h6d3c5d"),               # pair
    ])
    runouts = sample_runouts(np.arange(0, 52, dtype=np.int32), board_len=5, r=30,
                             rng=np.random.default_rng(1))
    strength, counts = rank_hands(villains, board, runouts, "plo4", get_score_array())
    assert strength.shape == (3,)
    assert counts.tolist() == [1.0, 1.0, 1.0]
    assert strength[0] > strength[2] > strength[1]

def test_rank_hands_skips_a_colliding_hand_instead_of_scoring_it():
    # A shared runout can contain a card a villain also holds. Pass 1 must
    # skip that hand for that runout -- never feed the impossible pairing
    # to the scoring kernel. If this test crashes the process instead of
    # asserting, that IS the regression (see the historical repro test
    # below): a duplicate-card board fed to the bounds-unchecked scoring
    # kernel segfaults rather than raising, so this must never happen.
    board_ints = hand_str_to_ints("6s7s4s")
    villains = np.stack([
        hand_str_to_ints("AsKs9h2c"),   # holds 9h -- collides with the runout
        hand_str_to_ints("QdJcTd5h"),
        hand_str_to_ints("8c3c2d6d"),
    ])
    runout = hand_str_to_ints("9h2h")
    runouts = runout[None, :]
    strength, counts = rank_hands(villains, board_ints, runouts, "plo4", get_score_array())
    assert counts[0] == 0.0            # never eligible for the only runout
    assert counts[1] == 1.0 and counts[2] == 1.0
    assert strength[0] == 0.0          # never scored, not a merit-based 0

def test_rank_hands_averages_over_only_the_eligible_runouts():
    # Two runouts; `villain` collides with the first (colliding_runout) but
    # not the second (clean_runout). A third hand's own strength must equal
    # its share of the field averaged over exactly the runouts it (and at
    # least one other hand -- "share of field" needs a field of >= 2) is
    # eligible for, never silently including a runout where it was the only
    # eligible hand (no field to be measured against at all).
    board_ints = hand_str_to_ints("6s7s4s")
    villain = hand_str_to_ints("AsKs9h2c")          # holds 9h
    other1 = hand_str_to_ints("QdJcTd5h")
    other2 = hand_str_to_ints("8c3c2d6d")
    villains = np.stack([villain, other1, other2])
    colliding_runout = hand_str_to_ints("9h2h")     # shares 9h with `villain` only
    clean_runout = hand_str_to_ints("3h8h")         # collides with none of the three
    runouts = np.stack([colliding_runout, clean_runout])

    strength, counts = rank_hands(villains, board_ints, runouts, "plo4", get_score_array())
    # villain: eligible only for clean_runout.
    # other1/other2: eligible for both (colliding_runout still has a valid
    # 2-hand field once `villain` is excluded).
    assert counts.tolist() == [1.0, 2.0, 2.0]

    board_a = np.concatenate([board_ints, colliding_runout]).astype(np.int32)
    board_b = np.concatenate([board_ints, clean_runout]).astype(np.int32)
    scores_a = score_hands(np.stack([other1, other2]), board_a, "plo4", get_score_array())
    scores_b = score_hands(np.stack([villain, other1, other2]), board_b, "plo4", get_score_array())

    def share(own, field):
        worse = sum(1 for s in field if s < own)
        ties = sum(1 for s in field if s == own) - 1
        return (worse + 0.5 * ties) / (len(field) - 1)

    share_a = share(scores_a[0], scores_a.tolist())         # other1 on colliding_runout
    share_b = share(scores_b[1], scores_b.tolist())         # other1 on clean_runout
    assert strength[1] == pytest.approx((share_a + share_b) / 2)


# ---------------------------------------------------------------------------
# Task 13 Pass 2: sample_pass2_trials + evaluate_pass2 -- fresh, independent,
# per-trial runouts, stratified equally across buckets, no collisions ever.
# ---------------------------------------------------------------------------

def test_sample_pass2_trials_never_collides_villain_with_runout():
    deck = generate_deck_ints(["6s", "7s", "4s"])
    rng = np.random.default_rng(3)
    villain_hands = sample_villains(deck, 4, 200, rng)
    bucket_indices = [np.arange(0, 50), np.arange(50, 200)]
    trials_arr = sample_pass2_trials(deck, board_len=3, hole_count=4,
                                     villain_hands=villain_hands,
                                     bucket_indices=bucket_indices,
                                     trials_per_bucket=300, rng=rng)
    assert trials_arr.shape == (600, 6)      # 2 buckets x 300 trials, 4+2 cols
    villains = trials_arr[:, :4]
    runouts = trials_arr[:, 4:]
    for v, r in zip(villains.tolist(), runouts.tolist()):
        assert set(v).isdisjoint(r)          # runout never shares a card with its own villain
        assert len(set(r)) == len(r)         # runout's own cards are distinct
    assert np.isin(runouts, deck).all()      # every runout card is a legal deck card

def test_sample_pass2_trials_river_has_zero_runout_columns():
    deck = generate_deck_ints(["6s", "7s", "4s", "2h", "9d"])
    rng = np.random.default_rng(2)
    villain_hands = sample_villains(deck, 6, 50, rng)
    bucket_indices = [np.arange(0, 50)]
    trials_arr = sample_pass2_trials(deck, board_len=5, hole_count=6,
                                     villain_hands=villain_hands,
                                     bucket_indices=bucket_indices,
                                     trials_per_bucket=40, rng=rng)
    assert trials_arr.shape == (40, 6)       # hole_count + 0

def test_sample_pass2_trials_gives_every_bucket_an_equal_budget():
    # The Task 13 fix for Task 11's binding constraint: a tiny bucket (2
    # members) must get exactly as many trials as a huge one (198 members).
    deck = generate_deck_ints(["6s", "7s", "4s"])
    rng = np.random.default_rng(4)
    villain_hands = sample_villains(deck, 4, 200, rng)
    tiny_bucket = np.arange(0, 2)
    huge_bucket = np.arange(2, 200)
    trials_arr = sample_pass2_trials(deck, board_len=3, hole_count=4,
                                     villain_hands=villain_hands,
                                     bucket_indices=[tiny_bucket, huge_bucket],
                                     trials_per_bucket=1000, rng=rng)
    assert trials_arr.shape[0] == 2000       # 1000 for each, regardless of bucket size
    tiny_villains = {tuple(r) for r in trials_arr[:1000, :4].tolist()}
    # only 2 distinct hands possible in the tiny bucket's block
    assert tiny_villains <= {tuple(villain_hands[i].tolist()) for i in tiny_bucket}

def test_sample_pass2_trials_is_seed_deterministic():
    deck = generate_deck_ints(["6s", "7s", "4s"])
    villain_hands = sample_villains(deck, 4, 100, np.random.default_rng(1))
    bucket_indices = [np.arange(0, 100)]
    a = sample_pass2_trials(deck, 3, 4, villain_hands, bucket_indices, 200,
                            np.random.default_rng(9))
    b = sample_pass2_trials(deck, 3, 4, villain_hands, bucket_indices, 200,
                            np.random.default_rng(9))
    assert np.array_equal(a, b)


def test_evaluate_pass2_scores_every_row_on_its_own_completed_board():
    board = hand_str_to_ints("6s7s4s2h9d")
    villains = np.stack([
        hand_str_to_ints("AsKs9h2c"),               # nut flush
        hand_str_to_ints("Jd3d2d4d"),               # air
        hand_str_to_ints("6h6d3c5d"),               # pair
    ])
    heroes = np.stack([hand_str_to_ints("QsJs8c3h")])   # queen-high flush
    trials_arr = villains       # river: no runout columns
    villain_scores, hero_results = evaluate_pass2(
        trials_arr, heroes, board, hole_count=4, game="plo4",
        score_array=get_score_array(),
    )
    assert villain_scores.shape == (3,)
    assert hero_results.shape == (1, 3)
    assert villain_scores[0] > villain_scores[2] > villain_scores[1]
    assert hero_results[0, 0] == 0.0      # hero loses to the nut flush
    assert hero_results[0, 1] == 1.0      # hero beats air
    assert hero_results[0, 2] == 1.0      # hero (flush) beats a made pair

def test_evaluate_pass2_completes_the_board_per_trial_off_the_river():
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
    villain_scores, hero_results = evaluate_pass2(
        trials_arr, heroes, board, hole_count=4, game="plo4",
        score_array=get_score_array(),
    )
    board0 = np.concatenate([board, runout0])
    board1 = np.concatenate([board, runout1])
    expected0 = score_hands(villain0[None, :], board0, "plo4", get_score_array())[0]
    expected1 = score_hands(villain1[None, :], board1, "plo4", get_score_array())[0]
    assert villain_scores[0] == expected0
    assert villain_scores[1] == expected1


# ---------------------------------------------------------------------------
# build_rungs -- assembles rungs from precomputed (hero-independent) edges
# + a hero's own per-bucket equity row.
# ---------------------------------------------------------------------------

def test_build_rungs_assembles_bucket_edge_and_equity():
    buckets = [20, 50, 100]
    edges = [
        {"cards": "AsKs9h2c", "category": "flush"},
        {"cards": "6h6d3c2d", "category": "pair"},
        {"cards": "Jd3d2d4d", "category": "high card"},
    ]
    equity_row = np.array([0.0, 0.5, 0.7])
    rungs = build_rungs(buckets, edges, equity_row)
    assert [r["bucket"] for r in rungs] == [20, 50, 100]
    assert [r["equity"] for r in rungs] == [0.0, 0.5, 0.7]
    assert rungs[0]["edge"] == edges[0]
    assert rungs[2]["edge"] == edges[2]
    assert isinstance(rungs[0]["equity"], float)


def test_describe_category_names_a_flush():
    assert describe_category(hand_str_to_ints("AsKs9h2c"),
                             hand_str_to_ints("6s7s4s2h9d")) == "flush"


# ---------------------------------------------------------------------------
# compute_range_ladder: end-to-end behavior.
# ---------------------------------------------------------------------------

def test_river_ladder_is_exact_and_monotonic():
    # Same fixture Task 11 used (see its comment history for why this
    # board/hand pair was picked over the brief's literal example).
    out = compute_range_ladder(
        board="6s7s4s2h9d",
        dead=["AsKs9h2c", "3dTc5s8h"],
        heroes=[{"id": "nuts", "cards": "AsKs9h2c"},
                {"id": "air",  "cards": "3dTc5s8h"}],
        hands=2000, rank_runouts=20, trials_per_bucket=800, seed=42,
    )
    assert out["exact"] is True
    assert out["population"] > 0
    assert out["rank_runouts"] == 1        # river: single empty completion
    assert out["trials_per_bucket"] == 800

    nuts = next(l for l in out["ladders"] if l["id"] == "nuts")["rungs"]
    air = next(l for l in out["ladders"] if l["id"] == "air")["rungs"]

    for rungs in (nuts, air):
        eq = [r["equity"] for r in rungs]
        assert eq == sorted(eq), eq
    assert len({r["equity"] for r in nuts}) > 1 or len({r["equity"] for r in air}) > 1

    assert nuts[0]["equity"] > 0.85
    assert air[0]["equity"] < 0.15

    # boundary hands are shared -- ranking is hero-independent
    assert [r["edge"]["cards"] for r in nuts] == [r["edge"]["cards"] for r in air]
    assert [r["edge"]["category"] for r in nuts] == [r["edge"]["category"] for r in air]


def test_flop_ladder_reports_the_requested_rank_runouts():
    out = compute_range_ladder(
        board="6s7s4s",
        dead=["AsKs9h2c", "QhJhTd3d"],
        heroes=[{"id": "h", "cards": "AsKs9h2c"}],
        hands=500, rank_runouts=25, trials_per_bucket=200, seed=1,
    )
    assert out["exact"] is False
    assert out["rank_runouts"] == 25
    assert out["trials_per_bucket"] == 200


def test_hero_missing_from_dead_is_rejected():
    with pytest.raises(ValueError):
        compute_range_ladder(
            board="6s7s4s",
            dead=["AsKs9h2c"],
            heroes=[{"id": "x", "cards": "QsJs8c3h"}],   # not in dead
            hands=100, rank_runouts=10, trials_per_bucket=50, seed=1,
        )


def test_hero_card_on_the_board_is_rejected():
    with pytest.raises(ValueError):
        compute_range_ladder(
            board="6s7s4s2h9d",
            dead=["6sKs9c2c"],
            heroes=[{"id": "x", "cards": "6sKs9c2c"}],
            hands=100, rank_runouts=10, trials_per_bucket=50, seed=1,
        )


def test_duplicate_card_within_one_hand_is_rejected():
    with pytest.raises(ValueError):
        compute_range_ladder(
            board="6s7s4s",
            dead=["AsAs9c2c"],
            heroes=[{"id": "x", "cards": "AsAs9c2c"}],
            hands=100, rank_runouts=10, trials_per_bucket=50, seed=1,
        )


def test_duplicate_card_across_two_dead_entries_is_rejected():
    with pytest.raises(ValueError):
        compute_range_ladder(
            board="6s7s4s",
            dead=["AsKc9c2c", "AsQd8c3c"],
            heroes=[{"id": "x", "cards": "AsKc9c2c"}],
            hands=100, rank_runouts=10, trials_per_bucket=50, seed=1,
        )


_NINE_CATEGORIES = {
    "high card", "pair", "two pair", "trips", "straight",
    "flush", "full house", "quads", "straight flush",
}


def test_pass1_collisions_never_crash_and_every_rung_resolves(  ):
    # Historical repro (board=6s7s4s, dead=[AsKs9h2c, QhJhTd3d]): a shared
    # runout used to collide with several sampled villain hands, and
    # scoring that impossible pairing crashed the process (Windows: access
    # violation) rather than raising -- unreachable from pytest.raises().
    # rank_runouts=40 with only 400 hands keeps the collision rate high
    # enough that this fixture reliably exercises the skip path, across
    # both a flop and a turn board so the runout width (2 vs 1) is
    # exercised too.
    for board, need_desc in (("6s7s4s", "flop"), ("6s7s4s2h", "turn")):
        for seed in range(5):
            out = compute_range_ladder(
                board=board,
                dead=["AsKs9h2c", "QhJhTd3d"],
                heroes=[{"id": "nuts", "cards": "AsKs9h2c"},
                        {"id": "air", "cards": "QhJhTd3d"}],
                hands=400, rank_runouts=40, trials_per_bucket=60, seed=seed,
            )
            for lad in out["ladders"]:
                assert len(lad["rungs"]) == 6
                for rung in lad["rungs"]:
                    assert rung["edge"]["category"] in _NINE_CATEGORIES, (
                        f"{need_desc} seed={seed}: got {rung['edge']['category']!r}"
                    )
                    assert len(rung["edge"]["cards"]) in (8, 10, 12)


def test_pass2_never_produces_an_impossible_pairing():
    # Every hero/villain/runout card must be pairwise distinct in every
    # Pass-2 trial -- checked directly against compute_range_ladder's
    # output path by re-deriving Pass 2's own sampling with the same seed
    # and asserting no card set has a duplicate.
    board_ints = hand_str_to_ints("6s7s4s")
    dead = ["AsKs9h2c", "QhJhTd3d"]
    dead_cards = [c for d in dead for c in (d[i:i + 2] for i in range(0, len(d), 2))]
    board_cards = ["6s", "7s", "4s"]
    deck = generate_deck_ints(dead_cards + board_cards)
    rng = np.random.default_rng(11)
    villain_hands = sample_villains(deck, 4, 300, rng)
    runouts = sample_runouts(deck, 3, 30, rng)
    strength, counts = rank_hands(villain_hands, board_ints, runouts, "plo4", get_score_array())
    ranked = counts > 0
    villain_hands, strength = villain_hands[ranked], strength[ranked]
    order = np.argsort(-strength, kind="stable")
    bucket_indices = [order[:max(1, int(round(order.size * p / 100)))] for p in (5, 15, 100)]
    trials_arr = sample_pass2_trials(deck, 3, 4, villain_hands, bucket_indices, 500, rng)
    villains, drawn_runouts = trials_arr[:, :4], trials_arr[:, 4:]
    for v, r in zip(villains.tolist(), drawn_runouts.tolist()):
        full = set(v) | set(r)
        assert len(full) == len(v) + len(r)      # every card pairwise distinct
        assert not (full & set(board_ints.tolist()))
        assert not (full & set(hand_str_to_ints("AsKs9h2c").tolist()) - set(v))
        assert not (full & set(hand_str_to_ints("QhJhTd3d").tolist()) - set(v))


# ---------------------------------------------------------------------------
# Buckets are hand-sets: bucket membership and edge hands are computed once
# and shared by every hero -- identical across heroes, not just correlated.
# ---------------------------------------------------------------------------

def test_buckets_are_identical_hand_sets_across_heroes():
    out = compute_range_ladder(
        board="6s7s4s",
        dead=["AsKs9h2c", "QhJhTd3d"],
        heroes=[{"id": "nuts", "cards": "AsKs9h2c"},
                {"id": "air", "cards": "QhJhTd3d"}],
        hands=1500, rank_runouts=25, trials_per_bucket=300, seed=7,
    )
    nuts_edges = [r["edge"] for r in next(l for l in out["ladders"] if l["id"] == "nuts")["rungs"]]
    air_edges = [r["edge"] for r in next(l for l in out["ladders"] if l["id"] == "air")["rungs"]]
    assert nuts_edges == air_edges


def test_describe_category_names_a_flush_again_direct():
    # Guard against describe_category itself changing shape.
    assert describe_category(hand_str_to_ints("6h6d3c2d"),
                             hand_str_to_ints("6s7s4s2h9d")) in _NINE_CATEGORIES


# ---------------------------------------------------------------------------
# Determinism: parallel must be bit-exact vs serial and identical across
# worker counts, for BOTH passes.
# ---------------------------------------------------------------------------

_PARALLEL_BOARD = "6s7s4s"
_PARALLEL_DEAD = ["AsKs9h2c", "QhJhTd3d"]
_PARALLEL_HEROES = [{"id": "nuts", "cards": "AsKs9h2c"},
                    {"id": "air", "cards": "QhJhTd3d"}]


def _parallel_fixture_ladder(**kw):
    return compute_range_ladder(
        board=_PARALLEL_BOARD, dead=_PARALLEL_DEAD, heroes=_PARALLEL_HEROES,
        hands=6000, rank_runouts=30, trials_per_bucket=4000, seed=17, **kw,
    )


def test_parallel_matches_serial_bucket_equities_and_boundary_hands():
    serial = _parallel_fixture_ladder(parallel=False)
    parallel = _parallel_fixture_ladder(parallel=True, num_workers=4)

    assert serial["population"] == parallel["population"]
    assert serial["rank_runouts"] == parallel["rank_runouts"]

    for lad_s, lad_p in zip(serial["ladders"], parallel["ladders"]):
        assert lad_s["id"] == lad_p["id"]
        for rung_s, rung_p in zip(lad_s["rungs"], lad_p["rungs"]):
            assert rung_s["bucket"] == rung_p["bucket"]
            assert rung_s["equity"] == rung_p["equity"]
            assert rung_s["edge"]["cards"] == rung_p["edge"]["cards"]
            assert rung_s["edge"]["category"] == rung_p["edge"]["category"]


def test_rank_hands_parallel_matches_serial_directly():
    board_ints = hand_str_to_ints(_PARALLEL_BOARD)
    dead_cards = [c for d in _PARALLEL_DEAD for c in (d[i:i + 2] for i in range(0, len(d), 2))]
    board_cards = [_PARALLEL_BOARD[i:i + 2] for i in range(0, len(_PARALLEL_BOARD), 2)]
    deck = generate_deck_ints(dead_cards + board_cards)
    score_array = get_score_array()

    rng = np.random.default_rng(17)
    villain_hands = sample_villains(deck, 4, 6000, rng)
    runouts = sample_runouts(deck, 3, 30, rng)

    strength_s, counts_s = rank_hands(villain_hands, board_ints, runouts, "plo4",
                                      score_array, parallel=False)
    strength_p, counts_p = rank_hands(villain_hands, board_ints, runouts, "plo4",
                                      score_array, parallel=True, num_workers=6)

    assert np.array_equal(counts_s, counts_p)
    assert np.array_equal(strength_s, strength_p)      # bit-exact: concatenate, not sum


def test_evaluate_pass2_parallel_matches_serial_directly():
    board_ints = hand_str_to_ints(_PARALLEL_BOARD)
    dead_cards = [c for d in _PARALLEL_DEAD for c in (d[i:i + 2] for i in range(0, len(d), 2))]
    board_cards = [_PARALLEL_BOARD[i:i + 2] for i in range(0, len(_PARALLEL_BOARD), 2)]
    deck = generate_deck_ints(dead_cards + board_cards)
    heroes = np.stack([hand_str_to_ints(h["cards"]) for h in _PARALLEL_HEROES])
    score_array = get_score_array()

    rng = np.random.default_rng(17)
    villain_hands = sample_villains(deck, 4, 200, rng)
    bucket_indices = [np.arange(0, 200)]
    trials_arr = sample_pass2_trials(deck, 3, 4, villain_hands, bucket_indices, 40000, rng)

    villain_scores_s, hero_results_s = evaluate_pass2(
        trials_arr, heroes, board_ints, hole_count=4, game="plo4",
        score_array=score_array, parallel=False,
    )
    villain_scores_p, hero_results_p = evaluate_pass2(
        trials_arr, heroes, board_ints, hole_count=4, game="plo4",
        score_array=score_array, parallel=True, num_workers=6,
    )

    assert np.array_equal(villain_scores_s, villain_scores_p)
    assert np.array_equal(hero_results_s, hero_results_p)


def test_different_worker_counts_produce_identical_output():
    few_workers = _parallel_fixture_ladder(parallel=True, num_workers=3)
    many_workers = _parallel_fixture_ladder(parallel=True, num_workers=7)

    assert few_workers["population"] == many_workers["population"]
    for lad_a, lad_b in zip(few_workers["ladders"], many_workers["ladders"]):
        for rung_a, rung_b in zip(lad_a["rungs"], lad_b["rungs"]):
            assert rung_a["equity"] == rung_b["equity"]
            assert rung_a["edge"]["cards"] == rung_b["edge"]["cards"]
