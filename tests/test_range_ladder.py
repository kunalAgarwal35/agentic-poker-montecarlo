from itertools import combinations

import numpy as np
from card_encoding import hand_str_to_ints, generate_deck_ints
from hand_rank_evaluator import get_score_array
from range_ladder import (
    sample_villains,
    sample_runouts,
    eligible_mask,
    score_hands,
    evaluate_population,
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
    strength, hero_equity = evaluate_population(
        villains, heroes, board, runouts, "plo4", get_score_array()
    )
    assert strength[0] > strength[2] > strength[1]
    assert hero_equity.shape == (1, 3)
    assert hero_equity[0, 0] == 0.0      # loses to the nut flush
    assert hero_equity[0, 1] == 1.0      # beats air

def test_a_villain_holding_a_runout_card_is_skipped_not_scored():
    # Board 6s7s4s; the only runout is 2h9d. Villain 0 holds 2h, so the pairing
    # is impossible -- it must be skipped, not counted as a loss.
    board = hand_str_to_ints("6s7s4s")
    villains = np.stack([
        hand_str_to_ints("As2hKd3c"),
        hand_str_to_ints("AhKh9c8c"),
    ])
    heroes = np.stack([hand_str_to_ints("QsJs8d3d")])
    runouts = np.stack([hand_str_to_ints("2h9d")])
    strength, hero_equity = evaluate_population(
        villains, heroes, board, runouts, "plo4", get_score_array()
    )
    assert strength[0] == 0.0            # never eligible -> no data
    assert hero_equity[0, 0] == 0.0
