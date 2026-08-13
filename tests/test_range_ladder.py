from itertools import combinations

import numpy as np
from range_ladder import sample_villains

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
