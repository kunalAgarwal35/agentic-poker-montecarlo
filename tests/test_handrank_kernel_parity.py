"""Bit-parity between the Numba and NumPy five-card scoring kernels.

Collected by pytest, unlike test_handrank_numba_parity.py, which is a manual
`main()` script with no test_ functions -- which is why the defect below shipped.

The defect: `_best_score_numba` wrote the two hole cards into slots 0 and 1 of a
reused `cards5` buffer ONCE per hole-card pair, then sorted that buffer in place
for every board combination. The sort permutes all five slots, so after the first
board combo a hole card could be sitting in slot 2, 3 or 4 -- which the next
board combo then overwrote, destroying it and leaving a stale board card behind.
Only the first of ten board combos scored the real five cards.

Effect measured before the fix: 277/300 random hands mis-scored, 35% of
head-to-head comparisons flipped, premium hands understated by ~13 equity points.

These assertions compare scores exactly. Both kernels index the same
`score_array`, so any difference at all is a bug -- never use a tolerance here.
"""

import numpy as np
import pytest

import hand_rank_evaluator as hre
from hand_indexing import BINOMIAL

pytestmark = pytest.mark.skipif(
    not hre.HAVE_NUMBA, reason="numba not available on this platform"
)

GAMES = {"plo4": 4, "plo5": 5, "plo6": 6}
BOARD_LENS = (5,)  # the kernel always scores a completed 5-card board


def _score_both(hand, board5, game):
    hand_combos = hre._HAND_COMBOS[game]
    board_combos = hre._BOARD_COMBOS
    score_array = hre.get_score_array()
    numba = float(
        hre._best_score_numba(hand, board5, hand_combos, board_combos, score_array, BINOMIAL)
    )
    numpy_ = float(
        hre._batch_best_score(
            hand[None, :], board5[None, :], hand_combos, board_combos, score_array, BINOMIAL
        )[0]
    )
    return numba, numpy_


@pytest.mark.parametrize("game,num_cards", sorted(GAMES.items()))
def test_kernels_agree_exactly_on_random_deals(game, num_cards):
    """The regression that matters: every combination must be scored, not just
    the first board combo of each hole-card pair."""
    rng = np.random.default_rng(20260814)
    mismatches = []
    for _ in range(200):
        cards = rng.choice(52, num_cards + 5, replace=False).astype(np.int32)
        hand = np.sort(cards[:num_cards])
        board5 = np.sort(cards[num_cards:])
        numba, numpy_ = _score_both(hand, board5, game)
        if numba != numpy_:
            mismatches.append((hand.tolist(), board5.tolist(), numba, numpy_))
    assert not mismatches, (
        f"{len(mismatches)}/200 {game} deals scored differently; "
        f"first: hand={mismatches[0][0]} board={mismatches[0][1]} "
        f"numba={mismatches[0][2]} numpy={mismatches[0][3]}"
    )


def test_made_flush_is_not_missed():
    """A hand whose only strong holding comes from a LATER board combo.

    Under the buffer defect the flush combination was never scored intact, so
    the kernel returned a much lower score than the true best five.
    """
    from card_encoding import hand_str_to_ints

    hand = np.sort(hand_str_to_ints("AsKs9h2c"))
    board5 = np.sort(hand_str_to_ints("QsJs4s7d9d"))
    numba, numpy_ = _score_both(hand, board5, "plo4")
    assert numba == numpy_


def test_head_to_head_winner_agrees():
    """Ordering, not just magnitude -- /handrank compares hero against opponent,
    so a consistent-but-wrong scale would still be harmless; a different
    ORDERING is not."""
    rng = np.random.default_rng(7)
    flips = 0
    for _ in range(200):
        cards = rng.choice(52, 13, replace=False).astype(np.int32)
        hero, opp, board5 = np.sort(cards[:4]), np.sort(cards[4:8]), np.sort(cards[8:13])
        hn, hp = _score_both(hero, board5, "plo4")
        on, op = _score_both(opp, board5, "plo4")
        if np.sign(hn - on) != np.sign(hp - op):
            flips += 1
    assert flips == 0, f"{flips}/200 head-to-head comparisons disagreed"


# ---------------------------------------------------------------------------
# Sampler uniformity (defect #2)
# ---------------------------------------------------------------------------
# run_handrank_mc drew its per-trial cards with
#   np.argpartition(rand_vals, needed - 1, axis=1)[:, :needed]
# and a comment claiming "we only need a random unordered slice". That is false:
# the result is sliced POSITIONALLY into board cards and opponent hands, so the
# order within the selection matters. argpartition returns a uniform random
# SUBSET but leaves it in partition order, which correlates with deck position --
# so the board systematically drew lower-ranked cards than the opponents.
# Measured before the fix: mean deck index 22.64 in board slots vs 23.73 in
# opponent slots (spread 2.78 across slots, against an unbiased 23.5).


def test_sampler_slots_are_positionally_unbiased():
    """Every output slot must be an unbiased draw from the deck.

    Without this, which cards land on the board vs in a villain's hand is
    decided partly by deck position rather than chance.
    """
    from hand_rank_evaluator import _sample_without_replacement

    deck_len, needed, trials = 48, 10, 200_000
    rng = np.random.default_rng(0)
    idx = _sample_without_replacement(rng, trials, deck_len, needed)

    assert idx.shape == (trials, needed)
    slot_means = idx.mean(axis=0)
    expected = (deck_len - 1) / 2.0
    # 0.35 is ~7x the Monte Carlo standard error at this trial count and ~8x
    # tighter than the 2.78 spread the defect produced.
    assert slot_means.max() - slot_means.min() < 0.35, (
        f"positional bias across slots: {np.round(slot_means, 2)}"
    )
    assert abs(slot_means.mean() - expected) < 0.1


def test_sampler_draws_without_replacement():
    from hand_rank_evaluator import _sample_without_replacement

    rng = np.random.default_rng(3)
    idx = _sample_without_replacement(rng, 5000, 48, 10)
    assert (idx < 48).all() and (idx >= 0).all()
    for row in idx:
        assert len(set(row.tolist())) == 10
