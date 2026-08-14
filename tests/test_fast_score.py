"""Task 12: bit-identical equivalence between the numba scoring kernel
(fast_score._batch_best_score_numba) and the numpy reference path it
replaces (hand_rank_evaluator._batch_best_score).

Both index the same score_array with the same combinatorial-number-system
index, so ANY difference between them is a bug, not sampling noise --
compared throughout with np.array_equal, never allclose (see
task-12-brief.md). Covers PLO4/PLO5/PLO6 (hand-combo counts 6, 10, 15),
flop/turn/river boards, and the degenerate shapes: N == 0, N == 1, and an N
that is not a multiple of score_hands' default chunk size (2000).
"""
import numpy as np
import pytest

from card_encoding import generate_deck_ints
from hand_indexing import BINOMIAL, PLO4_BOARD_COMBOS
from hand_rank_evaluator import _batch_best_score, _HAND_COMBOS, get_score_array
import fast_score
from range_ladder import score_hands

pytestmark = pytest.mark.skipif(
    not fast_score.HAVE_NUMBA, reason="numba not available on this platform"
)

SCORE_ARRAY = get_score_array()
GAMES = [("plo4", 4), ("plo5", 5), ("plo6", 6)]

# Fixed literal seeds, one per random-input test below. These used to be
# `abs(hash((game, n))) % 2**32`, but Python randomises string hashing per
# process (three interpreters produced 3410057940 / 4015792210 / 1094733947
# for the same tuple), so every session ran these parity checks on different
# inputs and a failure could not be reproduced from the test source -- final
# review, Finding 6. The same seed across the parametrised games/board
# lengths is deliberate: the shapes already differ, and holding the draw
# fixed makes a failure comparable across them.
SEED_KERNEL_RANDOM_SAMPLE = 1_000_001
SEED_SCORE_HANDS = 1_000_002
SEED_HERO_TILE = 1_000_003


def _sample_joint(deck, hole_count, board_len, trials, rng):
    """Local stand-in for Task 11's deleted `sample_trials`: `trials` joint
    draws of hole_count + need DISTINCT cards from `deck` (first hole_count
    columns = a hand, remaining `need` = a board completion). This file only
    needs a generic "collision-impossible-by-construction" row generator to
    exercise score_hands' numba-vs-numpy equivalence -- it is not testing
    Task 13's ranking/equity sampling, so it does not need range_ladder's
    public sampling API (which now expects a hand population + buckets, not
    a bare joint draw)."""
    deck = np.asarray(deck, dtype=np.int32)
    need = 5 - board_len
    num_cards = hole_count + need
    draw = rng.random((trials, len(deck))).argsort(axis=1)[:, :num_cards]
    return deck[draw].astype(np.int32)


def _random_hands_and_boards(rng, num_cards, n):
    """n independent random (hand, 5-card board) rows, every card within a
    row distinct -- mirrors the collision-impossible-by-construction
    guarantee compute_range_ladder relies on (see task-11-report.md)."""
    hands = np.empty((n, num_cards), dtype=np.int32)
    boards = np.empty((n, 5), dtype=np.int32)
    for i in range(n):
        draw = rng.choice(52, size=num_cards + 5, replace=False)
        hands[i] = draw[:num_cards]
        boards[i] = draw[num_cards:]
    return hands, boards


# ---------------------------------------------------------------------------
# Raw kernel vs numpy reference, direct call (no score_hands chunking layer).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game,num_cards", GAMES)
@pytest.mark.parametrize("n", [50, 4321])  # 4321: not a multiple of chunk=2000
def test_kernel_matches_numpy_random_sample(game, num_cards, n):
    rng = np.random.default_rng(SEED_KERNEL_RANDOM_SAMPLE)
    hands, boards = _random_hands_and_boards(rng, num_cards, n)
    hand_combos = _HAND_COMBOS[game]

    expected = _batch_best_score(hands, boards, hand_combos, PLO4_BOARD_COMBOS,
                                 SCORE_ARRAY, BINOMIAL)
    actual = fast_score._batch_best_score_numba(hands, boards, hand_combos,
                                                PLO4_BOARD_COMBOS, SCORE_ARRAY, BINOMIAL)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == np.float64
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("game,num_cards", GAMES)
def test_kernel_matches_numpy_n_equals_zero(game, num_cards):
    hands = np.empty((0, num_cards), dtype=np.int32)
    boards = np.empty((0, 5), dtype=np.int32)
    hand_combos = _HAND_COMBOS[game]

    expected = _batch_best_score(hands, boards, hand_combos, PLO4_BOARD_COMBOS,
                                 SCORE_ARRAY, BINOMIAL)
    actual = fast_score._batch_best_score_numba(hands, boards, hand_combos,
                                                PLO4_BOARD_COMBOS, SCORE_ARRAY, BINOMIAL)
    assert actual.shape == expected.shape == (0,)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("game,num_cards", GAMES)
def test_kernel_matches_numpy_n_equals_one(game, num_cards):
    rng = np.random.default_rng(1)
    hands, boards = _random_hands_and_boards(rng, num_cards, 1)
    hand_combos = _HAND_COMBOS[game]

    expected = _batch_best_score(hands, boards, hand_combos, PLO4_BOARD_COMBOS,
                                 SCORE_ARRAY, BINOMIAL)
    actual = fast_score._batch_best_score_numba(hands, boards, hand_combos,
                                                PLO4_BOARD_COMBOS, SCORE_ARRAY, BINOMIAL)
    assert actual.shape == (1,)
    assert np.array_equal(actual, expected)


def test_kernel_sorts_five_cards_identically_to_numpy_sort():
    # Ties are impossible (cards within a row are always distinct -- see
    # `_sample_joint` above and compute_range_ladder), so there is exactly
    # one correct ascending permutation for any 5 distinct cards; this checks the
    # kernel's insertion sort finds it -- and that the resulting
    # combinatorial index matches the numpy path's -- for all 5! = 120
    # orderings a row could arrive in, not just "typical" already-mostly-
    # sorted input. Minimal 1-combo hand/board-combo arrays isolate exactly
    # one 5-tuple per call, with no other combo's score able to win the max
    # and mask a sort/index disagreement.
    from itertools import permutations
    hand_combos_1 = np.array([[0, 1]], dtype=np.int32)
    board_combos_1 = np.array([[0, 1, 2]], dtype=np.int32)
    base = np.array([3, 17, 29, 41, 50], dtype=np.int32)  # 5 distinct card ints
    for perm in permutations(range(5)):
        cards = base[list(perm)]
        hand = cards[:2].reshape(1, 2)
        board = cards[2:].reshape(1, 3)
        # Pad board to 5 columns; board_combos_1 only ever reads columns 0-2.
        board = np.concatenate([board, np.array([[12, 33]], dtype=np.int32)], axis=1)

        expected = _batch_best_score(hand, board, hand_combos_1, board_combos_1,
                                     SCORE_ARRAY, BINOMIAL)
        actual = fast_score._batch_best_score_numba(hand, board, hand_combos_1,
                                                    board_combos_1, SCORE_ARRAY, BINOMIAL)
        assert np.array_equal(actual, expected), perm


# ---------------------------------------------------------------------------
# score_hands-level equivalence: shared (river-style) vs per-row (flop/turn-
# style) boards, through the SAME call site range_ladder actually uses --
# exercising the broadcast (hero_tile) and non-contiguous (trials_arr column
# slice) layouts real requests produce, not just tidy contiguous arrays.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game,num_cards", GAMES)
@pytest.mark.parametrize("board_len", [3, 4, 5])  # flop, turn, river
def test_score_hands_numba_matches_numpy(game, num_cards, board_len):
    rng = np.random.default_rng(SEED_SCORE_HANDS)
    deck = generate_deck_ints(["As", "Ks"])  # arbitrary fixed removed cards
    board_ints = deck[:board_len]
    remaining_deck = deck[board_len:]

    trials_arr = _sample_joint(remaining_deck, hole_count=num_cards, board_len=board_len,
                               trials=777, rng=rng)  # 777: not a multiple of chunk=2000
    hands = trials_arr[:, :num_cards]
    need = 5 - board_len
    if need == 0:
        boards = board_ints
    else:
        runouts = trials_arr[:, num_cards:]
        board_tile = np.broadcast_to(board_ints, (trials_arr.shape[0], board_len)).astype(np.int32)
        boards = np.concatenate([board_tile, runouts], axis=1)

    numpy_out = score_hands(hands, boards, game, SCORE_ARRAY, use_numba=False)
    numba_out = score_hands(hands, boards, game, SCORE_ARRAY, use_numba=True)

    assert numpy_out.shape == numba_out.shape == (777,)
    assert np.array_equal(numpy_out, numba_out)


@pytest.mark.parametrize("game,num_cards", GAMES)
def test_score_hands_numba_matches_numpy_broadcast_hero_tile(game, num_cards):
    # Mirrors range_ladder._evaluate_pass2_chunk's hero-side call exactly:
    # a single hero hand broadcast (stride-0) across every trial row.
    #
    # The combinatorial-number-system index (both here and in production)
    # assumes every 5-card row is genuinely 5 DISTINCT cards -- a row with a
    # repeated card breaks the c0<c1<c2<c3<c4 assumption the index formula
    # relies on and can compute an out-of-range index (confirmed: an
    # earlier version of this test let the broadcast hero collide with a
    # trial's own board/runout and _batch_best_score raised IndexError).
    # compute_range_ladder guarantees this never happens in production by
    # keeping hero cards out of the trial-sampling deck entirely (they're
    # part of `dead`); this fixture reproduces that same guarantee by
    # construction, using three disjoint slices of the 52-card deck.
    rng = np.random.default_rng(SEED_HERO_TILE)
    full = np.arange(52, dtype=np.int32)
    hero = full[:num_cards]
    board_ints = full[num_cards:num_cards + 3]
    remaining_deck = full[num_cards + 3:]

    trials_arr = _sample_joint(remaining_deck, hole_count=num_cards, board_len=3,
                               trials=513, rng=rng)
    runouts = trials_arr[:, num_cards:]
    board_tile = np.broadcast_to(board_ints, (trials_arr.shape[0], 3)).astype(np.int32)
    boards = np.concatenate([board_tile, runouts], axis=1)
    hero_tile = np.broadcast_to(hero, (trials_arr.shape[0], num_cards))

    numpy_out = score_hands(hero_tile, boards, game, SCORE_ARRAY, use_numba=False)
    numba_out = score_hands(hero_tile, boards, game, SCORE_ARRAY, use_numba=True)
    assert np.array_equal(numpy_out, numba_out)
