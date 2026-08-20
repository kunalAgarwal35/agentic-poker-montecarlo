"""Pass-1's shared-board scoring shortcut must be BIT-identical.

Pass 1 scores every villain hand against ONE board per runout. Omaha's rule is
exactly-2-hole + exactly-3-board, so for a fixed board the best board-triple for
a given hole PAIR does not depend on the rest of the hand -- it is the same for
all 60,000 hands sharing that runout. So the inner max can be computed once per
board for all 52*51/2 pairs and reused, turning ~100 five-card lookups per hand
into ~10 gathers.

`max` is exact on floats and nothing is summed, so this is required to agree
BIT-for-bit with the general kernel -- compare with np.array_equal, never
allclose. These tests are the proof; the speedup is worthless if the numbers
move.
"""
import numpy as np
import pytest

import range_ladder as RL
from fast_score import (
    batch_best_score,
    shared_board_best_score,
    shared_board_pair_table,
)
from hand_rank_evaluator import get_score_array

GAMES = [("plo4", 4), ("plo5", 5), ("plo6", 6)]


def _reference(hands, board5, game, score_array, use_numba):
    """The general path, exactly as score_hands drives it."""
    boards = np.ascontiguousarray(np.broadcast_to(board5, (hands.shape[0], 5)))
    return batch_best_score(
        hands, boards, RL._HAND_COMBOS[game], RL._BOARD_COMBOS,
        score_array, RL.BINOMIAL, use_numba=use_numba,
    )


def _draw(rng, n_hole, count):
    board5 = rng.choice(52, size=5, replace=False).astype(np.int32)
    pool = np.array([c for c in range(52) if c not in board5], dtype=np.int32)
    hands = np.array(
        [rng.choice(pool, size=n_hole, replace=False) for _ in range(count)],
        dtype=np.int32,
    )
    return board5, hands


@pytest.mark.parametrize("game,n_hole", GAMES)
@pytest.mark.parametrize("use_numba", [True, False])
def test_bit_identical_to_general_kernel(game, n_hole, use_numba):
    score_array = get_score_array()
    rng = np.random.default_rng(1234)
    for _ in range(5):
        board5, hands = _draw(rng, n_hole, 400)
        ref = _reference(hands, board5, game, score_array, use_numba)
        tbl = shared_board_pair_table(board5, RL._BOARD_COMBOS, score_array, RL.BINOMIAL)
        got = shared_board_best_score(hands, RL._HAND_COMBOS[game], tbl)
        assert np.array_equal(ref, got)


@pytest.mark.parametrize("game,n_hole", GAMES)
def test_score_hands_shared_board_matches_per_hand_board(game, n_hole):
    """score_hands' two branches must agree: handing it ONE board and handing
    it that same board repeated per row are the same question."""
    score_array = get_score_array()
    rng = np.random.default_rng(99)
    board5, hands = _draw(rng, n_hole, 300)
    shared = RL.score_hands(hands, board5, game, score_array)
    per_hand = RL.score_hands(
        hands, np.broadcast_to(board5, (hands.shape[0], 5)).copy(), game, score_array
    )
    assert np.array_equal(shared, per_hand)


@pytest.mark.parametrize("game,n_hole", GAMES)
def test_empty_hand_set(game, n_hole):
    """A runout every villain collides with leaves zero eligible hands."""
    score_array = get_score_array()
    rng = np.random.default_rng(5)
    board5, _ = _draw(rng, n_hole, 1)
    empty = np.empty((0, n_hole), dtype=np.int32)
    out = RL.score_hands(empty, board5, game, score_array)
    assert out.shape == (0,)


def test_non_contiguous_hands():
    """_rank_chunk passes villain_hands[idx] -- a fancy-indexed copy -- and
    Pass 2 passes column slices. Neither is guaranteed C-contiguous."""
    score_array = get_score_array()
    rng = np.random.default_rng(77)
    board5, hands = _draw(rng, 6, 200)
    view = hands[::2]
    ref = _reference(np.ascontiguousarray(view), board5, "plo6", score_array, True)
    tbl = shared_board_pair_table(board5, RL._BOARD_COMBOS, score_array, RL.BINOMIAL)
    got = shared_board_best_score(view, RL._HAND_COMBOS["plo6"], tbl)
    assert np.array_equal(ref, got)
