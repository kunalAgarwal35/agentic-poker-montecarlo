"""Pass 2 must not recompute a hero's score once per trial.

A hero's hand is FIXED for the whole request, so its score on a trial depends
only on that trial's completed board. The Pass-2 sampler draws far fewer
distinct runouts than trials -- measured 1,892 distinct across 300,000 trials on
a flop, and 43 across 300,000 on a turn -- so the same hero score was being
recomputed ~159 times (flop) or ~6,977 times (turn) per distinct board.

Deduplicating changes no draw, no trial, no bucket membership and no villain
score. It only skips repeated evaluation of an identical (hand, board) pair, so
the result must be BIT-identical -- which is what these tests pin.
"""
import numpy as np
import pytest

import range_ladder as RL
from hand_rank_evaluator import get_score_array


def _reference(trials_arr, heroes, board_ints, hole_count, game, score_array):
    """The pre-dedup implementation: every hero scored on every trial."""
    t = trials_arr.shape[0]
    h = heroes.shape[0]
    villain_hands = trials_arr[:, :hole_count]
    runouts = trials_arr[:, hole_count:]
    need = runouts.shape[1]
    if need == 0:
        boards = board_ints
    else:
        board_tile = np.broadcast_to(board_ints, (t, board_ints.shape[0])).astype(np.int32)
        boards = np.concatenate([board_tile, runouts], axis=1)
    villain_scores = RL.score_hands(villain_hands, boards, game, score_array)
    hero_results = np.empty((h, t), dtype=np.float64)
    for j in range(h):
        hero_tile = np.broadcast_to(heroes[j], (t, hole_count))
        hero_scores = RL.score_hands(hero_tile, boards, game, score_array)
        hero_results[j] = np.where(
            hero_scores > villain_scores, 1.0,
            np.where(hero_scores == villain_scores, 0.5, 0.0),
        )
    return villain_scores, hero_results


def _build(rng, game, hole_count, n_heroes, t, board_len, repeats):
    """Trials whose runouts repeat heavily, like the real sampler's."""
    board_ints = rng.choice(52, size=board_len, replace=False).astype(np.int32)
    need = 5 - board_len
    pool = np.array([c for c in range(52) if c not in board_ints], dtype=np.int32)
    heroes = np.array(
        [rng.choice(pool, size=hole_count, replace=False) for _ in range(n_heroes)],
        dtype=np.int32,
    )
    distinct = [rng.choice(pool, size=need, replace=False) for _ in range(repeats)] if need else []
    rows = []
    for _ in range(t):
        villain = rng.choice(pool, size=hole_count, replace=False)
        runout = distinct[rng.integers(0, repeats)] if need else np.empty(0, dtype=np.int32)
        rows.append(np.concatenate([villain, runout]))
    return np.array(rows, dtype=np.int32), heroes, board_ints


@pytest.mark.parametrize("game,hole_count", [("plo4", 4), ("plo5", 5), ("plo6", 6)])
@pytest.mark.parametrize("board_len,repeats", [(3, 7), (4, 3)])
def test_bit_identical_off_river(game, hole_count, board_len, repeats):
    score_array = get_score_array()
    rng = np.random.default_rng(31337)
    trials, heroes, board_ints = _build(rng, game, hole_count, 3, 400, board_len, repeats)
    exp_v, exp_h = _reference(trials, heroes, board_ints, hole_count, game, score_array)
    got_v, got_h = RL._evaluate_pass2_chunk(trials, heroes, board_ints, hole_count, game, score_array)
    assert np.array_equal(got_v, exp_v)
    assert np.array_equal(got_h, exp_h)


@pytest.mark.parametrize("game,hole_count", [("plo5", 5), ("plo6", 6)])
def test_bit_identical_on_river(game, hole_count):
    """need == 0: no runout at all, so there is nothing to group."""
    score_array = get_score_array()
    rng = np.random.default_rng(4)
    trials, heroes, board_ints = _build(rng, game, hole_count, 2, 200, 5, 1)
    exp_v, exp_h = _reference(trials, heroes, board_ints, hole_count, game, score_array)
    got_v, got_h = RL._evaluate_pass2_chunk(trials, heroes, board_ints, hole_count, game, score_array)
    assert np.array_equal(got_v, exp_v)
    assert np.array_equal(got_h, exp_h)


def test_every_runout_distinct():
    """The degenerate case for the grouping: no repeats at all."""
    score_array = get_score_array()
    rng = np.random.default_rng(11)
    trials, heroes, board_ints = _build(rng, "plo5", 5, 2, 120, 3, 120)
    exp_v, exp_h = _reference(trials, heroes, board_ints, 5, "plo5", score_array)
    got_v, got_h = RL._evaluate_pass2_chunk(trials, heroes, board_ints, 5, "plo5", score_array)
    assert np.array_equal(got_v, exp_v)
    assert np.array_equal(got_h, exp_h)


def test_same_cards_different_draw_order_still_correct():
    """[A,B] and [B,A] complete the SAME board. They may or may not group
    together; either way the scores must match the reference."""
    score_array = get_score_array()
    board_ints = np.array([0, 1, 2], dtype=np.int32)
    heroes = np.array([[10, 11, 12, 13, 14]], dtype=np.int32)
    villain = [20, 21, 22, 23, 24]
    trials = np.array([villain + [30, 31], villain + [31, 30]], dtype=np.int32)
    exp_v, exp_h = _reference(trials, heroes, board_ints, 5, "plo5", score_array)
    got_v, got_h = RL._evaluate_pass2_chunk(trials, heroes, board_ints, 5, "plo5", score_array)
    assert np.array_equal(got_v, exp_v)
    assert np.array_equal(got_h, exp_h)
    assert got_h[0, 0] == got_h[0, 1]  # same board, same hero -> same result
