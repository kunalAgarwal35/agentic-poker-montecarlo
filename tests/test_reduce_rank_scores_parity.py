"""The ranking reduction must not change when its representation does.

Pass 1 used to hand `_reduce_rank_scores` a float64 matrix of SCORES with NaN
for ineligible cells; it now hands it an int16 matrix of RANK CODES with -1.
A code is the score's index in the sorted unique score_array values, so the
relabelling is strictly monotone: same ordering, same ties.

These tests do not merely check the new code against itself. They build a float
score matrix, derive its codes, and assert the new implementation on the CODES
reproduces the old float implementation on the SCORES bit-for-bit. That is the
property the change actually rests on -- `lower` and `upper` are integer counts
of "strictly worse" and "no better", which a monotone relabelling cannot move.
"""
import numpy as np
import pytest

from hand_rank_evaluator import get_score_array
from range_ladder import _reduce_rank_scores, score_values


def _float_reference(scores, eligible):
    """The pre-change implementation, on float SCORES. The oracle."""
    r, n = scores.shape
    beat_sum = np.zeros(n, dtype=np.float64)
    beat_cnt = np.zeros(n, dtype=np.float64)
    for i in range(r):
        idx = np.flatnonzero(eligible[i])
        if idx.size < 2:
            continue
        vs = scores[i, idx]
        order = np.sort(vs)
        lower = np.searchsorted(order, vs, side="left")
        upper = np.searchsorted(order, vs, side="right")
        ties = upper - lower - 1
        share = (lower + 0.5 * ties) / (idx.size - 1)
        beat_sum[idx] += share
        beat_cnt[idx] += 1.0
    return beat_sum, beat_cnt


def _make(rng, r, n, distinct, elig_p=0.9):
    """A score matrix with heavy ties -- 7,462 distinct values across 60,000
    hands makes ties the common case, not an edge case -- plus its codes."""
    values = score_values()
    pool = values[np.linspace(0, values.size - 1, distinct).astype(int)]
    scores = np.full((r, n), np.nan, dtype=np.float64)
    codes = np.full((r, n), -1, dtype=np.int16)
    eligible = rng.random((r, n)) < elig_p
    for i in range(r):
        idx = np.flatnonzero(eligible[i])
        drawn = rng.choice(pool, size=idx.size)
        scores[i, idx] = drawn
        codes[i, idx] = np.searchsorted(values, drawn).astype(np.int16)
    return scores, codes, eligible


@pytest.mark.parametrize("r,n,distinct", [(8, 500, 40), (5, 2000, 500), (12, 300, 3)])
def test_codes_reproduce_the_float_reference(r, n, distinct):
    rng = np.random.default_rng(4242)
    scores, codes, eligible = _make(rng, r, n, distinct)
    got_sum, got_cnt = _reduce_rank_scores(codes, eligible)
    exp_sum, exp_cnt = _float_reference(scores, eligible)
    assert np.array_equal(got_sum, exp_sum)
    assert np.array_equal(got_cnt, exp_cnt)


def test_all_tied():
    """Every hand identical: each beats half the field, exactly."""
    codes = np.zeros((3, 10), dtype=np.int16)
    eligible = np.ones((3, 10), dtype=bool)
    got_sum, got_cnt = _reduce_rank_scores(codes, eligible)
    assert np.allclose(got_sum / got_cnt, 0.5)
    scores = np.full((3, 10), float(get_score_array()[0]))
    exp_sum, _ = _float_reference(scores, eligible)
    assert np.array_equal(got_sum, exp_sum)


def test_strict_ordering():
    """Distinct codes: worst beats 0, best beats all."""
    codes = np.array([[0, 1, 2, 3]], dtype=np.int16)
    eligible = np.ones((1, 4), dtype=bool)
    got_sum, got_cnt = _reduce_rank_scores(codes, eligible)
    assert np.array_equal(got_sum, np.array([0.0, 1 / 3, 2 / 3, 1.0]))
    assert np.array_equal(got_cnt, np.ones(4))


def test_fewer_than_two_eligible_rows_are_skipped():
    codes = np.full((3, 4), -1, dtype=np.int16)
    eligible = np.zeros((3, 4), dtype=bool)
    eligible[1, 2] = True
    codes[1, 2] = 5
    got_sum, got_cnt = _reduce_rank_scores(codes, eligible)
    assert np.array_equal(got_sum, np.zeros(4))
    assert np.array_equal(got_cnt, np.zeros(4))
