"""Numba kernel for the scoring inner loop (Task 12).

`hand_rank_evaluator._batch_best_score` is pure NumPy. For PLO6 (15
hand-combos x 10 board-combos = 150 five-card rows per hand) it:

  - fancy-indexes to (N, C_h, 2) and (N, C_b, 3),
  - broadcasts both to (N, C_h, C_b, *),
  - concatenates into a (N, C_h*C_b, 5) int32 temporary -- 30MB at
    N=10,000 -- plus the broadcast intermediates,
  - sorts that temporary along the last axis,
  - computes a combinatorial-number-system index and gathers from
    score_array,
  - takes a row max.

The arithmetic is trivial; the cost is memory traffic and the sort. This
module does the identical work with zero temporaries: for each hand, loop
the hand-combo x board-combo pairs, build five cards on the stack,
insertion-sort them, compute the same combinatorial index, keep the running
max. Single-threaded by design -- see `_batch_best_score_numba`'s docstring
-- `cache=True` so the compiled machine code survives process restarts (see
`warmup()` below and task-12-report.md for the across-process timing that
verifies the cache is actually hit, not just assumed to be).

Deliberately its own module, not a change to hand_rank_evaluator.py, so the
numpy path there stays untouched as the reference implementation
tests/test_fast_score.py compares against. `_batch_best_score` and this
module's kernel must both index the same score_array with the same index
formula, so ANY disagreement between them is a bug, not sampling noise --
compare with np.array_equal, never allclose.
"""
import os

import numpy as np

try:
    from numba import njit  # noqa: F401
    HAVE_NUMBA = True
except Exception:  # pragma: no cover - platform without numba wheels
    HAVE_NUMBA = False

# Mirrors hand_rank_evaluator's HANDRANK_USE_NUMBA convention: set to '0' to
# force every score_hands call onto the numpy reference path (e.g. for A/B
# timing) without touching call sites. Independent of HANDRANK_USE_NUMBA --
# this flag gates the range-ladder scoring kernel, not hand_rank_evaluator's
# own per-trial MC kernel.
_ENV = os.environ.get('RANGE_LADDER_USE_NUMBA', '1')
USE_NUMBA_SCORE = HAVE_NUMBA and _ENV != '0'


if HAVE_NUMBA:

    @njit(cache=True)
    def _batch_best_score_numba(
        hands,             # (N, num_cards) int32
        boards,            # (N, 5) int32 -- always per-row; score_hands
                           #   broadcasts a shared board to this shape
                           #   before ever calling in here.
        hand_combos,       # (C_h, 2) int32
        board_combos,      # (C_b, 3) int32
        score_array,       # (2_598_960,) float64
        binomial,          # (53, 7) int64
    ):
        """Bit-identical replacement for
        hand_rank_evaluator._batch_best_score. Returns (N,) float64 -- the
        same values, same order, as the numpy path (no rounding is
        involved anywhere: both paths index the same score_array with the
        same combinatorial index, so agreement must be exact).

        No numba.prange here on purpose (see task-12-brief.md): the trial
        chunks this gets called on are already parallelised across the
        persistent ProcessPoolExecutor in range_ladder.evaluate_trials.
        Adding threads inside a single worker process on top of that
        oversubscribes the box and, per the brief, usually makes it
        slower rather than faster. This kernel stays single-threaded;
        parallelism stays exactly one layer up, at the process level.
        """
        n = hands.shape[0]
        c_h = hand_combos.shape[0]
        c_b = board_combos.shape[0]
        out = np.empty(n, dtype=np.float64)
        cards5 = np.empty(5, dtype=np.int32)

        for row in range(n):
            best = -1.0  # score_array values are >= 0 (see get_score_array)
            for h in range(c_h):
                # Scalars, not written into cards5 until just before each
                # sort below -- NOT the same pattern as
                # hand_rank_evaluator._best_score_numba, which stores the
                # hand pair into cards5[0]/cards5[1] once per `h` and then
                # lets the insertion sort permute the WHOLE cards5 buffer on
                # every `b` iteration, silently overwriting cards5[0]/[1]
                # with stale sorted remnants from the previous board combo
                # after the first iteration. That is a real, separate,
                # pre-existing bug (confirmed by direct comparison against
                # the numpy reference -- see task-12-report.md); this kernel
                # must not reproduce it, since it is required to be
                # bit-identical to _batch_best_score.
                h0 = hands[row, hand_combos[h, 0]]
                h1 = hands[row, hand_combos[h, 1]]
                for b in range(c_b):
                    cards5[0] = h0
                    cards5[1] = h1
                    cards5[2] = boards[row, board_combos[b, 0]]
                    cards5[3] = boards[row, board_combos[b, 1]]
                    cards5[4] = boards[row, board_combos[b, 2]]

                    # Insertion sort -- 5 elements, branch-predictable.
                    # Cards within a row are always distinct (see
                    # compute_range_ladder's collision-by-construction
                    # guarantee), so there is exactly one ascending
                    # permutation and this matches np.sort's output
                    # bit-for-bit (verified, not assumed, in
                    # tests/test_fast_score.py).
                    for i in range(1, 5):
                        key = cards5[i]
                        j = i - 1
                        while j >= 0 and cards5[j] > key:
                            cards5[j + 1] = cards5[j]
                            j -= 1
                        cards5[j + 1] = key

                    idx = (
                        binomial[cards5[0], 1]
                        + binomial[cards5[1], 2]
                        + binomial[cards5[2], 3]
                        + binomial[cards5[3], 4]
                        + binomial[cards5[4], 5]
                    )
                    s = score_array[idx]
                    if s > best:
                        best = s
            out[row] = best
        return out


def batch_best_score(hands, boards, hand_combos, board_combos, score_array,
                     binomial, use_numba=None):
    """Dispatcher: numba kernel by default, numpy reference path when numba
    is unavailable or explicitly disabled.

    `use_numba=None` (default) follows the module-level USE_NUMBA_SCORE flag
    (itself gated on HAVE_NUMBA + the RANGE_LADDER_USE_NUMBA env var). Pass
    True/False to force one path regardless of the flag -- this is how
    tests/test_fast_score.py runs both paths on IDENTICAL inputs within a
    single process for the bit-identical comparison.
    """
    fast = USE_NUMBA_SCORE if use_numba is None else (use_numba and HAVE_NUMBA)
    if fast:
        return _batch_best_score_numba(
            hands, boards, hand_combos, board_combos, score_array, binomial
        )
    # Local import: avoids a module-load-time cycle (hand_rank_evaluator
    # does not import fast_score, but importing it eagerly at module scope
    # here would still mean neither module could be the "first" one a
    # fresh interpreter imports without the other already existing).
    from hand_rank_evaluator import _batch_best_score
    return _batch_best_score(hands, boards, hand_combos, board_combos,
                             score_array, binomial)


def warmup(score_array=None):
    """Trigger numba JIT compilation for _batch_best_score_numba ahead of
    the first real request, and exercise the SAME argument layouts
    range_ladder.score_hands actually passes at runtime -- not just any
    tiny array of the right dtype.

    Numba specializes (and, with cache=True, separately disk-caches) per
    argument *layout* (C-contiguous vs. general-strided/broadcast), not
    just per dtype+ndim. range_ladder._evaluate_trials_chunk passes:
      - a broadcast (stride-0) `hands` array for the hero side
        (np.broadcast_to(heroes[j], (t, hole_count))),
      - a non-contiguous column-sliced `hands` array for the villain side
        (trials_arr[:, :hole_count]),
      - always a C-contiguous `boards` array (score_hands applies
        np.ascontiguousarray to it before calling in).
    Warming only a plain contiguous `hands` array would leave the other two
    layouts uncompiled, and the first REAL request hitting either one would
    still pay a first-call compile stall despite this warmup having run.
    """
    if not HAVE_NUMBA:
        return
    from hand_indexing import (
        BINOMIAL, PLO4_HAND_COMBOS, PLO5_HAND_COMBOS, PLO6_HAND_COMBOS,
        PLO4_BOARD_COMBOS,
    )
    if score_array is None:
        from hand_rank_evaluator import get_score_array
        score_array = get_score_array()

    board_combos = PLO4_BOARD_COMBOS
    board_rows = np.tile(np.arange(20, 25, dtype=np.int32), (2, 1))  # (2, 5), distinct cards

    for hand_combos, num_cards in (
        (PLO4_HAND_COMBOS, 4), (PLO5_HAND_COMBOS, 5), (PLO6_HAND_COMBOS, 6),
    ):
        # 1) plain C-contiguous hands.
        hand_row = np.arange(num_cards, dtype=np.int32)
        hands_c = np.tile(hand_row, (2, 1))
        _batch_best_score_numba(hands_c, board_rows, hand_combos, board_combos,
                                score_array, BINOMIAL)

        # 2) broadcast (stride-0) hands -- range_ladder's hero_tile layout.
        hands_bcast = np.broadcast_to(hand_row, (2, num_cards))
        _batch_best_score_numba(hands_bcast, board_rows, hand_combos, board_combos,
                                score_array, BINOMIAL)

        # 3) non-contiguous column-sliced hands -- range_ladder's
        #    trials_arr[:, :hole_count] layout.
        wide_row = np.arange(num_cards + 2, dtype=np.int32)
        wide = np.tile(wide_row, (2, 1))
        hands_sliced = wide[:, :num_cards]
        _batch_best_score_numba(hands_sliced, board_rows, hand_combos, board_combos,
                                score_array, BINOMIAL)
