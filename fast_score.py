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

        No numba.prange here on purpose (see task-12-brief.md): the chunks
        this gets called on are already parallelised across the persistent
        ProcessPoolExecutor in range_ladder.rank_hands / evaluate_pass2.
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


def shared_board_pair_table(board5, board_combos, score_array, binomial):
    """Best 5-card score for EVERY hole pair against one fixed board.

    Returns a symmetric (52, 52) float64 table: `tbl[a, b]` is the best score
    obtainable from hole cards {a, b} plus the best 3 of `board5`.

    Why this exists: Omaha's rule is exactly-2-hole + exactly-3-board, so for a
    FIXED board the inner `max over board triples` depends only on the pair --
    not on the other 2-4 cards in the hand. Pass 1 scores 60,000 hands against
    one shared board per runout, so that inner max is currently recomputed
    ~60,000 times for each of the 1,326 distinct pairs. Computing it once per
    board turns each hand's ~100 five-card lookups (PLO5: 10 pairs x 10 triples)
    into ~10 gathers, measured 9-11x on the whole of Pass 1.

    Bit-identity is structural, not empirical: this evaluates the same
    score_array at the same combinatorial indices as `_batch_best_score_numba`,
    and combines them with the same `max`. `max` is exact on floats and nothing
    is summed, so there is no reassociation to change a result.

    Pairs involving a board card are EXCLUDED and left at -1.0. They cannot
    legally reach a caller -- Pass 1 applies `eligible_mask` before scoring, so
    no hand here shares a card with the board -- and including them is not
    merely wasteful but wrong: the 5-set would contain a duplicate card, and the
    combinatorial index of a non-strictly-increasing 5-tuple runs off the end of
    score_array (caught by tests/test_shared_board_scoring.py as an IndexError
    at 2,602,629 vs its size of 2,598,960). -1.0 matches the general kernel's
    own `best` initialiser and sits below every score_array value, so it can
    never win a `max`.
    """
    board5 = np.asarray(board5, dtype=np.int32)
    triples = board5[board_combos]                    # (C_b, 3)
    # Only the 47 cards not on the board -- see above.
    pool = np.setdiff1d(np.arange(52, dtype=np.int32), board5)
    ia, ib = np.triu_indices(pool.shape[0], k=1)
    a_idx, b_idx = pool[ia], pool[ib]                 # 1,081 unordered pairs
    n_pairs, n_tri = a_idx.shape[0], triples.shape[0]

    five = np.empty((n_pairs, n_tri, 5), dtype=np.int32)
    five[:, :, 0] = a_idx[:, None]
    five[:, :, 1] = b_idx[:, None]
    five[:, :, 2:] = triples[None, :, :]
    five.sort(axis=2)                                 # same ascending order the
                                                      # kernel's insertion sort
                                                      # produces (cards distinct)
    idx = (binomial[five[:, :, 0], 1]
           + binomial[five[:, :, 1], 2]
           + binomial[five[:, :, 2], 3]
           + binomial[five[:, :, 3], 4]
           + binomial[five[:, :, 4], 5])
    best = score_array[idx].max(axis=1)               # (n_pairs,)

    tbl = np.full((52, 52), -1.0, dtype=np.float64)   # board-card pairs and the
                                                      # diagonal stay -1.0
    tbl[a_idx, b_idx] = best
    tbl[b_idx, a_idx] = best
    return tbl


def shared_board_best_score(hands, hand_combos, pair_table):
    """Best score per hand, given a `shared_board_pair_table` for the board.

    One gather per hole pair, then a row max -- no sorting, no temporaries of
    the (N, C_h*C_b, 5) shape the general path builds.
    """
    n = hands.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    a = hands[:, hand_combos[:, 0]]
    b = hands[:, hand_combos[:, 1]]
    return pair_table[a, b].max(axis=1)


def warmup(score_array=None):
    """Trigger numba JIT compilation for _batch_best_score_numba ahead of
    the first real request, and exercise the SAME argument layouts
    range_ladder.score_hands actually passes at runtime -- not just any
    tiny array of the right dtype.

    Numba specializes (and, with cache=True, separately disk-caches) per
    argument *layout* (C-contiguous vs. general-strided/broadcast), not
    just per dtype+ndim. range_ladder._evaluate_pass2_chunk passes:
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
