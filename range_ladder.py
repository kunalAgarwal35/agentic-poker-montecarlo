"""Postflop range ladder: hero equity vs board-strength percentile slices.

See docs/superpowers/plans/2026-08-13-postflop-range-ladder.md

Task 13 replaces Task 11's single joint-trial pass with TWO passes that
sample differently, because ranking and equity want opposite things:

- Pass 1 (`rank_hands`) ranks villain HANDS -- not scenarios -- by scoring
  every sampled hand on the SAME small set of shared runouts (common random
  numbers: a paired comparison that reduces ranking noise per runout). This
  answers "how strong is this hand against the population", which is a
  property of the hand, not of any one runout it happens to hit.
- Pass 2 (`sample_pass2_trials` / `evaluate_pass2`) measures hero EQUITY by
  drawing a FRESH, independent runout per trial, stratified so every bucket
  gets an equal trial budget regardless of how few hands populate it.
  Sharing runouts here (as Task 11 effectively did, by ranking trials
  instead of hands) correlates every villain's outcome and was the direct
  cause of 5-7pp of noise surviving even at the whole-population bucket.

Task 11's design ranked the 10,000 TRIALS by the villain's hand rank on
its OWN completed board -- that ranks scenarios (a weak hand on a lucky
runout could out-rank a premium hand that bricked), not hands, so the
"top 5%" wasn't a hand-range and the edge hand wasn't nameable. See
task-13-brief.md for the full defect writeup and the two-pass fix.
"""
import os
from itertools import combinations

import numpy as np

import fast_score
from card_encoding import generate_deck_ints, hand_str_to_ints, ints_to_hand_str
from fast_score import batch_best_score
from hand_categories import CATEGORY_TOKENS
from hand_indexing import BINOMIAL
from hand_rank_evaluator import (
    _BOARD_COMBOS,
    _HAND_COMBOS,
    detect_game_type,
    get_score_array,
)
from multithread_ploequities3 import get_global_executor
from optimized_evaluator import best5_category_omaha_numba, get_category_array

DEFAULT_BUCKETS = (5, 15, 25, 40, 60, 100)

# --- Pass 1 defaults: rank hands on shared runouts ---------------------
#
# BUDGET: ~6s, not 2.5s. This is a deliberate user tradeoff (fix round 1,
# 2026-08-14), traded UP from the original 2.5s budget in exchange for a
# tighter top bucket -- range-ladder computation runs once per street, so
# the user chose to spend more wall-clock for less noise. Do NOT
# "optimise" this back down toward 2.5s; that would silently undo a
# decision the user made after seeing the accuracy/latency tradeoff
# below, not a performance regression to fix.
#
# DEFAULT_HANDS: size of the villain-hand population Pass 1 ranks. Once
# rank_runouts=300 was fixed (see DEFAULT_RANK_RUNOUTS below), `hands` IS
# the remaining lever for shrinking per-bucket spread -- spread scales as
# ~1/sqrt(hands) (order-statistics estimation noise on the percentile
# boundary, i.e. Stage-A "which hands even got sampled" noise, not Pass-1
# ranking noise -- see task-13-report.md's fix-round-1 addendum for the
# measurement that separated these two effects).
#
# 16-seed PLO6 sweep (board=6s7s4s, dead=[AsKs9h2c3d4d, QhJhTd3s5s8s], 1
# hero, rank_runouts=300, trials_per_bucket=50,000 fixed):
#   hands= 10,000: mean 1.83s                    worst-bucket spread 3.32pp
#   hands= 40,000: mean 4.38s, MAX 10.26s (*)     worst-bucket spread 2.40pp
#   hands= 60,000: mean 5.61s, max 5.78s          worst-bucket spread 1.75pp
# (*) A single 10.26s outlier against an otherwise-~4s-mean run, seen once
# across 16 seeds at hands=40,000 -- exactly the tail-latency behaviour
# flagged in Task 10 (parallel dispatch is set by the slowest chunk, and
# occasional stragglers happen). A cheap repro attempt (8 fresh seeds,
# same parameters) did NOT reproduce it (max 4.11s, mean 3.92s) -- treated
# as a one-off (pool contention / GC / OS scheduling on the measuring
# box), not a deterministic property of hands=40,000, and not chased
# further per the fix-round-1 instruction not to spend long on it. This IS
# the reason hands=60,000 was chosen over the faster-on-average 40,000:
# its max (5.78s) sits close to its mean, which is the property wanted --
# not "fast when nothing goes wrong".
#
# hands=60,000, PLO6, 1 hero, 16 seeds, rank_runouts=300,
# trials_per_bucket=50,000: mean 5.61s, max 5.78s. Per-bucket spread (pp):
# 5%=1.62  15%=1.27  25%=1.75  40%=1.67  60%=1.25  100%=0.47.
# PLAINLY: +/-1pp is NOT reached at this operating point -- the worst
# bucket (25%, 1.75pp) is still well above it, and only bucket 100% (the
# whole population) is actually inside +/-1pp. Do not read these numbers
# as "close enough" -- they are the accuracy this budget buys, not the
# target. Closing the remaining gap to 1pp needs roughly 2.6x more hands
# again (since spread ~ 1/sqrt(hands), (1.75/1.0)^2 ~= 3.1x fewer... i.e.
# ~2.6-3x more hands than 60,000 -- call it ~155,000 hands, extrapolated
# to ~14s at this box's measured per-hand Pass-1 cost). This is a
# MEASURED EXTRAPOLATION, not a measurement -- nobody has actually run
# hands=155,000 and confirmed 1pp or ~14s.
DEFAULT_HANDS = 60_000

# DEFAULT_RANK_RUNOUTS: how many shared runouts Pass 1 scores every hand
# against. The brief's own suggested starting point (30) was measured and
# rejected: a boundary-churn check (16 seeds, PLO6, a FIXED 10,000-hand
# population so only the runout sample varies between seeds) found 30
# shared runouts leaves 73.6% of hands flipping which bucket they land in
# from one independent Pass-1 run to the next, with edge-hand strength
# spreads of 3.4-6.6pp -- nowhere near "good enough to bucket correctly".
# 300 cuts churn to 29.5% and edge-hand spread to 1.4-2.4pp -- a real,
# large improvement, though not a complete fix (see task-13-report.md's
# "Correctness bar" section for the honest remainder: some churn survives
# even here). 300 was chosen, at the ORIGINAL 2.5s budget, as the point
# past which pushing rank_runouts further stopped being the most
# efficient use of the remaining time budget (rank_runouts=1000 alone
# blew that budget, 3.3s+, without proportionally shrinking the residual
# spread). Fix round 1 (2026-08-14) raised the time budget to ~6s and
# spent essentially all of that extra headroom on DEFAULT_HANDS instead
# of raising this further -- see DEFAULT_HANDS's comment above for why:
# once rank_runouts=300 was fixed, `hands` (not rank_runouts) was the
# lever that measurably moved the remaining spread.
DEFAULT_RANK_RUNOUTS = 300

# --- Pass 2 defaults: stratified equity, independent runouts ----------
#
# DEFAULT_TRIALS_PER_BUCKET: independent Pass-2 trials PER BUCKET (not
# total) -- stratification is what fixes Task 11's binding constraint
# (population sampling gave the top-5% bucket only 5% of trials, so the
# tight buckets were the worst-measured ones even though they mattered
# most). Equal budgets give equal precision across buckets regardless of
# how few hands populate the tightest one. Held fixed at 50,000 through
# fix round 1's hands sweep (see DEFAULT_HANDS above) because `hands`, not
# `trials_per_bucket`, was the lever that measurably moved the remaining
# spread once rank_runouts=300 was fixed -- see DEFAULT_HANDS's comment
# for the current (2026-08-14) chosen operating point, its measured mean
# and max seconds, its per-bucket spread, and the plain statement that
# +/-1pp is NOT reached there.
#
# Original selection (Task 13, 2.5s budget, hands=10,000): PLO6 (the
# binding case for TIME), >=16 seeds (the brief's minimum -- an 8-seed and
# a 3-seed estimate each produced a confidently wrong conclusion earlier
# in this project), inside budget with real headroom for a loaded machine
# rather than the largest value that fits an idle box (the old
# DEFAULT_TRIALS was picked at 3.6% headroom and overran 2.5s on a busy
# machine -- see git history / task-12-report.md). That 2.5s-budget sweep
# (hands=10,000/rank_runouts=300/trials_per_bucket in {30k,50k,80k}) found
# 80k already down to 5.1% headroom (too tight) and 50k the largest
# candidate with real margin -- see task-13-report.md for that sweep's
# full numbers, since read on their own they describe the ORIGINAL 2.5s
# operating point, not the current one.
DEFAULT_TRIALS_PER_BUCKET = 50_000

# Task 10: worker count for the persistent process pool (see
# multithread_ploequities3.get_global_executor). Sized to the box's core
# count rather than that module's own _DEFAULT_WORKERS=4 (tuned for a
# different, lighter-weight caller): this is the first thing in the app to
# create the pool, so it gets to pick the size. get_global_executor()
# ignores num_workers on every call after the first, so if some other path
# creates the pool first with a smaller count, evaluate_pass2/rank_hands
# silently ride along on that smaller pool rather than failing -- correct,
# just less parallel.
DEFAULT_POOL_WORKERS = os.cpu_count() or 4

# Below this many total hand-evaluations, process-pool dispatch overhead
# (pickling villains/heroes/board per chunk, IPC round-trip) costs more
# than it saves. "A few thousand" per the Task 10 brief; not re-measured
# precisely because every realistic caller sits far on one side of this
# line or the other -- see task-10-report.md.
_PARALLEL_MIN_EVALS = 4000

# hand_categories.CATEGORY_TOKENS is index-aligned (low -> high strength) with
# the category_array.npy lookup used below, so mapping through it keeps this
# module's labels in lockstep with the engine's own naming (PQL's
# exactHandType, category_playground, etc.) instead of re-deriving strength
# bands from score_array by hand. Only the spacing/casing differs from the
# brief's nine literal labels.
_CATEGORY_LABELS = {
    "highcard": "high card",
    "pair": "pair",
    "twopair": "two pair",
    "trips": "trips",
    "straight": "straight",
    "flush": "flush",
    "fullhouse": "full house",
    "quads": "quads",
    "straightflush": "straight flush",
}


def _pool_worker_warmup_task():
    """Module-level (Windows-spawn-picklable, same rationale as the other
    module-level worker entrypoints below) warmup task run inside EVERY
    persistent pool worker process: forces the worker to spawn and, via
    fast_score.warmup(), triggers that worker's own numba JIT (or, with
    cache=True and the disk cache already populated by whichever process
    warmed first, a cache load instead of a fresh compile).

    Real evaluate_pass2/rank_hands chunks land on these SAME worker
    processes, so warming only the main process (as fast_score.warmup()
    called directly would) leaves every worker's first real chunk paying a
    compile-or-cache-load stall the first time score_hands' numba path
    executes there.
    """
    fast_score.warmup()
    return 1


def warmup_pool():
    """Spawn and warm every worker of the persistent process pool, sized to
    DEFAULT_POOL_WORKERS, so the first real range-ladder request doesn't pay
    process-creation cost. Meant to be called once, from server.py's
    existing background warmup thread.

    Also warms fast_score's numba kernel in THIS (main) process first --
    compute_range_ladder's serial path (small workloads, or
    `parallel=False`) runs score_hands in-process, never touching the pool
    -- and then in every pool worker via `_pool_worker_warmup_task`, so
    both the serial and parallel code paths are covered.

    Deliberately does NOT call multithread_ploequities3.warmup_executor():
    that function hardcodes _DEFAULT_WORKERS (4 -- sized for a different,
    lighter-weight caller, see the DEFAULT_POOL_WORKERS comment above) for
    both pool creation *and* for how many warmup tasks it submits, so it
    would warm only 4 of this box's cores even if the pool were already
    sized larger. This calls the same underlying interface
    (get_global_executor) at DEFAULT_POOL_WORKERS instead -- still the one
    persistent pool, not a second one.
    """
    fast_score.warmup()
    executor = get_global_executor(DEFAULT_POOL_WORKERS)
    futures = [executor.submit(_pool_worker_warmup_task) for _ in range(DEFAULT_POOL_WORKERS)]
    for f in futures:
        f.result()


def sample_villains(deck, num_cards, n, rng):
    """Distinct legal villain hands drawn from `deck`, as sorted rows.

    Falls back to exhaustive enumeration when the space is smaller than `n`,
    so tiny decks return every hand exactly once instead of looping forever.

    This is Pass 1's villain population: `n` DISTINCT hands, each scored
    against Pass 1's shared runouts to produce a ranking. Distinctness here
    (unlike Pass 2's trials, which reuse bucket members with replacement)
    matters because a duplicated hand would be double-counted in the
    "share of the field it beats" computation without contributing any new
    information.
    """
    deck = np.asarray(deck, dtype=np.int32)
    total = len(deck)
    if total < num_cards:
        return np.empty((0, num_cards), dtype=np.int32)

    # Exhaustive when the space is small enough to enumerate cheaply.
    space = int(BINOMIAL[total, num_cards]) if total <= 52 and num_cards <= 6 else n + 1
    if space <= max(n, 1):
        rows = [sorted(c) for c in combinations(deck.tolist(), num_cards)]
        return np.array(rows, dtype=np.int32)

    seen = set()
    rows = []
    while len(rows) < n:
        need = n - len(rows)
        draw = rng.random((need * 2, total)).argsort(axis=1)[:, :num_cards]
        cand = np.sort(deck[draw], axis=1)
        for row in cand:
            key = tuple(row.tolist())
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            if len(rows) == n:
                break
    return np.array(rows, dtype=np.int32)


def sample_runouts(deck, board_len, r, rng):
    """`r` shared board completions, drawn ONCE and reused across every
    villain hand Pass 1 ranks -- the common-random-numbers trick that makes
    ranking a paired comparison instead of an independent one per hand.

    River (`board_len == 5`, `need == 0`): a single empty runout. The board
    is already complete, so there is nothing to sample and ranking is
    exact (score every hand once, no averaging over runouts at all).
    """
    deck = np.asarray(deck, dtype=np.int32)
    need = 5 - board_len
    if need <= 0:
        return np.empty((1, 0), dtype=np.int32)
    draw = rng.random((r, len(deck))).argsort(axis=1)[:, :need]
    return deck[draw].astype(np.int32)


def eligible_mask(hands, runout):
    """False for hands holding a card that `runout` also uses.

    Those pairings are impossible and must never be scored. Pass 1's
    shared runouts are drawn BEFORE the villain hands, so a runout can
    legally hold a card a given villain hand also holds -- unlike Pass 2,
    where the runout is drawn after (and explicitly excludes) the
    villain's cards, so no collision is possible there by construction.
    Skipping an ineligible (hand, runout) pair here and averaging each
    hand only over the runouts it IS eligible for is the correct
    conditional distribution -- unbiased, not an approximation.
    """
    if runout.size == 0:
        return np.ones(hands.shape[0], dtype=bool)
    return ~np.isin(hands, runout).any(axis=1)


def score_hands(hands, board5, game, score_array, chunk=2000, use_numba=None):
    """Best 5-card score for each hand. Higher = better.

    `board5` is either a single board shared by every hand, shape (5,), or
    a per-hand board, shape (hands.shape[0], 5) -- one row per hand, as
    Pass 2's evaluator needs: every trial completes the board with its own
    independent runout, so hand i must be scored against board5[i], not one
    board shared across the whole call.

    The actual per-chunk scoring is `fast_score.batch_best_score`, a numba
    kernel. `use_numba` passes straight through to that dispatcher (None ->
    module-level default, True/False -> force a path) -- exposed here, not
    just in fast_score, so tests/test_fast_score.py can force either path
    through the SAME call site range_ladder actually uses.
    """
    hand_combos = _HAND_COMBOS[game]
    n = hands.shape[0]
    board5 = np.asarray(board5)
    per_hand_board = board5.ndim == 2
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk):
        block = hands[start:start + chunk]
        if per_hand_board:
            boards = np.ascontiguousarray(board5[start:start + chunk])
        else:
            boards = np.ascontiguousarray(np.broadcast_to(board5, (block.shape[0], 5)))
        out[start:start + chunk] = batch_best_score(
            block, boards, hand_combos, _BOARD_COMBOS, score_array, BINOMIAL,
            use_numba=use_numba,
        )
    return out


def _split_rows(arr, num_chunks):
    """Contiguous, order-preserving split into at most `num_chunks` pieces.

    Shared by Pass 1 (splits the runout axis) and Pass 2 (splits the trial
    axis). Clamped so num_chunks never exceeds the row count -- np.array_split
    would otherwise hand back empty trailing chunks, which is wasted
    dispatch, not a correctness problem.
    """
    num_chunks = max(1, min(int(num_chunks), arr.shape[0]))
    return np.array_split(arr, num_chunks, axis=0)


# ---------------------------------------------------------------------------
# Pass 1: rank hands on shared runouts.
# ---------------------------------------------------------------------------

def _rank_chunk(villain_hands, board_ints, runouts_chunk, game, score_array):
    """Score EACH runout in this CONTIGUOUS chunk of shared runouts against
    only the villain hands ELIGIBLE for it. Returns (scores, eligible), both
    shape (runouts_chunk.shape[0], villain_hands.shape[0]) -- NOT reduced.

    A shared runout can hold a card a villain hand also holds -- collisions
    return in Pass 1 (unlike Pass 2, where the runout is drawn after the
    villain and cannot collide). Scoring an ineligible (hand, runout) pair
    would feed the bounds-unchecked scoring kernel a card set with a
    duplicate card, which crashes the process rather than raising -- so the
    eligibility mask is computed and applied BEFORE score_hands ever sees
    that runout's board, never as an after-the-fact filter on results that
    were already (illegally) computed. Ineligible cells are left as NaN;
    `_reduce_rank_scores` only ever reads cells `eligible` marks True.

    This is the only part of Pass 1 that gets parallelised: the caller
    concatenates chunks along axis 0 (the runout axis, never summed) and
    performs the ranking reduction (`_reduce_rank_scores`) exactly once,
    afterward, on the fully assembled matrix. That keeps the reduction's
    floating-point summation order fixed regardless of how many chunks or
    workers produced the pieces being reduced -- see rank_hands.
    """
    hands = villain_hands.shape[0]
    r = runouts_chunk.shape[0]
    scores = np.full((r, hands), np.nan, dtype=np.float64)
    eligible = np.empty((r, hands), dtype=bool)
    for i, runout in enumerate(runouts_chunk):
        mask = eligible_mask(villain_hands, runout)
        eligible[i] = mask
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        board5 = (np.concatenate([board_ints, runout]) if runout.size
                 else np.asarray(board_ints)).astype(np.int32)
        scores[i, idx] = score_hands(villain_hands[idx], board5, game, score_array)
    return scores, eligible


def _rank_chunk_worker(villain_hands, board_ints, runouts_chunk, game):
    """Module-level entrypoint submitted to the persistent process pool.

    Deliberately does NOT take score_array as a parameter -- see the same
    rationale on `_evaluate_pass2_chunk_worker` below (shipping a ~21MB
    array per dispatch would dwarf the actual per-chunk computation).  Must
    stay at module level: Windows' spawn start method pickles a reference
    to this function by qualified name and re-imports `range_ladder` in the
    child process to resolve it.
    """
    return _rank_chunk(villain_hands, board_ints, runouts_chunk, game, get_score_array())


def _reduce_rank_scores(scores, eligible):
    """The ranking reduction: for each runout (row), each ELIGIBLE hand's
    share of the eligible field it beats (win + 1/2 tie, normalised by
    field_size - 1), summed into beat_sum/beat_cnt across runouts.

    Runs ONCE, always serially, on the fully assembled (runouts, hands)
    matrices -- never split across workers -- so it produces the exact
    same beat_sum/beat_cnt regardless of how `scores`/`eligible` were
    computed (single process or any number of parallel chunks): those
    chunks are only ever concatenated (never summed) before reaching here,
    so this function's own summation order is the only one that matters,
    and it never changes.

    Ineligible hands (colliding with that runout) are excluded from BOTH
    the ranking field AND their own beat_sum/beat_cnt for that runout --
    Task 13's "skip, don't score" collision rule (see eligible_mask).
    """
    r, n = scores.shape
    beat_sum = np.zeros(n, dtype=np.float64)
    beat_cnt = np.zeros(n, dtype=np.float64)
    for i in range(r):
        idx = np.flatnonzero(eligible[i])
        if idx.size < 2:
            # Need at least 2 eligible hands for "share of the field"
            # (excluding self) to be defined. Essentially impossible with
            # a realistic hand population and deck; skipped defensively.
            continue
        vs = scores[i, idx]
        order = np.sort(vs)
        lower = np.searchsorted(order, vs, side="left")     # strictly worse
        upper = np.searchsorted(order, vs, side="right")
        ties = upper - lower - 1                              # excluding self
        share = (lower + 0.5 * ties) / (idx.size - 1)
        beat_sum[idx] += share
        beat_cnt[idx] += 1.0
    return beat_sum, beat_cnt


def rank_hands(villain_hands, board_ints, runouts, game, score_array,
               parallel=None, num_workers=None):
    """Pass 1: rank every villain hand by its mean share of the field it
    beats, averaged over the shared runouts it is eligible for.

    Returns (strength, eligible_counts): strength[i] is hand i's ranking
    (NaN-free -- 0.0 for a hand eligible on zero runouts, which
    `compute_range_ladder` filters out via eligible_counts before bucketing,
    the same guard the pre-Task-11 `evaluate_population` used). Cost is
    hands x runouts.shape[0] evaluations -- each hand scored once per
    runout, never pairwise (the sort/searchsorted trick above avoids an
    O(hands^2) comparison).
    """
    hands = villain_hands.shape[0]
    r = runouts.shape[0]
    n_evals = hands * r

    if parallel is None:
        parallel = r > 1 and n_evals >= _PARALLEL_MIN_EVALS

    if parallel and r > 1:
        workers = num_workers or DEFAULT_POOL_WORKERS
        executor = get_global_executor(workers)
        chunks = _split_rows(runouts, workers)
        futures = [executor.submit(_rank_chunk_worker, villain_hands, board_ints, chunk, game)
                   for chunk in chunks]
        # .result() in chunk-submission order, not as_completed(): the
        # concatenation below must reassemble runouts in a fixed order
        # regardless of which worker finishes first.
        partials = [f.result() for f in futures]
        scores = np.concatenate([p[0] for p in partials], axis=0)
        eligible = np.concatenate([p[1] for p in partials], axis=0)
    else:
        scores, eligible = _rank_chunk(villain_hands, board_ints, runouts, game, score_array)

    beat_sum, beat_cnt = _reduce_rank_scores(scores, eligible)
    strength = np.divide(beat_sum, beat_cnt, out=np.zeros_like(beat_sum), where=beat_cnt > 0)
    return strength, beat_cnt


# ---------------------------------------------------------------------------
# Pass 2: stratified hero equity on independent runouts.
# ---------------------------------------------------------------------------

def sample_pass2_trials(deck, board_len, hole_count, villain_hands, bucket_indices,
                        trials_per_bucket, rng):
    """Stratified Pass-2 draws: `trials_per_bucket` INDEPENDENT trials per
    bucket, concatenated into one array as contiguous per-bucket blocks (in
    `bucket_indices` order) -- equal budget per bucket, not a share
    proportional to population weight, which is what fixes Task 11's
    binding constraint (the top-5% bucket getting only 5% of trials).

    Each row: one villain hand drawn UNIFORMLY AT RANDOM WITH REPLACEMENT
    from that bucket's members (`villain_hands[bucket_indices[b]]`, Pass
    1's ranking) + a FRESH runout drawn from `deck` minus board (deck
    already excludes board/dead, which includes every hero's cards) minus
    THAT row's own villain cards.

    No collision is possible: villain-card columns of `deck` have their
    random key forced to +inf before the argsort-of-random-keys draw picks
    the `need` lowest-keyed columns, so a masked column can only be chosen
    if fewer than `need` unmasked columns remain -- never true for a real
    deck. This is the structural guarantee behind "Pass 2 has no
    collisions": the runout is drawn AFTER the villain and explicitly
    excludes its cards, unlike Pass 1's shared runouts (drawn before any
    villain, so a collision is possible there and must be masked out by
    `eligible_mask` instead).
    """
    deck_arr = np.asarray(deck, dtype=np.int32)
    total = deck_arr.size
    need = 5 - board_len

    blocks = []
    for idx in bucket_indices:
        pick = rng.integers(0, idx.size, size=trials_per_bucket)
        blocks.append(villain_hands[idx[pick]])
    villains_trial = np.concatenate(blocks, axis=0)      # (B*T, hole_count)

    if need == 0:
        # River: the board is already complete -- no runout to draw.
        runouts = np.empty((villains_trial.shape[0], 0), dtype=np.int32)
        return np.concatenate([villains_trial, runouts], axis=1).astype(np.int32)

    keys = rng.random((villains_trial.shape[0], total))
    collide = np.zeros((villains_trial.shape[0], total), dtype=bool)
    for k in range(hole_count):
        collide |= (villains_trial[:, k:k + 1] == deck_arr[None, :])
    keys = np.where(collide, np.inf, keys)
    order = keys.argsort(axis=1)[:, :need]
    runouts = deck_arr[order].astype(np.int32)

    return np.concatenate([villains_trial, runouts], axis=1).astype(np.int32)


def _evaluate_pass2_chunk(trials_arr, heroes, board_ints, hole_count, game, score_array):
    """Score one CONTIGUOUS chunk of Pass-2 trial rows: exact villain score
    and every hero's result, per trial, on that trial's own completed
    board. Structurally the per-trial computation Task 11's
    `_evaluate_trials_chunk` used, reused here for Pass 2's stratified
    draws instead of Task 11's population-wide joint sampling.

    Returns (villain_scores (T,), hero_results (H, T)) for this chunk's T
    rows -- the caller concatenates chunks back together along the trial
    axis, in chunk-submission order.
    """
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

    villain_scores = score_hands(villain_hands, boards, game, score_array)

    hero_results = np.empty((h, t), dtype=np.float64)
    for j in range(h):
        hero_tile = np.broadcast_to(heroes[j], (t, hole_count))
        hero_scores = score_hands(hero_tile, boards, game, score_array)
        hero_results[j] = np.where(
            hero_scores > villain_scores, 1.0,
            np.where(hero_scores == villain_scores, 0.5, 0.0),
        )
    return villain_scores, hero_results


def _evaluate_pass2_chunk_worker(trials_arr, heroes, board_ints, hole_count, game):
    """Module-level entrypoint submitted to the persistent process pool.

    Deliberately does NOT take score_array as a parameter. ProcessPoolExecutor
    pickles every argument across the process boundary; score_array is a
    ~2.6M-element float64 array (~21MB), and shipping it on every dispatch
    would dwarf the actual per-chunk computation. Instead each worker calls
    get_score_array(), which np.load()s score_array.npy once per process
    and caches it at module scope; the cost is paid once per worker's
    lifetime, not once per chunk.

    Must stay at module level: Windows' spawn start method pickles a
    reference to this function by qualified name and re-imports
    `range_ladder` in the child process to resolve it.
    """
    return _evaluate_pass2_chunk(trials_arr, heroes, board_ints, hole_count,
                                 game, get_score_array())


def evaluate_pass2(trials_arr, heroes, board_ints, hole_count, game, score_array,
                   parallel=None, num_workers=None):
    """The O(T x (1 + H)) pass over every Pass-2 trial. Returns
    (villain_scores (T,), hero_results (H, T)).

    Parallelised over trial chunks: every trial is independent and exact on
    its own -- there is no reduction across trials at this layer, only a
    split-then-concatenate, exactly like Task 11's evaluate_trials was.
    Chunks are reassembled by np.concatenate in chunk-SUBMISSION order
    (never `as_completed()`), which reproduces the exact same per-trial
    numbers regardless of how many workers ran or which one finished first
    -- parallel output is bit-identical to serial output, not merely close.

    `parallel`: None (default) auto-decides from population size -- below
    `_PARALLEL_MIN_EVALS` total hand-evaluations, or a single trial (which
    cannot be split), the pool overhead isn't worth it. Pass True/False to
    force one path or the other.
    `num_workers`: chunk count for the parallel path (defaults to
    DEFAULT_POOL_WORKERS).
    """
    t = trials_arr.shape[0]
    h = heroes.shape[0]

    if parallel is None:
        n_evals = (1 + h) * t
        parallel = t > 1 and n_evals >= _PARALLEL_MIN_EVALS

    if parallel and t > 1:
        workers = num_workers or DEFAULT_POOL_WORKERS
        executor = get_global_executor(workers)
        chunks = _split_rows(trials_arr, workers)
        futures = [executor.submit(_evaluate_pass2_chunk_worker,
                                   chunk, heroes, board_ints, hole_count, game)
                   for chunk in chunks]
        partials = [f.result() for f in futures]
        villain_scores = np.concatenate([p[0] for p in partials])
        hero_results = np.concatenate([p[1] for p in partials], axis=1)
    else:
        villain_scores, hero_results = _evaluate_pass2_chunk(
            trials_arr, heroes, board_ints, hole_count, game, score_array
        )
    return villain_scores, hero_results


def describe_category(hand_ints, board5):
    """Human label for a hand's made category, e.g. "flush", "full house".

    Delegates to optimized_evaluator's numba category lookup (the same
    category_array.npy the rest of the engine uses via
    pql.runtime.evaluator.player_categories) rather than re-deriving hand
    strength here, so labels agree with the rest of the engine by
    construction.
    """
    game = detect_game_type(len(hand_ints))
    idx = best5_category_omaha_numba(
        np.asarray(hand_ints, dtype=np.int32),
        np.asarray(board5, dtype=np.int32),
        _HAND_COMBOS[game],
        _BOARD_COMBOS,
        get_score_array(),
        get_category_array(),
        BINOMIAL,
    )
    return _CATEGORY_LABELS[CATEGORY_TOKENS[int(idx)]]


def _compatible_board5(board_ints, runout_rows, hand):
    """Board built from `board_ints` + the first `runout_rows` row `hand` is
    eligible for (no shared card), or None if every runout collides.

    describe_category feeds `hand` + this board straight into a
    bounds-unchecked numba kernel; a runout sharing a card with `hand`
    would produce a card set with a duplicate. Reuses `eligible_mask` --
    the same collision check Pass 1 uses when scoring the whole population
    -- applied to a single hand instead. On the river `runout_rows` is the
    single empty completion, so every hand is trivially compatible
    (eligible_mask is vacuously True for an empty runout).
    """
    hand2d = hand[None, :]
    for runout in runout_rows:
        if eligible_mask(hand2d, runout)[0]:
            return (np.concatenate([board_ints, runout]) if runout.size
                   else np.asarray(board_ints)).astype(np.int32)
    return None


def _bucket_slices_and_edges(strength, villain_hands, buckets, board_ints, rank_runouts_arr):
    """Cumulative top-`pct`% index slices (strongest first) and each
    bucket's edge (the weakest hand in that slice, from Pass 1's own
    ranking -- a genuine hand ranked by its own averaged strength, not a
    per-trial artefact).

    Hero-independent by construction: computed once from `strength` alone,
    before any hero is considered, so every hero's ladder shares the exact
    same bucket membership and edge hands.
    """
    order = np.argsort(-strength, kind="stable")      # strongest first
    n = order.size
    bucket_indices = []
    edges = []
    for pct in buckets:
        take = max(1, int(round(n * pct / 100.0)))
        sl = order[:take]
        bucket_indices.append(sl)
        edge_idx = int(sl[-1])                         # weakest hand in the slice
        hand = villain_hands[edge_idx]
        board5 = _compatible_board5(board_ints, rank_runouts_arr, hand)
        edges.append({
            "cards": ints_to_hand_str(hand),
            # Essentially impossible with any real rank_runouts count, but
            # never assumed: an empty label is a cosmetic gap, not a crash.
            "category": describe_category(hand, board5) if board5 is not None else "",
        })
    return bucket_indices, edges


def build_rungs(buckets, edges, equity_row):
    """One rung per bucket: `edges[i]` (hero-independent, from Pass 1) +
    `equity_row[i]` (this hero's Pass-2 equity for that bucket)."""
    return [
        {"bucket": pct, "equity": float(eq), "edge": edge}
        for pct, edge, eq in zip(buckets, edges, equity_row)
    ]


def compute_range_ladder(board, dead, heroes, buckets=DEFAULT_BUCKETS,
                         hands=DEFAULT_HANDS, rank_runouts=DEFAULT_RANK_RUNOUTS,
                         trials_per_bucket=DEFAULT_TRIALS_PER_BUCKET, seed=None,
                         parallel=None, num_workers=None):
    """The whole feature: hero equity vs board-strength percentile slices.

    Two passes (see the module docstring): Pass 1 (`hands` x `rank_runouts`)
    ranks villain hands on shared runouts; Pass 2 (`trials_per_bucket` per
    bucket, `len(buckets)` buckets) measures each hero's equity against
    each bucket on fresh, independent, per-trial runouts.

    `parallel`/`num_workers` pass straight through to both rank_hands and
    evaluate_pass2 (see their docstrings); left at their defaults, sizing
    auto-decides from the workload, which is the right choice for normal
    callers. Exposed here mainly for tests that need to force one path or a
    specific worker count while holding every other input fixed.

    See spec Section 5 for the response shape; `rank_runouts` and
    `trials_per_bucket` are additionally reported so the sampling effort
    behind each number is visible (see task-13-brief.md).
    """
    board_ints = hand_str_to_ints(board)
    board_len = len(board_ints)
    if board_len not in (3, 4, 5):
        raise ValueError(f"board must be 3, 4 or 5 cards, got {board_len}")

    # Built WITHOUT deduping first: a card repeated within one `dead` entry
    # ("AsAs9c2c") or across two different entries (two hands both claiming
    # "As") must show up as a literal duplicate here. Collapsing straight
    # into a set (as the old code did) silently absorbs both -- two hands
    # can never legally share a card.
    dead_cards = []
    for d in dead:
        dead_cards.extend(ints_to_hand_str(hand_str_to_ints(d))[i:i + 2]
                          for i in range(0, len(d), 2))
    if len(dead_cards) != len(set(dead_cards)):
        seen, dupes = set(), []
        for c in dead_cards:
            if c in seen and c not in dupes:
                dupes.append(c)
            seen.add(c)
        raise ValueError(f"duplicate card(s) in `dead`: {sorted(dupes)}")
    dead_set = set(dead_cards)

    # A card cannot be simultaneously dead (in a hand) and live (on the
    # board).
    board_norm = ints_to_hand_str(board_ints)
    board_cards = {board_norm[i:i + 2] for i in range(0, len(board_norm), 2)}
    overlap = dead_set & board_cards
    if overlap:
        raise ValueError(
            f"card(s) {sorted(overlap)} appear in both `dead` and `board`"
        )

    for hero in heroes:
        # Normalise exactly as `dead` was, so case differences cannot cause a
        # spurious rejection (postfloper lower-cases dead cards in places).
        norm = ints_to_hand_str(hand_str_to_ints(hero["cards"]))
        cards = [norm[i:i + 2] for i in range(0, len(norm), 2)]
        missing = [c for c in cards if c not in dead_set]
        if missing:
            raise ValueError(
                f"hero {hero['id']} cards {missing} absent from `dead`; "
                "villains would be allowed to hold them"
            )

    hero_arrays = np.stack([hand_str_to_ints(h["cards"]) for h in heroes])
    hole_count = hero_arrays.shape[1]
    game = detect_game_type(hole_count)
    score_array = get_score_array()

    deck = generate_deck_ints(list(dead_set) + [board[i:i + 2]
                                                for i in range(0, len(board), 2)])
    rng = np.random.default_rng(seed)

    # --- Pass 1: rank hands on shared runouts ---
    villain_hands = sample_villains(deck, hole_count, hands, rng)
    if villain_hands.shape[0] == 0:
        raise ValueError("no legal villain hands remain")

    rank_runouts_arr = sample_runouts(deck, board_len, rank_runouts, rng)
    strength, eligible_counts = rank_hands(
        villain_hands, board_ints, rank_runouts_arr, game, score_array,
        parallel=parallel, num_workers=num_workers,
    )

    # Drop hands that were never eligible on any shared runout (collided
    # with every one of them) -- their strength is meaningless, not 0.0 by
    # merit. See rank_hands' docstring; essentially impossible in practice.
    ranked = eligible_counts > 0
    villain_hands = villain_hands[ranked]
    strength = strength[ranked]
    if villain_hands.shape[0] == 0:
        raise ValueError("no ranked villain hands remain")

    buckets_list = list(buckets)
    bucket_indices, edges = _bucket_slices_and_edges(
        strength, villain_hands, buckets_list, board_ints, rank_runouts_arr,
    )

    # --- Pass 2: stratified hero equity on independent runouts ---
    trials_arr = sample_pass2_trials(
        deck, board_len, hole_count, villain_hands, bucket_indices,
        trials_per_bucket, rng,
    )
    _villain_scores, hero_results = evaluate_pass2(
        trials_arr, hero_arrays, board_ints, hole_count, game, score_array,
        parallel=parallel, num_workers=num_workers,
    )

    h = hero_arrays.shape[0]
    b = len(buckets_list)
    equity = hero_results.reshape(h, b, trials_per_bucket).mean(axis=2)

    ladders = []
    for j, hero in enumerate(heroes):
        ladders.append({
            "id": hero["id"],
            "rungs": build_rungs(buckets_list, edges, equity[j]),
        })

    return {
        "population": int(villain_hands.shape[0]),
        "exact": board_len == 5,
        "rank_runouts": int(rank_runouts_arr.shape[0]),
        "trials_per_bucket": int(trials_per_bucket),
        "ladders": ladders,
    }
