"""Postflop range ladder: hero equity vs board-strength percentile slices.

See docs/superpowers/plans/2026-08-13-postflop-range-ladder.md

Task 11 replaced the N-villains x R-shared-runouts grid with joint trial
sampling: one trial draws hole_count + need cards in a single shot, the
first hole_count go to the villain and the rest complete the board. See
task-11-brief.md for why -- in short, shared runouts made hero equity an
average over R board outcomes (precision governed by R alone, 5-7pp of
noise even at the whole-population bucket) and required an eligible_mask
skip-and-renormalise path to avoid villain/runout card collisions, which is
exactly the class of bug that produced a duplicate-card crash. Joint
sampling makes collisions impossible by construction (villain and runout
cards come from the same draw) and every per-trial number exact (a
completed 5-card board scores exactly, never an estimate).
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

# Task 11 chose T=350000 this way; Task 12 re-ran the identical sweep
# methodology (PLO6, the binding case for TIME; 16 seeds -- the brief's
# minimum, after this project already saw an 8-seed estimate read 11.12pp
# where 16 seeds read 38.99pp on the predecessor design) after routing
# score_hands through fast_score's numba kernel instead of
# hand_rank_evaluator's pure-numpy _batch_best_score (same math, no
# (N, C_h*C_b, 5) int32 temporary -- see task-12-brief.md). That is a pure
# throughput win, so the SAME 2.5s budget now buys ~4.3x more trials before
# hitting the wall. See bench_range_ladder.py and task-12-report.md for the
# full sweep table and the numpy-vs-numba speedup measurement.
#
# T=1500000: PLO6 mean 2.361s, max 2.411s, 0/16 seeds over the 2.5s budget.
# T=1600000 measured 9/16 seeds OVER budget (max 2.598s) in the same sweep
# -- the exact tail-latency risk Task 10 flagged, so (as in Task 11) the
# candidate is the largest one with EVERY seed's wall time under budget,
# not just the mean.
#
# Per-bucket spread at T=1500000, 16 seeds (pp = percentage points) -- ALL
# SIX BUCKETS now reach +/-1pp, including the 15% bucket that was Task 11's
# holdout (1.42pp there vs 0.47pp here):
#   bucket   5%: 0.20pp  REACHED
#   bucket  15%: 0.47pp  REACHED  <- was the binding bucket at T=350000
#   bucket  25%: 0.28pp  REACHED
#   bucket  40%: 0.17pp  REACHED
#   bucket  60%: 0.12pp  REACHED
#   bucket 100%: 0.07pp  REACHED
DEFAULT_TRIALS = 1500000

# Task 10: worker count for the persistent process pool (see
# multithread_ploequities3.get_global_executor). Sized to the box's core
# count rather than that module's own _DEFAULT_WORKERS=4 (tuned for a
# different, lighter-weight caller): this is the first thing in the app to
# create the pool, so it gets to pick the size. get_global_executor()
# ignores num_workers on every call after the first, so if some other path
# creates the pool first with a smaller count, evaluate_trials silently
# rides along on that smaller pool rather than failing -- correct, just
# less parallel.
DEFAULT_POOL_WORKERS = os.cpu_count() or 4

# Below this many total hand-evaluations ((1 + heroes) x trials),
# process-pool dispatch overhead (pickling villains/heroes/board per chunk,
# IPC round-trip) costs more than it saves. "A few thousand" per the Task 10
# brief; not re-measured precisely because every realistic caller (trials in
# the hundreds-to-tens-of-thousands range) sits far on one side of this line
# or the other -- see task-10-report.md.
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
    """Module-level (Windows-spawn-picklable, same rationale as
    `_evaluate_trials_chunk_worker`) warmup task run inside EVERY persistent
    pool worker process: forces the worker to spawn (same role the old
    `_dummy_warmup_task` no-op played) and, via fast_score.warmup(), triggers
    that worker's own numba JIT (or, with cache=True and the disk cache
    already populated by whichever process warmed first, a cache load
    instead of a fresh compile -- see task-12-report.md for the measurement
    confirming this actually happens rather than being assumed).

    Real evaluate_trials chunks land on these SAME worker processes, so
    warming only the main process (as fast_score.warmup() called directly
    would) leaves every worker's first real chunk paying a compile-or-
    cache-load stall the first time score_hands' numba path executes there.
    """
    fast_score.warmup()
    return 1


def warmup_pool():
    """Spawn and warm every worker of the persistent process pool, sized to
    DEFAULT_POOL_WORKERS, so the first real range-ladder request doesn't pay
    process-creation cost. Meant to be called once, from server.py's
    existing background warmup thread.

    Also warms fast_score's numba kernel in THIS (main) process first --
    compute_range_ladder's serial path (small `trials`, or `parallel=False`)
    runs score_hands in-process, never touching the pool -- and then in
    every pool worker via `_pool_worker_warmup_task`, so both the serial and
    parallel code paths are covered.

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

    Kept from the pre-Task-11 design (not called by compute_range_ladder
    any more -- see `sample_trials`, which reuses this function's
    argsort-of-random-keys distinct-draw trick but, being one independent
    Monte Carlo trial per row rather than a fixed population, does not need
    this function's cross-row dedup/rejection-sampling loop).
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


def sample_trials(deck, hole_count, board_len, trials, rng):
    """`trials` independent joint draws: villain hand + board completion.

    Each row draws `hole_count + need` DISTINCT cards from `deck`, where
    `need = 5 - board_len`. The first `hole_count` cards are the villain's
    hand; the remaining `need` are the runout that completes the board.
    Reuses sample_villains' argsort-of-random-keys trick for drawing k
    distinct cards from n in one vectorized shot -- but unlike
    sample_villains, rows here are NOT deduplicated against each other:
    these are `trials` independent Monte Carlo scenarios, not a fixed
    population, so two trials landing on the same cards is expected and
    unbiased, not a defect to reject.

    River (`board_len == 5`, `need == 0`): each row is just a villain hand,
    exactly `hole_count` cards -- the "board completion" is empty because
    there is nothing left to complete.

    Returns an empty (0, hole_count + need) array if the deck is smaller
    than what one trial needs (mirrors sample_villains' behavior for a
    too-small deck).
    """
    deck = np.asarray(deck, dtype=np.int32)
    total = len(deck)
    need = 5 - board_len
    num_cards = hole_count + need
    if total < num_cards:
        return np.empty((0, num_cards), dtype=np.int32)
    draw = rng.random((trials, total)).argsort(axis=1)[:, :num_cards]
    return deck[draw].astype(np.int32)


def score_hands(hands, board5, game, score_array, chunk=2000, use_numba=None):
    """Best 5-card score for each hand. Higher = better.

    `board5` is either a single board shared by every hand, shape (5,), or
    a per-hand board, shape (hands.shape[0], 5) -- one row per hand, as
    `evaluate_trials` needs: every trial completes the board with its own
    runout, so hand i must be scored against board5[i], not one board
    shared across the whole call.

    Task 12: the actual per-chunk scoring is `fast_score.batch_best_score`,
    a numba kernel that does the same work as the numpy reference path
    (`hand_rank_evaluator._batch_best_score`) with none of its (N, C_h*C_b,
    5) temporaries. `use_numba` passes straight through to that dispatcher
    (None -> module-level default, True/False -> force a path) -- exposed
    here, not just in fast_score, so tests/test_fast_score.py can force
    either path through the SAME call site range_ladder actually uses,
    rather than only at the raw kernel level.
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


def _evaluate_trials_chunk(trials_arr, heroes, board_ints, hole_count, game, score_array):
    """Score one CONTIGUOUS chunk of trial rows: exact villain score and
    every hero's result, per trial, on that trial's own completed board.

    Pure, no globals touched besides reading `score_array` -- this is the
    whole per-trial computation `evaluate_trials` used to run directly; it
    is now shared by the serial path (called in-process with the caller's
    own score_array) and the parallel worker entrypoint below (which
    supplies its own copy via get_score_array()). No behavior change
    between the two paths.

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
        # River: the board is already complete and identical for every
        # trial -- pass it as the single shared (5,) board so score_hands
        # broadcasts it, rather than materializing a redundant (T, 5) copy.
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


def _evaluate_trials_chunk_worker(trials_arr, heroes, board_ints, hole_count, game):
    """Module-level entrypoint submitted to the persistent process pool.

    Deliberately does NOT take score_array as a parameter. ProcessPoolExecutor
    pickles every argument across the process boundary; score_array is a
    ~2.6M-element float64 array (~21MB), and shipping it on every dispatch
    would dwarf the actual per-chunk computation -- exactly the "silent
    performance killer" flagged in the Task 10 brief. Instead each worker
    calls get_score_array(), which np.load()s score_array.npy once per
    process and caches it at module scope (hand_rank_evaluator._SCORE_ARRAY);
    the cost is paid once per worker's lifetime, not once per chunk.

    Must stay at module level (not a closure inside evaluate_trials):
    Windows' spawn start method pickles a reference to this function by
    qualified name and re-imports `range_ladder` in the child process to
    resolve it, which only works for names reachable at import time.
    """
    return _evaluate_trials_chunk(trials_arr, heroes, board_ints, hole_count,
                                  game, get_score_array())


def _split_trials(trials_arr, num_chunks):
    """Contiguous, order-preserving split into at most `num_chunks` pieces.

    Clamped so num_chunks never exceeds the row count -- np.array_split would
    otherwise hand back empty trailing chunks, which is wasted dispatch, not
    a correctness problem (an empty chunk just contributes nothing).
    """
    num_chunks = max(1, min(int(num_chunks), trials_arr.shape[0]))
    return np.array_split(trials_arr, num_chunks, axis=0)


def evaluate_trials(trials_arr, heroes, board_ints, hole_count, game, score_array,
                    parallel=None, num_workers=None):
    """The single O(T x (1 + H)) pass over every trial. Returns
    (villain_scores (T,), hero_results (H, T)).

    villain_scores: each trial's exact villain best-five score (never an
        estimate -- the board is complete and the evaluator is exact).
    hero_results: hero_results[j, i] is hero j's result (1.0 win, 0.5 tie,
        0.0 loss) against trial i's villain, scored on trial i's OWN
        completed board.

    Parallelised over trial chunks: unlike the predecessor evaluate_population
    (which additively accumulated beat_sum/beat_cnt/hero_sum across shared
    runouts, so summation ORDER mattered for float-exact reproducibility),
    every trial here is independent and exact on its own -- there is no
    reduction across trials at this layer, only a split-then-concatenate.
    Chunks are reassembled by np.concatenate in chunk-SUBMISSION order
    (never `as_completed()`), which reproduces the exact same per-trial
    numbers regardless of how many workers ran or which one finished first:
    concatenation does not reorder any floating-point operation the way
    summation would, so parallel output is not merely close to serial
    output within a tolerance, it is bit-identical.

    `parallel`: None (default) auto-decides from population size -- below
    `_PARALLEL_MIN_EVALS` total hand-evaluations, or a single trial (which
    cannot be split), the pool overhead isn't worth it and this runs the
    single-process loop unchanged. Pass True/False to force one path or the
    other (used by the parallel-equals-serial and worker-count-invariance
    tests, which need identical inputs on both paths).
    `num_workers`: chunk count for the parallel path (defaults to
    DEFAULT_POOL_WORKERS). Does not resize the process pool itself if it was
    already created with a different count elsewhere -- see get_global_executor.
    """
    t = trials_arr.shape[0]
    h = heroes.shape[0]

    if parallel is None:
        n_evals = (1 + h) * t
        parallel = t > 1 and n_evals >= _PARALLEL_MIN_EVALS

    if parallel and t > 1:
        workers = num_workers or DEFAULT_POOL_WORKERS
        executor = get_global_executor(workers)
        chunks = _split_trials(trials_arr, workers)
        futures = [executor.submit(_evaluate_trials_chunk_worker,
                                   chunk, heroes, board_ints, hole_count, game)
                   for chunk in chunks]
        # .result() in chunk-submission order, not as_completed(): see the
        # determinism note in the docstring above.
        partials = [f.result() for f in futures]
        villain_scores = np.concatenate([p[0] for p in partials])
        hero_results = np.concatenate([p[1] for p in partials], axis=1)
    else:
        villain_scores, hero_results = _evaluate_trials_chunk(
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


def build_rungs(villain_scores, hero_results_row, villain_hands, runouts, buckets, board_ints):
    """One rung per bucket: hero equity vs that slice + the slice's weakest
    trial's villain hand, labelled on THAT SAME TRIAL's own completed board.

    Ranks trials by villain_scores alone (strongest first) -- hero-
    independent by construction: the same ranking, the same slices, and the
    same edge trial are used for every hero, only `hero_results_row`
    differs per call.

    Task 11: no eligibility/collision check is needed here (the predecessor
    `_compatible_board5` helper, and the whole "does the boundary hand
    collide with the runout used to label it" problem, are gone). A trial's
    villain hand and its runout are drawn from the same distinct 7-or-8-card
    pull, so they can never share a card -- the completed board used to
    label the edge hand is that exact trial's own board, not a runout
    borrowed from elsewhere.
    """
    order = np.argsort(-villain_scores, kind="stable")   # strongest first
    n = order.size
    rungs = []
    for pct in buckets:
        take = max(1, int(round(n * pct / 100.0)))
        sl = order[:take]
        edge_idx = int(sl[-1])                            # weakest trial in the slice
        hand = villain_hands[edge_idx]
        runout = runouts[edge_idx]
        board5 = (np.concatenate([board_ints, runout]) if runout.size
                 else np.asarray(board_ints)).astype(np.int32)
        rungs.append({
            "bucket": pct,
            "equity": float(hero_results_row[sl].mean()),
            "edge": {
                "cards": ints_to_hand_str(hand),
                "category": describe_category(hand, board5),
            },
        })
    return rungs


def compute_range_ladder(board, dead, heroes, buckets=DEFAULT_BUCKETS,
                         trials=DEFAULT_TRIALS, seed=None,
                         parallel=None, num_workers=None):
    """The whole feature: hero equity vs board-strength percentile slices.

    `parallel`/`num_workers` pass straight through to evaluate_trials (see
    its docstring); left at their defaults, sizing auto-decides from the
    trial count, which is the right choice for normal callers. Exposed here
    mainly for tests that need to force one path or a specific worker count
    while holding every other input fixed.

    Task 11: `trials` (default DEFAULT_TRIALS) replaces the old `hands` x
    `runouts` grid entirely -- each of the `trials` Monte Carlo draws is one
    complete, independent scenario (villain hand + board completion), not a
    villain sampled once and then re-scored against a shared pool of
    runouts. `rng` is used only for `sample_trials`.

    See spec Section 5 for the response shape (minus `runouts`, which no
    longer has a meaning under joint sampling -- see task-11-report.md).
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

    trials_arr = sample_trials(deck, hole_count, board_len, trials, rng)
    if trials_arr.shape[0] == 0:
        raise ValueError("no legal trials remain")

    villain_hands = trials_arr[:, :hole_count]
    runouts = trials_arr[:, hole_count:]

    villain_scores, hero_results = evaluate_trials(
        trials_arr, hero_arrays, board_ints, hole_count, game, score_array,
        parallel=parallel, num_workers=num_workers,
    )

    ladders = []
    for j, hero in enumerate(heroes):
        ladders.append({
            "id": hero["id"],
            "rungs": build_rungs(villain_scores, hero_results[j], villain_hands,
                                 runouts, list(buckets), board_ints),
        })

    return {
        "population": int(trials_arr.shape[0]),
        "exact": board_len == 5,
        "ladders": ladders,
    }
