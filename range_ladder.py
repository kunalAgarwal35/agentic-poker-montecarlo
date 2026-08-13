"""Postflop range ladder: hero equity vs board-strength percentile slices.

See docs/superpowers/plans/2026-08-13-postflop-range-ladder.md
"""
import os
from itertools import combinations

import numpy as np

from card_encoding import generate_deck_ints, hand_str_to_ints, ints_to_hand_str
from hand_categories import CATEGORY_TOKENS
from hand_indexing import BINOMIAL
from hand_rank_evaluator import (
    _BOARD_COMBOS,
    _HAND_COMBOS,
    _batch_best_score,
    detect_game_type,
    get_score_array,
)
from multithread_ploequities3 import _dummy_warmup_task, get_global_executor
from optimized_evaluator import best5_category_omaha_numba, get_category_array

DEFAULT_BUCKETS = (5, 15, 25, 40, 60, 100)

# Task 10 SUPERSEDES the Task 8 values below (25/10000), which were chosen
# against single-threaded timings. With evaluate_population parallelised
# over multithread_ploequities3's persistent process pool (measured
# DEFAULT_POOL_WORKERS=24 on this box: 12 physical / 24 logical hyperthreaded
# cores -- see task-10-report.md for the full scaling curve), PLO6 throughput
# went from ~113k-115k hand-evals/sec (Task 8, single-threaded) to ~590k-600k/
# sec (measured directly: hands=10000, runouts=200 -> n_evals=2,000,200 in
# ~3.3-3.4s at 24 workers). That is only a ~5.2-5.3x speedup, not the ~24x a
# naive reading of "24 cores" suggests -- hyperthreading gives little extra
# throughput on this CPU-bound numeric workload past the 12 physical cores,
# confirmed by the scaling curve plateauing hard between 12 and 24 workers
# (12: ~5.0x, 16: ~5.1x, 24: ~5.2-5.3x -- essentially flat past 12).
#
# Re-swept N (hands) x R (runouts) for PLO6 -- the binding case, per Task
# 8's own finding, reconfirmed here (PLO4 measured 0.75-1.4s at the chosen
# N/R, comfortably under budget) -- across the new ~590k-evals/sec budget
# boundary, then measured run-to-run spread (max equity spread across
# seeds, per bucket) at up to 24 seeds (1-8, 21-28, 41-48; the brief flags
# that a 3-seed estimate produced a misleading conclusion earlier in this
# project -- one data point in this very sweep reproduced that lesson: the
# old R=25 default's worst-bucket spread measured 11.12pp at 8 seeds but
# 38.99pp once seeds 21-28 were added, a >3x swing from adding more seeds
# to an already->8-seed sample).
#
# A second, unplanned finding drove the final choice as much as accuracy
# did: parallel dispatch has real tail latency that the old single-process
# loop never had (the wall-clock time of N chunks run across W workers is
# set by the SLOWEST chunk, a straggler effect). Every N/R candidate with a
# ~2.0-2.3s mean (R in roughly 110-350 at N in roughly 4000-10000) showed at
# least one run spike to 2.6-2.9s in a 12-to-24-trial timing sample, despite
# looking perfectly safe in a smaller sample -- e.g. N=8000/R=150 showed
# 0/12 over budget in one batch and 1/24 in the next. Only N=10000/R=100
# (mean ~1.73-1.94s) showed zero overshoots across 24+16 combined trials
# (max observed 1.78-2.59s -- even its own edges are not perfectly clean,
# but its margin was consistently the widest of everything tried). Per
# Task 8's own decision rule ("the budget wins"), this -- not the single
# best worst-bucket number found -- is what got shipped.
#
#   N=10000 R=25   (old default)   16 seeds: worst 38.99pp  100%-bucket 15.28pp
#   N=10000 R=100  (chosen)        16 seeds: worst 11.24pp  100%-bucket  5.37pp
#                                  24 seeds: worst 12.84pp  100%-bucket  6.81pp
#                                  mean 1.73-1.94s, max 1.78-2.59s, 0/24 over 2.5s
#   N=8000  R=150                  16 seeds: worst  6.79pp  100%-bucket  3.56pp
#                                  (better spread, but 1 timing overshoot in 24 trials)
#   N=4000  R=350                  16 seeds: worst  6.21pp  100%-bucket  2.77pp
#                                  (best spread found; 1-2 overshoots (up to 2.92s)
#                                  across 24-36 trials -- rejected on budget risk)
#
# R (runouts) reduces whole-population noise; N (hands) sets how many hands
# land in the thinnest bucket (top 5% = N*0.05 -- only 500 hands even at
# N=10000). The R=150-350 candidates score better on spread specifically
# because they let R grow further -- but only by shrinking N or eating
# further into the pool's tail-latency margin, and either one reintroduces
# budget risk this box couldn't absorb reliably.
#
# +/-1pp is NOT reached: worst-bucket spread is ~11-13pp against the 1pp
# target, still an order of magnitude off (down from the old default's true,
# 16-seed-measured ~39pp -- Task 8's own headline 11.12pp number for R=25
# undersold how bad it really was, for exactly the too-few-seeds reason the
# Task 10 brief warns about). The whole-population (100%) bucket is the
# least N-confounded number here: at ~6pp today, extrapolating the observed
# roughly-1/sqrt(R) scaling implies needing R on the order of 3000-4000 (at
# N=10000) to approach 1pp there alone, i.e. very roughly another ~20-30x
# throughput beyond what this task's pool already delivers -- and the
# top-5% bucket, being N-limited rather than purely R-limited, likely needs
# more than that on top. This is an order-of-magnitude extrapolation, not a
# separately measured number; closing the gap for real needs either
# substantially more raw throughput (faster hardware, more real cores, or an
# algorithmic variance-reduction technique such as stratified/control-variate
# sampling instead of brute-force resampling) or a relaxed accuracy target.
# Equities should still be rounded to the nearest whole percent for display;
# true run-to-run noise remains well above fractional-percent precision.
DEFAULT_RUNOUTS = 100

# Paired with DEFAULT_RUNOUTS above -- see that comment for the full N x R
# sweep and the tail-latency finding that drove this choice. Unchanged from
# Task 8's 10000: within the parallel pool's SAFE eval budget (the ceiling
# before per-call tail latency starts risking the 2.5s cutoff), keeping N at
# 10000 -- the largest top-5%-bucket hand count tried -- while raising R
# 4x (25 -> 100) gave the best spread/budget-safety combination found.
DEFAULT_HANDS = 10000

# Task 10: worker count for the persistent process pool (see
# multithread_ploequities3.get_global_executor). Sized to the box's core
# count rather than that module's own _DEFAULT_WORKERS=4 (tuned for a
# different, lighter-weight caller): this is the first thing in the app to
# create the pool, so it gets to pick the size. get_global_executor()
# ignores num_workers on every call after the first, so if some other path
# creates the pool first with a smaller count, evaluate_population silently
# rides along on that smaller pool rather than failing -- correct, just
# less parallel.
DEFAULT_POOL_WORKERS = os.cpu_count() or 4

# Below this many total hand-evaluations (villains + heroes, times runouts),
# process-pool dispatch overhead (pickling villains/heroes/board per chunk,
# IPC round-trip) costs more than it saves. "A few thousand" per the Task 10
# brief; not re-measured precisely because every realistic caller (hands in
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


def warmup_pool():
    """Spawn and warm every worker of the persistent process pool, sized to
    DEFAULT_POOL_WORKERS, so the first real range-ladder request doesn't pay
    process-creation cost. Meant to be called once, from server.py's
    existing background warmup thread.

    Deliberately does NOT call multithread_ploequities3.warmup_executor():
    that function hardcodes _DEFAULT_WORKERS (4 -- sized for a different,
    lighter-weight caller, see the DEFAULT_POOL_WORKERS comment above) for
    both pool creation *and* for how many dummy tasks it submits, so it
    would warm only 4 of this box's cores even if the pool were already
    sized larger. This calls the same underlying interface
    (get_global_executor, the same _dummy_warmup_task used elsewhere) at
    DEFAULT_POOL_WORKERS instead -- still the one persistent pool, not a
    second one.
    """
    executor = get_global_executor(DEFAULT_POOL_WORKERS)
    futures = [executor.submit(_dummy_warmup_task) for _ in range(DEFAULT_POOL_WORKERS)]
    for f in futures:
        f.result()


def sample_villains(deck, num_cards, n, rng):
    """Distinct legal villain hands drawn from `deck`, as sorted rows.

    Falls back to exhaustive enumeration when the space is smaller than `n`,
    so tiny decks return every hand exactly once instead of looping forever.
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
    """`r` random completions of the board. River -> a single empty runout."""
    deck = np.asarray(deck, dtype=np.int32)
    need = 5 - board_len
    if need <= 0:
        return np.empty((1, 0), dtype=np.int32)
    draw = rng.random((r, len(deck))).argsort(axis=1)[:, :need]
    return deck[draw].astype(np.int32)


def eligible_mask(villains, runout):
    """False for villains holding a card that the runout also uses.

    Those pairings are impossible and must never be scored. Skipping them
    evaluates each villain over exactly the runouts compatible with it, which
    is the correct conditional distribution -- unbiased. See spec 4.3.
    """
    if runout.size == 0:
        return np.ones(villains.shape[0], dtype=bool)
    return ~np.isin(villains, runout).any(axis=1)


def score_hands(hands, board5, game, score_array, chunk=2000):
    """Best 5-card score for each hand on one complete board. Higher = better."""
    hand_combos = _HAND_COMBOS[game]
    n = hands.shape[0]
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk):
        block = hands[start:start + chunk]
        boards = np.broadcast_to(board5, (block.shape[0], 5))
        out[start:start + chunk] = _batch_best_score(
            block, np.ascontiguousarray(boards),
            hand_combos, _BOARD_COMBOS, score_array, BINOMIAL,
        )
    return out


def _evaluate_runout_chunk(villains, heroes, board_ints, runouts, game, score_array):
    """Sum beat_sum/beat_cnt/hero_sum contributions over one CONTIGUOUS chunk
    of runouts. Pure, no globals touched besides reading `score_array` -- this
    is the whole per-runout loop `evaluate_population` used to run directly;
    it is now shared by the serial path (called in-process with the caller's
    own score_array) and the parallel worker entrypoint below (which supplies
    its own copy via get_score_array()). No behavior change from before.
    """
    n = villains.shape[0]
    h = heroes.shape[0]

    beat_sum = np.zeros(n, dtype=np.float64)     # share of field beaten, summed
    beat_cnt = np.zeros(n, dtype=np.float64)     # runouts this villain was eligible for
    hero_sum = np.zeros((h, n), dtype=np.float64)

    for runout in runouts:
        board5 = np.concatenate([board_ints, runout]).astype(np.int32)
        mask = eligible_mask(villains, runout)
        idx = np.flatnonzero(mask)
        if idx.size < 2:
            # `strength` needs at least 2 eligible villains -- with idx.size == 1
            # the "share of field beaten" denominator (idx.size - 1) is zero,
            # and with idx.size == 0 there's no field at all. Skipping here is
            # correct for strength, but it also throws away hero-vs-villain
            # equity for this runout, which IS well defined for a single
            # eligible villain. At N=10000 the chance of <2 eligible villains
            # on a runout is essentially zero; at N==1 every runout hits this
            # guard and hero_equity comes back silently all-zero instead of
            # the correct heads-up number. Degenerate for tiny populations --
            # not restructured here, flagged for awareness.
            continue

        vs = score_hands(villains[idx], board5, game, score_array)
        hs = score_hands(heroes, board5, game, score_array)

        # Share of the eligible field each villain beats: win + 1/2 tie.
        order = np.sort(vs)
        lower = np.searchsorted(order, vs, side="left")          # strictly worse
        upper = np.searchsorted(order, vs, side="right")
        ties = upper - lower - 1                                  # excluding self
        share = (lower + 0.5 * ties) / (idx.size - 1)
        beat_sum[idx] += share
        beat_cnt[idx] += 1.0

        # Hero vs each eligible villain, heads-up.
        for j in range(h):
            hero_sum[j, idx] += np.where(hs[j] > vs, 1.0, np.where(hs[j] == vs, 0.5, 0.0))

    return beat_sum, beat_cnt, hero_sum


def _evaluate_runout_chunk_worker(villains, heroes, board_ints, runouts, game):
    """Module-level entrypoint submitted to the persistent process pool.

    Deliberately does NOT take score_array as a parameter. ProcessPoolExecutor
    pickles every argument across the process boundary; score_array is a
    ~2.6M-element float64 array (~21MB), and shipping it on every dispatch
    would dwarf the actual per-chunk computation -- exactly the "silent
    performance killer" flagged in the Task 10 brief. Instead each worker
    calls get_score_array(), which np.load()s score_array.npy once per
    process and caches it at module scope (hand_rank_evaluator._SCORE_ARRAY);
    the cost is paid once per worker's lifetime, not once per chunk.

    Must stay at module level (not a closure inside evaluate_population):
    Windows' spawn start method pickles a reference to this function by
    qualified name and re-imports `range_ladder` in the child process to
    resolve it, which only works for names reachable at import time.
    """
    return _evaluate_runout_chunk(villains, heroes, board_ints, runouts, game,
                                  get_score_array())


def _split_runouts(runouts, num_chunks):
    """Contiguous, order-preserving split into at most `num_chunks` pieces.

    Clamped so num_chunks never exceeds the row count -- np.array_split would
    otherwise hand back empty trailing chunks, which is wasted dispatch, not
    a correctness problem (an empty chunk just contributes zeros).
    """
    num_chunks = max(1, min(int(num_chunks), runouts.shape[0]))
    return np.array_split(runouts, num_chunks, axis=0)


def evaluate_population(villains, heroes, board_ints, runouts, game, score_array,
                        parallel=None, num_workers=None):
    """The single O((N + H) x R) pass. Returns (strength, hero_equity, counts).

    strength: (N,) float64 -- each villain's mean share of the eligible field
        it beats (win + 1/2 tie), i.e. its equity against the population.
        NaN-free: villains eligible on zero counted runouts get 0.0.
    hero_equity: (H, N) float64 -- hero h's equity against villain i, averaged
        over the runouts counted for i.
    counts: (N,) float64 -- the number of runouts that actually contributed to
        villain i's strength/hero_equity (i.e. i was eligible AND the runout
        had at least 2 eligible villains). This is a strict subset of "i was
        eligible on this runout" -- see the guard in `_evaluate_runout_chunk`
        -- so callers that need to know whether a villain has any real data
        must check `counts`, not re-derive eligibility themselves.

    Parallelised over runout chunks (Task 10): each runout contributes
    additively to beat_sum/beat_cnt/hero_sum and nothing crosses runouts, so
    the runouts array is split into contiguous chunks, one per worker, run on
    multithread_ploequities3's persistent process pool, and the partials are
    summed back together IN CHUNK-INDEX ORDER (never `as_completed()` order:
    float addition is not associative, so summing in completion order would
    make results depend on worker scheduling timing and silently break
    determinism run to run).

    `parallel`: None (default) auto-decides from population size -- below
    `_PARALLEL_MIN_EVALS` total hand-evaluations, or when there's only one
    runout (the river; a single row cannot be split), the pool overhead
    isn't worth it and this runs the old single-process loop unchanged. Pass
    True/False to force one path or the other (used by the parallel-equals-
    serial and worker-count-invariance tests, which need identical inputs on
    both paths).
    `num_workers`: chunk count for the parallel path (defaults to
    DEFAULT_POOL_WORKERS). Does not resize the process pool itself if it was
    already created with a different count elsewhere -- see get_global_executor.
    """
    n = villains.shape[0]
    h = heroes.shape[0]
    r = runouts.shape[0]

    if parallel is None:
        n_evals = (n + h) * r
        parallel = r > 1 and n_evals >= _PARALLEL_MIN_EVALS

    if parallel and r > 1:
        workers = num_workers or DEFAULT_POOL_WORKERS
        executor = get_global_executor(workers)
        chunks = _split_runouts(runouts, workers)
        futures = [executor.submit(_evaluate_runout_chunk_worker,
                                   villains, heroes, board_ints, chunk, game)
                   for chunk in chunks]
        # .result() in chunk-submission order, not as_completed(): see the
        # determinism note in the docstring above.
        partials = [f.result() for f in futures]
        beat_sum = sum(p[0] for p in partials)
        beat_cnt = sum(p[1] for p in partials)
        hero_sum = sum(p[2] for p in partials)
    else:
        beat_sum, beat_cnt, hero_sum = _evaluate_runout_chunk(
            villains, heroes, board_ints, runouts, game, score_array
        )

    seen = beat_cnt > 0
    strength = np.zeros(n, dtype=np.float64)
    strength[seen] = beat_sum[seen] / beat_cnt[seen]
    hero_equity = np.zeros((h, n), dtype=np.float64)
    hero_equity[:, seen] = hero_sum[:, seen] / beat_cnt[seen]
    return strength, hero_equity, beat_cnt


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
    """Board built from `board_ints` + the first sampled runout `hand` is
    eligible for (no shared card), or None if every sampled runout collides.

    describe_category feeds `hand` + this board straight into a
    bounds-unchecked numba kernel; a runout sharing a card with `hand`
    produces a 9-card set with a duplicate, which crashes the process
    (access violation), not a Python exception. Reuses `eligible_mask` --
    the same collision check evaluate_population uses for scoring -- rather
    than re-deriving it, just applied to a single hand instead of the whole
    villain population.
    """
    hand2d = hand[None, :]
    for runout in runout_rows:
        if eligible_mask(hand2d, runout)[0]:
            return np.concatenate([board_ints, runout]).astype(np.int32)
    return None


def build_rungs(strength, hero_equity_row, villains, buckets, board_ints, runout_rows):
    """One rung per bucket: hero equity vs that slice + the slice's weakest hand.

    Each boundary hand is labelled on one sampled runout it is actually
    compatible with (see `_compatible_board5`), not a single runout shared
    across every bucket -- a shared runout can hold a card the boundary
    hand also holds, which is an impossible 9-card set for describe_category
    to score. On the river, `runout_rows` is the single empty completion, so
    every hand is trivially compatible with it and this is a no-op.
    """
    order = np.argsort(-strength, kind="stable")     # strongest first
    n = order.size
    rungs = []
    for pct in buckets:
        take = max(1, int(round(n * pct / 100.0)))
        sl = order[:take]
        edge_idx = int(sl[-1])                        # weakest hand in the slice
        hand = villains[edge_idx]
        board5 = _compatible_board5(board_ints, runout_rows, hand)
        # Essentially impossible with any real sample size, but never
        # assumed: an empty label is a cosmetic gap, a duplicate card fed
        # to the numba kernel is a process crash.
        category = describe_category(hand, board5) if board5 is not None else ""
        rungs.append({
            "bucket": pct,
            "equity": float(hero_equity_row[sl].mean()),
            "edge": {
                "cards": ints_to_hand_str(hand),
                "category": category,
            },
        })
    return rungs


def compute_range_ladder(board, dead, heroes, buckets=DEFAULT_BUCKETS,
                         hands=DEFAULT_HANDS, runouts=None, seed=None,
                         parallel=None, num_workers=None):
    """The whole feature: hero equity vs board-strength percentile slices.

    `parallel`/`num_workers` pass straight through to evaluate_population
    (see its docstring); left at their defaults, sizing auto-decides from
    the population, which is the right choice for normal callers. Exposed
    here mainly for tests that need to force one path or a specific worker
    count while holding every other input fixed.

    See spec Section 5 for the response shape.
    """
    board_ints = hand_str_to_ints(board)
    board_len = len(board_ints)
    if board_len not in (3, 4, 5):
        raise ValueError(f"board must be 3, 4 or 5 cards, got {board_len}")

    # Zero (or negative) runouts is only meaningful on the river, where the
    # count is 0 by construction and sample_runouts already returns the
    # single empty completion. Off the river it must raise, not silently
    # produce a (0, need) runout array -- that starves evaluate_population's
    # per-runout loop entirely, leaving every villain/hero at a structural
    # 0.0 strength/equity, and then crashes downstream on runout_rows[0]
    # (IndexError: empty array). `runouts is None` (the "use DEFAULT_RUNOUTS"
    # case) is unaffected by this check.
    if runouts is not None and runouts <= 0 and board_len < 5:
        raise ValueError(
            f"runouts={runouts} is only meaningful on the river (5-card "
            f"board); board here has {board_len} cards. Omit `runouts` "
            f"(defaults to {DEFAULT_RUNOUTS}) or pass a positive count."
        )

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
    num_cards = hero_arrays.shape[1]
    game = detect_game_type(num_cards)
    score_array = get_score_array()

    deck = generate_deck_ints(list(dead_set) + [board[i:i + 2]
                                                for i in range(0, len(board), 2)])
    rng = np.random.default_rng(seed)

    villains = sample_villains(deck, num_cards, hands, rng)
    if villains.shape[0] == 0:
        raise ValueError("no legal villain hands remain")

    r = 1 if board_len == 5 else (runouts if runouts is not None else DEFAULT_RUNOUTS)
    runout_rows = sample_runouts(deck, board_len, r, rng)

    strength, hero_equity, counts = evaluate_population(
        villains, hero_arrays, board_ints, runout_rows, game, score_array,
        parallel=parallel, num_workers=num_workers,
    )

    # A villain that contributed to no runout carries no data; drop it rather
    # than letting a structural 0.0 masquerade as "weakest hand in the
    # population". Filter on `counts` -- the contribution count evaluate_population
    # actually used -- NOT on a recomputed eligible_mask union. The two differ on
    # runouts skipped by the <2-eligible guard, and that gap is exactly how an
    # artefact reaches the user as a boundary hand.
    seen = counts > 0
    villains, strength, hero_equity = villains[seen], strength[seen], hero_equity[:, seen]

    ladders = []
    for j, hero in enumerate(heroes):
        # Each bucket's boundary hand is labelled on one sampled runout that
        # *that hand* is compatible with (build_rungs / _compatible_board5),
        # not a single runout shared across every bucket and every hero --
        # a shared runout can hold a card the boundary hand also holds,
        # which is an impossible duplicate-card board for describe_category
        # to score. On the river, runout_rows is the single empty
        # completion, so every hand is trivially compatible and this is
        # unchanged from before.
        ladders.append({
            "id": hero["id"],
            "rungs": build_rungs(strength, hero_equity[j], villains, list(buckets),
                                 board_ints, runout_rows),
        })

    return {
        "population": int(villains.shape[0]),
        "runouts": 0 if board_len == 5 else int(runout_rows.shape[0]),
        "exact": board_len == 5,
        "ladders": ladders,
    }
