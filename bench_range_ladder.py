"""Task 11 re-measurement: joint trial sampling replaced the N (hands) x R
(runouts) grid entirely, so there is exactly one knob left to sweep --
`trials`. Cost is trials x (1 + heroes) hand-evaluations, not
hands x runouts x (1 + heroes), so this sweep runs at a small fraction of
the wall-clock the old N x R grid needed for the same trial count. See
range_ladder.py's DEFAULT_TRIALS comment and task-11-report.md for the
numbers this produced and how they were chosen.

MUST run under `if __name__ == "__main__":` (see bottom of this file), not
as bare module-level code: ProcessPoolExecutor uses Windows' spawn start
method, which re-imports this file's top-level code in every worker
process. Without the guard, each spawned worker would re-run the entire
sweep and spawn its own pool of workers -- unbounded recursive process
creation.

Scope trimmed to fit comfortably inside a ~25 minute run:
  - Scaling curve measured once, at one moderate workload -- how parallel
    speedup scales with worker count doesn't depend on which `trials` the
    accuracy sweep below eventually picks.
  - Only PLO6 gets the full accuracy sweep (single hero). PLO6 was already
    the binding case for TIME in every prior task on this feature; a direct
    check below (3 seeds, timing only, no spread claim) confirms PLO4 stays
    comfortably ahead at the chosen `trials`, so a second full accuracy
    sweep for PLO4 was skipped.
  - The candidate grid is a hand-picked set bracketing the 2.5s boundary
    (found via a separate interactive probe, not reproduced here) plus a
    few points well past it, included ONLY to answer "what T would close
    the +/-1pp gap" empirically rather than by extrapolation -- Task 11's
    per-trial cost is cheap enough that measuring those over-budget points
    directly is itself cheap, even though using them in production would
    not be.
"""
import time

import numpy as np

from range_ladder import compute_range_ladder, warmup_pool, DEFAULT_POOL_WORKERS

PLO6_DEAD = ["AsKs9h2c3d4d", "QhJhTd3s5s8s"]
PLO6_HEROES = [{"id": "h", "cards": "AsKs9h2c3d4d"}]
PLO4_DEAD = ["AsKs9h2c", "QhJhTd3d"]
PLO4_HEROES = [{"id": "h", "cards": "AsKs9h2c"}]
BOARD = "6s7s4s"
BUDGET_S = 2.5

# 16 seeds (the brief's minimum): the brief explicitly warns that an 8-seed
# spread estimate produced a confidently wrong conclusion earlier in this
# project (11.12pp at 8 seeds vs 38.99pp at 16, same fixture, Task 10), so
# the number that actually gets shipped in DEFAULT_TRIALS uses at least
# that many, split 8+8 across two non-adjacent seed bands rather than one
# contiguous run of 16, in case of any seed-adjacency correlation.
SPREAD_SEEDS = tuple(range(1, 9)) + tuple(range(21, 29))

# Brackets the ~2.5s boundary (found via a separate interactive timing probe
# before this file was finalized -- see task-11-report.md for that probe's
# output): 300k-400k is the bracket, budget wins inside it. The candidates
# past 400k are ONLY here to measure how much larger T needs to get before
# every bucket's spread falls under +/-1pp -- they are known in advance to
# be over budget and are never candidates for DEFAULT_TRIALS.
CANDIDATES = [10000, 50000, 100000, 200000, 300000, 350000, 400000]
OVER_BUDGET_PROBES = [700000, 1400000, 2800000]


def _warm():
    warmup_pool()
    # Throwaway call: pays the one-time numba JIT-compile cost so it
    # doesn't land on whichever timed call happens to run first.
    compute_range_ladder(board=BOARD, dead=PLO6_DEAD, heroes=PLO6_HEROES,
                         trials=200, seed=0)


def measure_scaling():
    """How parallel speedup scales with worker count, PLO6, a fixed
    moderate workload (trials=200000 -> 200,000 hand-evaluations for a
    single hero, big enough that dispatch overhead is a small fraction of
    total time)."""
    print(f"\n=== Scaling with worker count (DEFAULT_POOL_WORKERS={DEFAULT_POOL_WORKERS}) ===")
    trials = 200000
    t0 = time.perf_counter()
    compute_range_ladder(board=BOARD, dead=PLO6_DEAD, heroes=PLO6_HEROES,
                         trials=trials, seed=1, parallel=False)
    serial_t = time.perf_counter() - t0
    print(f"serial:            {serial_t:6.3f}s")

    for workers in (1, 2, 4, 8, 12, 16, 24):
        if workers > DEFAULT_POOL_WORKERS:
            continue
        t0 = time.perf_counter()
        compute_range_ladder(board=BOARD, dead=PLO6_DEAD, heroes=PLO6_HEROES,
                             trials=trials, seed=1,
                             parallel=True, num_workers=workers)
        t = time.perf_counter() - t0
        print(f"parallel workers={workers:3d}: {t:6.3f}s  speedup={serial_t / t:5.2f}x")


def probe_budget_boundary():
    """Quick single-seed timing-only probe (PLO6) confirming CANDIDATES
    brackets the 2.5s boundary."""
    print("\n=== Budget-boundary probe (PLO6, single seed, timing only) ===")
    print(f"{'trials':>9} {'n_evals':>10} {'time(s)':>8}")
    for t in CANDIDATES:
        t0 = time.perf_counter()
        compute_range_ladder(board=BOARD, dead=PLO6_DEAD, heroes=PLO6_HEROES,
                             trials=t, seed=1)
        elapsed = time.perf_counter() - t0
        flag = "  OVER BUDGET" if elapsed > BUDGET_S else ""
        print(f"{t:9d} {t * 2:10d} {elapsed:8.3f}{flag}")


def _spread_row(t, seeds=SPREAD_SEEDS):
    """(trials, mean_t, max_t, spread_pp array, over_budget_count) for one
    candidate, PLO6, across `seeds`."""
    eqs, times = [], []
    for seed in seeds:
        t0 = time.perf_counter()
        out = compute_range_ladder(board=BOARD, dead=PLO6_DEAD, heroes=PLO6_HEROES,
                                   trials=t, seed=seed)
        times.append(time.perf_counter() - t0)
        eqs.append([x["equity"] for x in out["ladders"][0]["rungs"]])
    eqs = np.array(eqs)
    spread_pp = np.ptp(eqs, axis=0) * 100
    mean_t, max_t = float(np.mean(times)), float(np.max(times))
    over = sum(1 for x in times if x > BUDGET_S)
    return mean_t, max_t, spread_pp, over


def sweep_accuracy():
    """The real re-measurement: elapsed time and per-bucket run-to-run
    spread (max equity spread across SPREAD_SEEDS, 16 seeds) for each
    candidate, PLO6. Picks and prints the largest candidate that stayed
    under budget across EVERY seed (not just the mean), per the Task 10
    lesson that a mean well under budget can still hide occasional
    tail-latency spikes over it."""
    buckets = None
    print(f"\n=== Accuracy sweep (PLO6, {len(SPREAD_SEEDS)} seeds) ===")
    results = []
    for t in CANDIDATES:
        mean_t, max_t, spread_pp, over = _spread_row(t)
        if buckets is None:
            out = compute_range_ladder(board=BOARD, dead=PLO6_DEAD, heroes=PLO6_HEROES,
                                       trials=200, seed=0)
            buckets = [r["bucket"] for r in out["ladders"][0]["rungs"]]
            header = "  ".join(f"{b:>4}%" for b in buckets)
            print(f"{'trials':>9} {'mean_t':>8} {'max_t':>8} {'over':>6}  [{header}]")
        results.append((t, mean_t, max_t, spread_pp, over))
        buckets_str = "  ".join(f"{x:5.2f}" for x in spread_pp)
        print(f"{t:9d} {mean_t:8.3f} {max_t:8.3f} {over:5d}/{len(SPREAD_SEEDS)}  [{buckets_str}]")

    under_budget = [row for row in results if row[2] <= BUDGET_S]
    if not under_budget:
        print("\nNo candidate stayed under budget across every seed -- widen CANDIDATES.")
        return results, buckets

    chosen = max(under_budget, key=lambda row: row[0])
    t, mean_t, max_t, spread_pp, over = chosen
    print(f"\nChosen: trials={t} (mean {mean_t:.3f}s, max {max_t:.3f}s, "
         f"0/{len(SPREAD_SEEDS)} over {BUDGET_S}s budget)")
    print("Per-bucket +/-1pp verdict:")
    for b, sp in zip(buckets, spread_pp):
        verdict = "REACHED" if sp <= 1.0 else "NOT reached"
        print(f"  bucket {b:>4}%: spread {sp:5.2f}pp -- {verdict}")
    return results, buckets


def probe_gap_to_one_pp(buckets):
    """Measure spread at several deliberately OVER-BUDGET trial counts, to
    answer empirically (not by extrapolation) what T would be needed to
    close the +/-1pp gap on every bucket, and to show the trend as T grows.
    None of these are candidates for DEFAULT_TRIALS -- they exist only to
    answer the brief's "what T would it take" question with real numbers.
    """
    print(f"\n=== Gap-to-+/-1pp probe (PLO6, over-budget T, {len(SPREAD_SEEDS)} seeds) ===")
    header = "  ".join(f"{b:>4}%" for b in buckets)
    print(f"{'trials':>9} {'mean_t':>8} {'max_t':>8}  [{header}]")
    for t in OVER_BUDGET_PROBES:
        mean_t, max_t, spread_pp, _over = _spread_row(t)
        buckets_str = "  ".join(f"{x:5.2f}" for x in spread_pp)
        print(f"{t:9d} {mean_t:8.3f} {max_t:8.3f}  [{buckets_str}]")
        if spread_pp.max() <= 1.0:
            print(f"  -> every bucket under +/-1pp at trials={t} (not budget-feasible: "
                 f"{mean_t:.2f}s mean vs {BUDGET_S}s budget)")


def check_plo4_stays_ahead(t):
    """3 seeds, timing only (no spread claim, so the 16-seed rule doesn't
    apply): confirms PLO4 has headroom at the chosen `trials`, same role
    Task 8 used PLO4 for -- the non-binding game."""
    print(f"\n=== PLO4 timing check at trials={t} (3 seeds, timing only) ===")
    for seed in (1, 2, 3):
        t0 = time.perf_counter()
        compute_range_ladder(board=BOARD, dead=PLO4_DEAD, heroes=PLO4_HEROES,
                             trials=t, seed=seed)
        print(f"  seed={seed}: {time.perf_counter() - t0:.3f}s")


if __name__ == "__main__":
    _warm()
    measure_scaling()
    probe_budget_boundary()
    _results, _buckets = sweep_accuracy()
    probe_gap_to_one_pp(_buckets)
    check_plo4_stays_ahead(350000)
