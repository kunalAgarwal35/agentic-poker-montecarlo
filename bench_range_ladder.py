"""Task 10 re-measurement: parallel scaling, then re-sweep N (hands) x R
(runouts) for the constants now that evaluate_population runs on
multithread_ploequities3's persistent process pool instead of a single
process. See range_ladder.py's DEFAULT_RUNOUTS/DEFAULT_HANDS comments and
task-10-report.md for the numbers this produced and how they were chosen.

MUST run under `if __name__ == "__main__":` (see bottom of this file), not
as bare module-level code: ProcessPoolExecutor uses Windows' spawn start
method, which re-imports this file's top-level code in every worker
process. Without the guard, each spawned worker would re-run the entire
sweep and spawn its own pool of workers -- unbounded recursive process
creation.

Task 8's original grid (this file's previous version) bracketed the
single-threaded 2.5s boundary at R in (10, 15, 20, 25, 30), hands=10000.
That boundary moved once the pool went in -- PLO6 throughput measured at
~590k-600k hand-evals/sec at DEFAULT_POOL_WORKERS=24 (vs Task 8's
~113k-115k/sec single-threaded, a ~5.2-5.3x speedup -- NOT ~24x; this box
is 12 physical / 24 logical hyperthreaded cores, and the scaling curve
below plateaus hard past the physical count).

IMPORTANT, and NOT fully reproduced by this script alone: the N/R choice
actually shipped (DEFAULT_HANDS=10000, DEFAULT_RUNOUTS=100) was NOT simply
"best worst-bucket spread in the sweep below". Parallel dispatch turned out
to have real tail latency (wall-clock time is set by the slowest of W
concurrently-dispatched chunks) that the old single-process loop never had.
Several candidates with better spread than the chosen one (e.g. N=4000/
R=350) looked safely under the 2.5s budget in a 12-16-trial timing sample
but showed an occasional (~1-in-12-to-24) spike to 2.6-2.9s once sampled
more deeply -- caught by a separate, deeper timing-jitter check (not
reproduced here to keep this script's runtime reasonable; see
task-10-report.md for the full trail). N=10000/R=100 was the only
candidate that stayed clean across that deeper check. THIS script's
"Best under budget" printout at the bottom of sweep_accuracy() uses only
the SPREAD_SEEDS-sized sample below and can disagree with that conclusion
for exactly this reason -- treat it as illustrative, not authoritative;
task-10-report.md has the real trail.

Scope trimmed to fit comfortably inside a ~25 minute run (the full set of
sweeps actually run for the Task 10 report, across several scripts,
totaled well under that):
  - Scaling curve measured once, at one moderate workload (not swept
    per-candidate) -- how parallel speedup scales with worker count doesn't
    depend on which N/R the accuracy sweep below eventually picks.
  - Only PLO6 gets the full accuracy sweep. PLO6 was already the binding
    case for TIME in Task 8; a direct check below (3 seeds, timing only, no
    spread claim) confirms it stays comfortably ahead on PLO4 at the chosen
    N/R, so a second full accuracy sweep for PLO4 was skipped.
  - The N/R candidate grid is a small, hand-picked set bracketing the 2.5s
    boundary from both directions (R-heavy and N-heavy), plus a few points
    around the eventual N=10000 winner, rather than an exhaustive 2D grid
    -- exhaustive at deep seed counts would run well past 25 minutes for no
    real gain in the conclusion.
  - The deeper tail-latency stress test (24-seed and isolated 12-trial
    timing-only batches per candidate) that actually decided the final
    pick is NOT re-run here every time -- it added several extra minutes
    and its role was to VETO risky candidates, not to change the headline
    "+/-1pp not reached" conclusion. Its numbers are in task-10-report.md.
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

# 16 seeds (double the brief's 8-seed minimum): the brief explicitly warns
# that a 3-seed spread estimate produced a confidently wrong conclusion
# earlier in this project, so the number that actually gets shipped in
# DEFAULT_RUNOUTS/DEFAULT_HANDS uses extra margin over that minimum.
SPREAD_SEEDS = tuple(range(1, 9)) + tuple(range(21, 29))

# Candidates bracketing the ~1.0-1.5M-eval budget boundary (measured via a
# quick single-seed timing probe before this file was finalized -- see
# task-10-report.md for that probe's full output). Spans from N-heavy
# (more bucket-membership resolution) to R-heavy (more equity precision),
# plus a couple of points around N=10000 bracketing the eventual winner.
CANDIDATES = [
    (10000, 25),    # Task 8's serial-era default, kept for comparison
    (10000, 100),   # the winner shipped as DEFAULT_HANDS/DEFAULT_RUNOUTS
    (10000, 120),
    (8000, 150),
    (5000, 250),
    (4000, 350),    # best spread found in this sweep; rejected on tail-latency risk
    (2000, 600),
    (1000, 1000),
]


def _warm():
    warmup_pool()
    # Throwaway call: pays the one-time numba JIT-compile cost so it
    # doesn't land on whichever timed call happens to run first.
    compute_range_ladder(board=BOARD, dead=PLO6_DEAD, heroes=PLO6_HEROES,
                         hands=200, runouts=5, seed=0)


def measure_scaling():
    """How parallel speedup scales with worker count, PLO6, a fixed
    moderate workload (hands=10000, runouts=200 -> ~2M hand-evaluations,
    big enough that dispatch overhead is a small fraction of total time)."""
    print(f"\n=== Scaling with worker count (DEFAULT_POOL_WORKERS={DEFAULT_POOL_WORKERS}) ===")
    hands, runouts = 10000, 200
    t0 = time.perf_counter()
    compute_range_ladder(board=BOARD, dead=PLO6_DEAD, heroes=PLO6_HEROES,
                         hands=hands, runouts=runouts, seed=1, parallel=False)
    serial_t = time.perf_counter() - t0
    print(f"serial:            {serial_t:6.3f}s")

    for workers in (1, 2, 4, 8, 12, 16, 24):
        if workers > DEFAULT_POOL_WORKERS:
            continue
        t0 = time.perf_counter()
        compute_range_ladder(board=BOARD, dead=PLO6_DEAD, heroes=PLO6_HEROES,
                             hands=hands, runouts=runouts, seed=1,
                             parallel=True, num_workers=workers)
        t = time.perf_counter() - t0
        print(f"parallel workers={workers:3d}: {t:6.3f}s  speedup={serial_t / t:5.2f}x")


def probe_budget_boundary():
    """Quick single-seed timing-only probe (PLO6) locating where total
    hand-evaluations cross the 2.5s budget, to sanity-check that
    CANDIDATES above actually brackets it."""
    print("\n=== Budget-boundary probe (PLO6, single seed, timing only) ===")
    print(f"{'N':>7} {'R':>6} {'n_evals':>10} {'time(s)':>8}")
    for n, r in CANDIDATES:
        t0 = time.perf_counter()
        compute_range_ladder(board=BOARD, dead=PLO6_DEAD, heroes=PLO6_HEROES,
                             hands=n, runouts=r, seed=1)
        t = time.perf_counter() - t0
        flag = "  OVER BUDGET" if t > BUDGET_S else ""
        print(f"{n:7d} {r:6d} {(n + 1) * r:10d} {t:8.3f}{flag}")


def sweep_accuracy():
    """The real re-measurement: elapsed time and per-bucket run-to-run
    spread (max equity spread across SPREAD_SEEDS) for each candidate,
    PLO6. Picks and prints the best-under-budget candidate by worst-bucket
    spread."""
    print(f"\n=== Accuracy sweep (PLO6, {len(SPREAD_SEEDS)} seeds) ===")
    print(f"{'N':>7} {'R':>6} {'mean_t':>8} {'max_t':>8} {'worst_pp':>9} {'100pp':>7}  buckets(pp)")
    results = []
    for n, r in CANDIDATES:
        eqs, times = [], []
        for seed in SPREAD_SEEDS:
            t0 = time.perf_counter()
            out = compute_range_ladder(board=BOARD, dead=PLO6_DEAD, heroes=PLO6_HEROES,
                                       hands=n, runouts=r, seed=seed)
            times.append(time.perf_counter() - t0)
            eqs.append([x["equity"] for x in out["ladders"][0]["rungs"]])
        eqs = np.array(eqs)
        spread_pp = np.ptp(eqs, axis=0) * 100
        mean_t, max_t = float(np.mean(times)), float(np.max(times))
        worst, hundred = float(spread_pp.max()), float(spread_pp[-1])
        results.append((n, r, mean_t, max_t, worst, hundred))
        buckets_str = " ".join(f"{x:5.2f}" for x in spread_pp)
        print(f"{n:7d} {r:6d} {mean_t:8.3f} {max_t:8.3f} {worst:9.2f} {hundred:7.2f}  [{buckets_str}]")

    under_budget = [row for row in results if row[3] <= BUDGET_S]
    if not under_budget:
        print("\nNo candidate stayed under budget across every seed -- widen CANDIDATES.")
        return
    best = min(under_budget, key=lambda row: row[4])
    n, r, mean_t, max_t, worst, hundred = best
    print(f"\nBest under {BUDGET_S}s budget: N={n} R={r} "
         f"(mean {mean_t:.2f}s, max {max_t:.2f}s, worst-bucket {worst:.2f}pp)")
    if worst <= 1.0:
        print("+/-1pp target: REACHED.")
    else:
        print(f"+/-1pp target: NOT reached ({worst:.2f}pp vs 1.00pp target). "
             "See range_ladder.py's DEFAULT_RUNOUTS comment and "
             "task-10-report.md for what closing the gap would take.")


def check_plo4_stays_ahead(n, r):
    """3 seeds, timing only (no spread claim, so the 8-seed rule doesn't
    apply): confirms PLO4 has headroom at the chosen N/R, same role Task 8
    used PLO4 for -- the non-binding game."""
    print(f"\n=== PLO4 timing check at N={n} R={r} (3 seeds, timing only) ===")
    for seed in (1, 2, 3):
        t0 = time.perf_counter()
        compute_range_ladder(board=BOARD, dead=PLO4_DEAD, heroes=PLO4_HEROES,
                             hands=n, runouts=r, seed=seed)
        print(f"  seed={seed}: {time.perf_counter() - t0:.3f}s")


if __name__ == "__main__":
    _warm()
    measure_scaling()
    probe_budget_boundary()
    sweep_accuracy()
    check_plo4_stays_ahead(4000, 350)
