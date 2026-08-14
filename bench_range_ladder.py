"""Task 13 re-measurement: the single joint-trial pass (Task 11/12) is
replaced by two passes with opposite sampling -- Pass 1 ranks villain HANDS
on SHARED runouts (`hands` x `rank_runouts`), Pass 2 measures hero equity on
INDEPENDENT, per-trial runouts stratified equally across buckets
(`trials_per_bucket` x len(buckets) x (1 + heroes)). There are now three
knobs instead of one; this file sweeps all three together, plus the new
boundary-churn measurement the brief calls for (does Pass 1's
`rank_runouts` actually rank hands consistently, independent of Pass 2
entirely).

MUST run under `if __name__ == "__main__":` (see bottom of this file), not
as bare module-level code: ProcessPoolExecutor uses Windows' spawn start
method, which re-imports this file's top-level code in every worker
process. Without the guard, each spawned worker would re-run the entire
sweep and spawn its own pool of workers -- unbounded recursive process
creation.

Scope trimmed to fit comfortably inside a ~25 minute run (see
task-13-report.md for exactly what got trimmed and why):
  - Only PLO6 gets the full accuracy/churn sweep (single hero) -- it was
    already the binding case for TIME in every prior task on this feature.
    A short PLO4 timing-only check (no spread claim) confirms it stays
    comfortably ahead of budget at the chosen defaults.
  - The candidate grid is the small, hand-picked set this task's actual
    investigation converged on (see task-13-report.md for the wider sweep
    that got them there -- including several candidates that turned out
    NOT to be worth including here because they were dominated on both
    time and accuracy by a nearby candidate that IS listed).
"""
import time

import numpy as np

import fast_score
from range_ladder import (
    compute_range_ladder, warmup_pool, DEFAULT_POOL_WORKERS, DEFAULT_BUCKETS,
    sample_villains, sample_runouts, rank_hands,
)
from card_encoding import generate_deck_ints, hand_str_to_ints
from hand_rank_evaluator import get_score_array, detect_game_type

PLO6_DEAD = ["AsKs9h2c3d4d", "QhJhTd3s5s8s"]
PLO6_HEROES = [{"id": "h", "cards": "AsKs9h2c3d4d"}]
PLO4_DEAD = ["AsKs9h2c", "QhJhTd3d"]
PLO4_HEROES = [{"id": "h", "cards": "AsKs9h2c"}]
BOARD = "6s7s4s"
BUDGET_S = 2.5

# 16 seeds (the brief's minimum): an 8-seed spread estimate produced a
# confidently wrong conclusion earlier in this project (Task 10), so
# anything shipped here uses at least that many, split 8+8 across two
# non-adjacent seed bands rather than one contiguous run of 16, in case of
# any seed-adjacency correlation.
SPREAD_SEEDS = tuple(range(1, 9)) + tuple(range(21, 29))

# (hands, rank_runouts, trials_per_bucket) candidates this task's
# investigation converged on -- see task-13-report.md for the wider sweep
# (rank_runouts alone from 30 to 1000; hands alone from 10,000 to 150,000;
# trials_per_bucket alone from 10,000 to 1,000,000) that motivated this
# specific set: rank_runouts=30 (the brief's own suggested starting point)
# measured 73.6% boundary churn -- nowhere near "enough to bucket
# correctly" -- so every candidate below uses 300 instead, the point past
# which pushing rank_runouts further stopped being the best use of the
# remaining time budget (1000 blew well past 2.5s without proportionally
# shrinking the residual spread -- Stage-A population-sampling noise, not
# Pass-1 ranking noise, dominates what's left at that point).
CANDIDATES = [
    (10_000, 300, 30_000),
    (10_000, 300, 50_000),
    (10_000, 300, 80_000),
]


def _warm():
    warmup_pool()
    compute_range_ladder(board=BOARD, dead=PLO6_DEAD, heroes=PLO6_HEROES,
                         hands=200, rank_runouts=10, trials_per_bucket=50, seed=0)


def measure_scaling():
    """How parallel speedup scales with worker count, PLO6, a fixed
    moderate workload."""
    print(f"\n=== Scaling with worker count (DEFAULT_POOL_WORKERS={DEFAULT_POOL_WORKERS}) ===")
    kw = dict(board=BOARD, dead=PLO6_DEAD, heroes=PLO6_HEROES,
             hands=10_000, rank_runouts=300, trials_per_bucket=30_000, seed=1)
    t0 = time.perf_counter()
    compute_range_ladder(parallel=False, **kw)
    serial_t = time.perf_counter() - t0
    print(f"serial:            {serial_t:6.3f}s")

    for workers in (1, 2, 4, 8, 12, 16, 24):
        if workers > DEFAULT_POOL_WORKERS:
            continue
        t0 = time.perf_counter()
        compute_range_ladder(parallel=True, num_workers=workers, **kw)
        t = time.perf_counter() - t0
        print(f"parallel workers={workers:3d}: {t:6.3f}s  speedup={serial_t / t:5.2f}x")


def _spread_row(hands, rank_runouts, trials_per_bucket, seeds=SPREAD_SEEDS):
    """(mean_t, max_t, spread_pp array, over_budget_count) for one
    candidate, PLO6, across `seeds`."""
    eqs, times = [], []
    for seed in seeds:
        t0 = time.perf_counter()
        out = compute_range_ladder(board=BOARD, dead=PLO6_DEAD, heroes=PLO6_HEROES,
                                   hands=hands, rank_runouts=rank_runouts,
                                   trials_per_bucket=trials_per_bucket, seed=seed)
        times.append(time.perf_counter() - t0)
        eqs.append([x["equity"] for x in out["ladders"][0]["rungs"]])
    eqs = np.array(eqs)
    spread_pp = np.ptp(eqs, axis=0) * 100
    mean_t, max_t = float(np.mean(times)), float(np.max(times))
    over = sum(1 for x in times if x > BUDGET_S)
    return mean_t, max_t, spread_pp, over


def sweep_accuracy():
    """Elapsed time, headroom and per-bucket run-to-run spread (16 seeds)
    for each (hands, rank_runouts, trials_per_bucket) candidate, PLO6.
    Picks the candidate with the most headroom among those that stayed
    under budget across EVERY seed (not just the mean) -- per the Task 10
    lesson that a mean well under budget can still hide occasional
    tail-latency spikes over it -- and reports whether every bucket lands
    inside +/-1pp (honestly: at this budget, it does not, see
    task-13-report.md)."""
    buckets = list(DEFAULT_BUCKETS)
    print(f"\n=== Accuracy sweep (PLO6, {len(SPREAD_SEEDS)} seeds) ===")
    header = "  ".join(f"{b:>4}%" for b in buckets)
    print(f"{'hands':>7} {'rr':>5} {'tpb':>7} {'mean_t':>7} {'max_t':>7} {'headroom':>9} {'over':>6}  [{header}]")
    results = []
    for hands, rr, tpb in CANDIDATES:
        mean_t, max_t, spread_pp, over = _spread_row(hands, rr, tpb)
        headroom = (BUDGET_S - max_t) / BUDGET_S * 100
        results.append((hands, rr, tpb, mean_t, max_t, spread_pp, over, headroom))
        buckets_str = "  ".join(f"{x:5.2f}" for x in spread_pp)
        print(f"{hands:7d} {rr:5d} {tpb:7d} {mean_t:7.3f} {max_t:7.3f} {headroom:8.1f}% "
             f"{over:5d}/{len(SPREAD_SEEDS)}  [{buckets_str}]")

    under_budget = [r for r in results if r[4] <= BUDGET_S]
    if not under_budget:
        print("\nNo candidate stayed under budget across every seed -- widen CANDIDATES.")
        return results, buckets, None

    # ~30% headroom target (not the largest candidate that merely fits --
    # see the brief: the old DEFAULT_TRIALS was picked at 3.6% headroom and
    # overran budget on a busy machine).
    chosen = max(under_budget, key=lambda r: r[7])
    hands, rr, tpb, mean_t, max_t, spread_pp, over, headroom = chosen
    print(f"\nChosen: hands={hands} rank_runouts={rr} trials_per_bucket={tpb} "
         f"(mean {mean_t:.3f}s, max {max_t:.3f}s, headroom {headroom:.1f}%, "
         f"0/{len(SPREAD_SEEDS)} over {BUDGET_S}s budget)")
    print("Per-bucket +/-1pp verdict:")
    all_reached = True
    for b, sp in zip(buckets, spread_pp):
        verdict = "REACHED" if sp <= 1.0 else "NOT reached"
        all_reached &= sp <= 1.0
        print(f"  bucket {b:>4}%: spread {sp:5.2f}pp -- {verdict}")
    print(f"Every bucket inside +/-1pp: {all_reached}")
    return results, buckets, (hands, rr, tpb)


def measure_boundary_churn(hands=10_000, rank_runouts_candidates=(30, 300), seeds=SPREAD_SEEDS):
    """The brief's headline diagnostic for Pass 1: with a FIXED villain-hand
    population (sampled ONCE, reused across every seed below -- isolating
    Pass-1 ranking noise from "which hands even got sampled" noise), how
    much does bucket membership move between independent Pass-1 rankings
    that differ only in which `rank_runouts` shared runouts got drawn?

    Reports, per rank_runouts candidate: the fraction of hands that land
    in a DIFFERENT bucket in at least one of the `seeds` runs, and the
    spread of each bucket's edge-hand strength across those runs. This is
    the number that says whether a given rank_runouts is "enough" -- see
    DEFAULT_RANK_RUNOUTS's comment in range_ladder.py for how 30 (the
    brief's own suggested starting point) and 300 (the value actually
    shipped) compare.
    """
    print(f"\n=== Boundary churn (PLO6, fixed {hands}-hand population, {len(seeds)} seeds) ===")
    dead_cards = [c for d in PLO6_DEAD for c in (d[i:i + 2] for i in range(0, len(d), 2))]
    board_cards = [BOARD[i:i + 2] for i in range(0, len(BOARD), 2)]
    deck = generate_deck_ints(dead_cards + board_cards)
    board_ints = hand_str_to_ints(BOARD)
    score_array = get_score_array()
    game = detect_game_type(6)

    pop_rng = np.random.default_rng(999)
    villain_hands = sample_villains(deck, 6, hands, pop_rng)
    n = villain_hands.shape[0]

    def bucket_label(strength):
        order = np.argsort(-strength, kind="stable")
        label = np.full(n, 999, dtype=np.int64)
        for pct in sorted(DEFAULT_BUCKETS):
            take = max(1, round(n * pct / 100))
            label[order[:take]] = np.minimum(label[order[:take]], pct)
        return label, order

    for rank_runouts in rank_runouts_candidates:
        labels_runs = []
        edge_strength = {p: [] for p in DEFAULT_BUCKETS}
        for seed in seeds:
            rng = np.random.default_rng(seed)
            runouts = sample_runouts(deck, 3, rank_runouts, rng)
            strength, counts = rank_hands(villain_hands, board_ints, runouts, game, score_array)
            strength = np.where(counts > 0, strength, -1.0)
            label, order = bucket_label(strength)
            labels_runs.append(label)
            for pct in DEFAULT_BUCKETS:
                take = max(1, round(n * pct / 100))
                edge_strength[pct].append(float(strength[order[take - 1]]))

        labels_runs = np.array(labels_runs)
        stable = (labels_runs == labels_runs[0]).all(axis=0)
        churn_frac = 1.0 - stable.mean()
        print(f"\nrank_runouts={rank_runouts}: boundary churn = {churn_frac:.4f} "
             f"({int(churn_frac * n)}/{n} hands changed bucket at least once)")
        for pct in DEFAULT_BUCKETS:
            vals = np.array(edge_strength[pct])
            print(f"  bucket {pct:>4}%: edge-strength spread = {100 * (vals.max() - vals.min()):5.2f}pp")


def check_plo4_stays_ahead(hands, rr, tpb):
    """3 seeds, timing only (no spread claim, so the 16-seed rule doesn't
    apply): confirms PLO4 has headroom at the chosen parameters."""
    print(f"\n=== PLO4 timing check at hands={hands} rank_runouts={rr} trials_per_bucket={tpb} (3 seeds) ===")
    for seed in (1, 2, 3):
        t0 = time.perf_counter()
        compute_range_ladder(board=BOARD, dead=PLO4_DEAD, heroes=PLO4_HEROES,
                             hands=hands, rank_runouts=rr, trials_per_bucket=tpb, seed=seed)
        print(f"  seed={seed}: {time.perf_counter() - t0:.3f}s")


if __name__ == "__main__":
    _warm()
    measure_scaling()
    measure_boundary_churn()
    _results, _buckets, _chosen = sweep_accuracy()
    if _chosen is not None:
        check_plo4_stays_ahead(*_chosen)
