"""Pick the runout count: largest R inside the 2-3s budget that holds +/-1pp.

Grid note (Task 8): the brief's original grid was R in (100, 200, 400, 800,
1600) x 2 games x 3 seeds at hands=10000. A single PLO6 datapoint at R=100
measured ~8.3-8.8s -- already 3.5x over the 2.5s budget -- and scaling is
linear in R (~0.07s/runout for PLO6, ~0.04s/runout for PLO4), so every value
in the brief's grid (all >= 100) would blow the budget by 3x-140x. Running
the brief's grid as written would take well over 20 minutes and every point
on it is off the table before it starts.

Instead this grid brackets the *actual* measured budget boundary for PLO6
(R ~ 25-30, found by a manual probe: R=20 -> 1.69s, R=30 -> 2.55s, R=40 ->
3.34s): R in (10, 15, 20, 25, 30), 2 games, 3 seeds, hands=10000. That is 30
total compute_range_ladder calls; PLO6 dominates at ~1.5s average and PLO4 at
~0.6s average, for a total runtime of well under a minute.
"""
import time

import numpy as np

from range_ladder import compute_range_ladder

CASES = {
    "plo4": (["AsKs9h2c", "QhJhTd3d"], [{"id": "h", "cards": "AsKs9h2c"}]),
    "plo6": (["AsKs9h2c3d4d", "QhJhTd3s5s8s"], [{"id": "h", "cards": "AsKs9h2c3d4d"}]),
}

RUNOUT_GRID = (10, 15, 20, 25, 30)

# One throwaway call before timing starts: importing this module pays a
# one-time ~0.45-0.5s cost (loading score_array.npy/category_array.npy,
# numba JIT-compiling the scoring kernels) that lands on whichever call
# happens to run first. That cost is paid once per live process, not once
# per board, so leaving it in the first grid row would overstate steady
# -state per-board cost by ~20-25% at small R. Warm it up here instead.
compute_range_ladder(board="6s7s4s", dead=CASES["plo6"][0], heroes=CASES["plo6"][1],
                     hands=10000, runouts=5, seed=0)

for game, (dead, heroes) in CASES.items():
    for r in RUNOUT_GRID:
        eqs = []
        t0 = time.perf_counter()
        for seed in (1, 2, 3):
            out = compute_range_ladder(board="6s7s4s", dead=dead, heroes=heroes,
                                       hands=10000, runouts=r, seed=seed)
            eqs.append([x["equity"] for x in out["ladders"][0]["rungs"]])
        elapsed = (time.perf_counter() - t0) / 3
        spread = float(np.ptp(np.array(eqs), axis=0).max())
        print(f"{game:5s} R={r:5d}  {elapsed:6.2f}s/board  max spread {spread*100:5.2f}pp")
