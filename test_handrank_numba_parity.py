"""
Verify the Numba and NumPy code paths in hand_rank_evaluator agree within
sampling noise.

This test is meant to run on production (or any x86_64 box where numba
installs successfully). On platforms where numba isn't available, it
prints a skip message and exits 0.

What it checks:
  * For a grid of (plo_variant, hero_hand), run N trials with the NumPy path
    AND the Numba path. Expected equity difference is ~ 1 / sqrt(trials)
    * 0.5 (Bernoulli-ish per-trial variance). We allow 3 sigma on each
    comparison.
  * Also times both paths back to back so you can eyeball the speedup.

Usage:
    python test_handrank_numba_parity.py
"""

import time
import math
import sys

import numpy as np

import hand_rank_evaluator as hre


TEST_HANDS = {
    'plo4': ['AsAhKsKh', 'JdTc9d8c', '7c6s5h4d'],
    'plo5': ['AsAhKsKhQs', 'JdTc9d8c7s', '6h5d4c3s2h'],
    'plo6': ['AsAhKsKhQsJs', 'JdTc9d8c7s6h', '7c6d5c4h3s2d'],
}

TRIALS = 5000
NUM_OPPONENTS = 2
# 3-way equity has per-trial variance bounded by ~0.25 (Bernoulli on {0, 0.5, 1}).
# Stdev of the mean equity ~= sqrt(0.25 / trials). Allow 3 sigma window for
# each path independently, hence 3*sqrt(0.5/trials) combined.
SIGMA = 3 * math.sqrt(0.5 / TRIALS)


def run_one(hand: str, use_numba: bool) -> tuple:
    """Run a single hand with a forced path and return (equity, wall_ms)."""
    original = hre.USE_NUMBA
    hre.USE_NUMBA = use_numba and hre.HAVE_NUMBA
    try:
        t0 = time.perf_counter()
        r = hre.hand_rank_single(hand, num_trials=TRIALS, seed=None)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return r['equity'], dt_ms
    finally:
        hre.USE_NUMBA = original


def main() -> int:
    print(f"HAVE_NUMBA={hre.HAVE_NUMBA}  USE_NUMBA={hre.USE_NUMBA}")
    if not hre.HAVE_NUMBA:
        print("[skip] numba not available on this platform - nothing to compare.")
        return 0

    hre.warmup_handrank()

    print(f"\nParity test: trials={TRIALS}, 3-sigma tolerance = {SIGMA:.4f} equity\n")
    header = f"{'variant':6}  {'hand':14}  {'eq_numpy':>9}  {'eq_numba':>9}  {'delta':>7}  {'numpy_ms':>8}  {'numba_ms':>8}  {'speedup':>7}  status"
    print(header)
    print('-' * len(header))

    failures = 0
    for variant, hands in TEST_HANDS.items():
        for hand in hands:
            eq_numpy, t_numpy = run_one(hand, use_numba=False)
            eq_numba, t_numba = run_one(hand, use_numba=True)
            delta = abs(eq_numba - eq_numpy)
            speedup = t_numpy / t_numba if t_numba > 0 else float('inf')
            status = 'ok' if delta <= SIGMA else 'FAIL'
            if status == 'FAIL':
                failures += 1
            print(
                f"{variant:6}  {hand:14}  "
                f"{eq_numpy:9.4f}  {eq_numba:9.4f}  {delta:7.4f}  "
                f"{t_numpy:8.1f}  {t_numba:8.1f}  {speedup:7.2f}  {status}"
            )

    print()
    if failures:
        print(f"FAIL: {failures} hand(s) exceeded 3-sigma tolerance.")
        return 1
    print("PASS: all hands within 3-sigma tolerance.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
