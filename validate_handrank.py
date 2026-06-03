"""
Local validation for the Hand Rank API.

Checks:
  1. Sanity - premium hands land in top quartile, trash hands in bottom quartile.
  2. Monotonicity - a set of preflop-ordered PLO4 hands rank in order.
  3. Consistency - 10 repeated calls on the same hand vary by <= 2 rank units.
  4. loh25 top-hand spot-check - random samples from loh25_plo4.txt / loh25_plo5.txt
     all land in the top quartile (these are labelled top-25% hands).

Run standalone (no Flask needed):
    python validate_handrank.py
"""

import random
import statistics
import sys
import time
from typing import List

import numpy as np

from hand_rank_evaluator import (
    hand_rank_single,
    get_handrank_cdf,
    warmup_handrank,
)


def fmt_ok(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def check(title: str, condition: bool, detail: str = "") -> bool:
    print(f"  [{fmt_ok(condition)}] {title}" + (f"  -> {detail}" if detail else ""))
    return condition


# ---------------------------------------------------------------------------
# 1. Sanity: top vs bottom
# ---------------------------------------------------------------------------

PREMIUM = {
    'plo4': ['AsAhKsKh', 'AsAhKdKs', 'AcAdKcKd', 'AsAhQsQh'],
    'plo5': ['AsAhKsKhQs', 'AsAhKsKhJs', 'AsAhQsQhJs'],
    'plo6': ['AsAhKsKhQsJs', 'AsAhKsKhQsQh', 'AsAhKsKhTsTh'],
}
TRASH = {
    'plo4': ['2s3c4d5h', '2s4c6d8h', '2s3c7d9h'],
    'plo5': ['2s3c4d5h7c', '2s4c6d8h9c', '2s3c5d7h9c'],
    'plo6': ['2s3c4d5h7c8d', '2s4c6d8h9cTd', '2s3c5d7h9cJd'],
}


def run_sanity() -> bool:
    print("1. Sanity: premium hands rank top, trash hands rank bottom")
    ok = True
    for variant, hands in PREMIUM.items():
        for h in hands:
            r = hand_rank_single(h, num_trials=3000, seed=1)
            ok &= check(
                f"{variant} premium  {h} -> rank {r['rank']} (equity {r['equity']:.3f})",
                r['rank'] is not None and r['rank'] <= 25,
            )
    for variant, hands in TRASH.items():
        for h in hands:
            r = hand_rank_single(h, num_trials=3000, seed=1)
            ok &= check(
                f"{variant} trash    {h} -> rank {r['rank']} (equity {r['equity']:.3f})",
                r['rank'] is not None and r['rank'] >= 75,
            )
    return ok


# ---------------------------------------------------------------------------
# 2. Monotonicity: stronger PLO4 hands rank <= weaker ones
# ---------------------------------------------------------------------------

MONOTONIC_PLO4 = [
    ('AsAhKsKh', 'strong (AAKK ds)'),
    ('AsAhQsTs', 'ok (AAQTss)'),
    ('Ks9s8s7s', 'ok (K987ss - single suit)'),
    ('JdTc9d8c', 'rundown mid'),
    ('7s6h5d4c', 'rundown low rainbow'),
    ('2s3c7d9h', 'scattered trash'),
]


def run_monotonicity() -> bool:
    print("2. Monotonicity: PLO4 hands ordered strong -> weak")
    ranks = []
    for hand, label in MONOTONIC_PLO4:
        r = hand_rank_single(hand, num_trials=3000, seed=1)
        print(f"  {label:36s}  {hand}  rank={r['rank']:3d}  equity={r['equity']:.3f}")
        ranks.append(r['rank'])
    # Allow small wobble at integer-rank level (adjacent monotonicity must be
    # non-strict, and at most 1 inversion tolerated).
    inversions = sum(1 for i in range(1, len(ranks)) if ranks[i] < ranks[i - 1])
    ok = inversions <= 1
    return check(f"Inversions <= 1 (got {inversions})", ok)


# ---------------------------------------------------------------------------
# 3. Consistency: same hand, 10 repeats
# ---------------------------------------------------------------------------

def run_consistency() -> bool:
    print("3. Consistency: 10 repeats on a mid-tier hand")
    hand = 'AsKsQdJd'  # mid-premium PLO4
    ranks = []
    equities = []
    for _ in range(10):
        r = hand_rank_single(hand, num_trials=5000)  # NOTE: no seed -> genuine variability
        ranks.append(r['rank'])
        equities.append(r['equity'])
    print(f"  {hand}  ranks={ranks}  equities=[{', '.join(f'{e:.3f}' for e in equities)}]")
    stdev = statistics.pstdev(ranks) if len(ranks) > 1 else 0
    print(f"  rank stdev={stdev:.2f}  rank range={max(ranks) - min(ranks)}")
    # Target: stdev <= 2 rank units, range <= 5 at 5000 trials
    return check(f"rank stdev <= 2.0 (got {stdev:.2f})", stdev <= 2.0) \
        and check(f"rank range <= 5 (got {max(ranks) - min(ranks)})", max(ranks) - min(ranks) <= 5)


# ---------------------------------------------------------------------------
# 4. loh25 top-hand spot check
# ---------------------------------------------------------------------------

def load_loh25(filepath: str) -> List[str]:
    import re
    with open(filepath, 'r') as f:
        content = f.read()
    if 'plo5' in filepath:
        pattern = r'[AKQJT98765432][shdc][AKQJT98765432][shdc][AKQJT98765432][shdc][AKQJT98765432][shdc][AKQJT98765432][shdc]'
    else:
        pattern = r'[AKQJT98765432][shdc][AKQJT98765432][shdc][AKQJT98765432][shdc][AKQJT98765432][shdc]'
    return list(set(re.findall(pattern, content)))


def run_loh25_spotcheck(variant: str, filepath: str, sample_size: int = 30) -> bool:
    print(f"4. loh25 spot check: {variant}")
    try:
        hands = load_loh25(filepath)
    except FileNotFoundError:
        return check(f"{filepath} present", False, "file missing - skipping")
    rng = random.Random(7)
    sampled = rng.sample(hands, min(sample_size, len(hands)))

    ranks = []
    for h in sampled:
        r = hand_rank_single(h, num_trials=2000, seed=1)
        ranks.append(r['rank'])
    n = len(ranks)
    in_top_quartile = sum(1 for r in ranks if r is not None and r <= 25)
    median_rank = statistics.median(ranks)
    print(f"  sampled {n} top-25% hands: median rank={median_rank}, top-quartile hits={in_top_quartile}/{n}")
    # At least 80% of loh25 hands should land in the top 25% by our metric.
    return check(
        f"{variant}: >= 80% of loh25 hands in top quartile (got {in_top_quartile}/{n})",
        in_top_quartile / n >= 0.80,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    t0 = time.perf_counter()
    warmup_handrank()
    for variant in ('plo4', 'plo5', 'plo6'):
        if get_handrank_cdf(variant) is None:
            print(f"[!] Baseline CDF for {variant} missing - run generate_handrank_cdf.py first.")
            return 2

    results = []
    results.append(('Sanity', run_sanity()))
    results.append(('Monotonicity', run_monotonicity()))
    results.append(('Consistency', run_consistency()))
    results.append(('loh25 plo4 spotcheck', run_loh25_spotcheck('plo4', 'loh25_plo4.txt')))
    results.append(('loh25 plo5 spotcheck', run_loh25_spotcheck('plo5', 'loh25_plo5.txt')))

    print("\n=== Summary ===")
    for name, ok in results:
        print(f"  [{fmt_ok(ok)}] {name}")
    total_ok = all(ok for _, ok in results)
    print(f"\nTotal elapsed: {time.perf_counter() - t0:.1f}s")
    return 0 if total_ok else 1


if __name__ == "__main__":
    sys.exit(main())
