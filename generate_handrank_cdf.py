"""
Offline generator for the Hand Rank baseline CDF.

For each PLO variant (plo4 / plo5 / plo6):
  1. Sample N random starting hands uniformly from a 52-card deck.
  2. For each hand, run a 3-way Monte Carlo (hero vs 2 random opponents, full random
     runout, `--trials` iterations) to estimate its 3-way equity.
  3. Sort the N equities ascending and save as `handrank_cdf_{variant}.npy`.

At request time, `hand_rank_evaluator.equity_to_rank` searches this sorted array
to convert a hero's MC equity into a 1-100 percentile rank.

Examples:
  python generate_handrank_cdf.py --all --num-hands 5000 --trials 2000
  python generate_handrank_cdf.py --variant plo4 --num-hands 50000 --trials 3000
  python generate_handrank_cdf.py --variant plo6 --num-hands 20000 --trials 2500 --seed 17

Runtime scales roughly linearly with num_hands * trials. On a modern x64 Linux
box with numpy+BLAS, expect ~15-30 hands/s at 3000 trials for PLO6. Windows ARM64
dev box is ~5-10x slower. Regenerate on production hardware for final CDFs.
"""

import argparse
import json
import os
import random
import sys
import time
from typing import List

import numpy as np

from hand_rank_evaluator import (
    hand_rank_single,
    get_score_array,
    reload_handrank_cdf,
)


RANKS = '23456789TJQKA'
SUITS = 'shdc'


def generate_deck() -> List[str]:
    return [r + s for r in RANKS for s in SUITS]


def sample_unique_hand(deck: List[str], num_cards: int, rng: random.Random, seen: set) -> str:
    """Uniform random draw of num_cards distinct cards. Canonicalize to avoid duplicates."""
    for _ in range(100):
        hand = rng.sample(deck, num_cards)
        key = ''.join(sorted(hand))
        if key not in seen:
            seen.add(key)
            return hand
    # Extremely unlikely with a 52-card deck and realistic num_hands, but fall back
    # to a plain random hand if we hit a pathological collision streak.
    return rng.sample(deck, num_cards)


def generate_cdf_for_variant(
    variant: str,
    num_hands: int,
    trials: int,
    seed: int,
    num_opponents: int = 2,
    progress_every: int = 100,
) -> np.ndarray:
    num_cards = {'plo4': 4, 'plo5': 5, 'plo6': 6}[variant]
    deck = generate_deck()
    rng = random.Random(seed)

    equities = np.zeros(num_hands, dtype=np.float64)
    seen: set = set()

    t_start = time.perf_counter()
    print(f"[CDF] {variant}: sampling {num_hands} hands @ {trials} trials, {num_opponents+1}-way...")
    for i in range(num_hands):
        hand = sample_unique_hand(deck, num_cards, rng, seen)
        # Per-hand deterministic seed so reruns with same --seed reproduce exactly.
        sub_seed = seed * 1_000_003 + i
        r = hand_rank_single(
            hand,
            num_trials=trials,
            num_opponents=num_opponents,
            seed=sub_seed,
        )
        equities[i] = r['equity']

        if (i + 1) % progress_every == 0 or (i + 1) == num_hands:
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (num_hands - (i + 1)) / max(rate, 1e-6)
            sys.stdout.write(
                f"\r[CDF] {variant}: {i+1}/{num_hands}  "
                f"rate={rate:.1f} hands/s  elapsed={elapsed:.0f}s  eta={eta:.0f}s"
            )
            sys.stdout.flush()
    print()

    equities.sort()
    return equities


def save_cdf(variant: str, equities: np.ndarray, output_dir: str) -> str:
    path = os.path.join(output_dir, f'handrank_cdf_{variant}.npy')
    np.save(path, equities)
    print(
        f"[CDF] {variant}: saved {path}  "
        f"n={len(equities)}  "
        f"min={equities.min():.4f}  "
        f"p25={np.percentile(equities, 25):.4f}  "
        f"median={np.median(equities):.4f}  "
        f"p75={np.percentile(equities, 75):.4f}  "
        f"max={equities.max():.4f}"
    )
    return path


def update_meta(meta_path: str, variant: str, info: dict) -> None:
    meta: dict = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    meta.setdefault('variants', {})[variant] = info
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--variant', choices=['plo4', 'plo5', 'plo6'], help='Single PLO variant to generate.')
    p.add_argument('--all', action='store_true', help='Generate for all three variants.')
    p.add_argument('--num-hands', type=int, default=50000, help='Number of random hands to sample.')
    p.add_argument('--trials', type=int, default=3000, help='MC trials per hand.')
    p.add_argument('--num-opponents', type=int, default=2, help='Opponents per trial (default 2 -> 3-way).')
    p.add_argument('--output-dir', default='.', help='Where to write the .npy files.')
    p.add_argument('--seed', type=int, default=42, help='Master RNG seed for reproducibility.')
    args = p.parse_args()

    if not args.all and not args.variant:
        p.error("Provide either --variant <plo4|plo5|plo6> or --all")

    variants = ['plo4', 'plo5', 'plo6'] if args.all else [args.variant]

    os.makedirs(args.output_dir, exist_ok=True)
    meta_path = os.path.join(args.output_dir, 'handrank_cdf_meta.json')

    get_score_array()  # preload cache

    for v in variants:
        t0 = time.perf_counter()
        equities = generate_cdf_for_variant(
            v,
            num_hands=args.num_hands,
            trials=args.trials,
            seed=args.seed,
            num_opponents=args.num_opponents,
        )
        save_cdf(v, equities, args.output_dir)
        info = {
            'num_hands': args.num_hands,
            'trials_per_hand': args.trials,
            'num_opponents': args.num_opponents,
            'seed': args.seed,
            'elapsed_s': round(time.perf_counter() - t0, 2),
            'min': float(equities.min()),
            'p01': float(np.percentile(equities, 1)),
            'p05': float(np.percentile(equities, 5)),
            'p10': float(np.percentile(equities, 10)),
            'p25': float(np.percentile(equities, 25)),
            'median': float(np.median(equities)),
            'p75': float(np.percentile(equities, 75)),
            'p90': float(np.percentile(equities, 90)),
            'p95': float(np.percentile(equities, 95)),
            'p99': float(np.percentile(equities, 99)),
            'max': float(equities.max()),
        }
        update_meta(meta_path, v, info)

    reload_handrank_cdf()
    print(f"[CDF] metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
