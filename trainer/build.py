"""Build the trainer's question banks.

    python -m trainer.build --out <dir> [--out <dir> ...] [--flops 60]

Writes one file per (mode, game): <mode>-<game>.json, e.g. range_reads-plo4.json.
Pass every directory that serves banks: the trainer frontend's public/data and
the agentic web app's data/trainer (which scores challenges server-side).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from trainer.questions import MODES, FlopRanges
from trainer.features import texture

GAMES = ["PLO4", "PLO5"]
FLOPS_PER_MODE = {"range_reads": 1.0, "card_removal": 0.6, "range_vs_range": 0.6, "hero_equity": 0.6}


def sample_flops(n: int, seed: int = 7) -> list[list[int]]:
    """Deterministic, texture-balanced flops."""
    rng = np.random.default_rng(seed)
    quota = {"rainbow": round(0.35 * n), "two-tone": round(0.40 * n),
             "monotone": max(1, round(0.10 * n)), "paired": max(1, round(0.15 * n))}
    seen, out = set(), []
    while any(v > 0 for v in quota.values()):
        flop = sorted(rng.choice(52, size=3, replace=False).tolist(), reverse=True)
        t = texture(flop)
        bucket = "paired" if t["paired"] else t["suits"]
        if tuple(flop) in seen or quota[bucket] <= 0:
            continue
        quota[bucket] -= 1
        seen.add(tuple(flop))
        out.append(flop)
    return out


def build(flops: int, seed: int, modes: list[str] | None = None) -> dict[str, list[dict]]:
    banks: dict[str, list[dict]] = {}
    all_flops = sample_flops(flops, seed)
    for game in GAMES:
        rng = np.random.default_rng(seed)
        for i, flop in enumerate(all_flops):
            t0 = time.time()
            fr = FlopRanges(game, flop)
            for mode, gen in MODES.items():
                if modes and mode not in modes:
                    continue
                # Spread the more expensive modes over a subset of flops.
                if i >= round(FLOPS_PER_MODE[mode] * len(all_flops)):
                    continue
                banks.setdefault(f"{mode}-{game.lower()}", []).extend(gen(fr, rng))
            print(f"{game} flop {i + 1}/{len(all_flops)} {time.time() - t0:.1f}s", flush=True)
    return banks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", action="append", required=True)
    ap.add_argument("--flops", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--modes", nargs="*", help="only rebuild these modes (default: all)")
    a = ap.parse_args()
    banks = build(a.flops, a.seed, a.modes)
    for d in a.out:
        Path(d).mkdir(parents=True, exist_ok=True)
        for name, qs in banks.items():
            body = {"version": 2, "name": name, "questions": qs}
            (Path(d) / f"{name}.json").write_text(json.dumps(body, separators=(",", ":")))
    for name, qs in sorted(banks.items()):
        print(f"{name}: {len(qs)} questions")


if __name__ == "__main__":
    main()
