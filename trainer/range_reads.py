"""Range Reads: exact "how often does villain's range have X on this flop" questions.

Villain ranges are the GTO raise-first-in (RFI) ranges from the PLO Trainer's
solver data (PLO50, 100BB, 6-max), stored in data/gto_rfi_plo50_100bb.npz as
(cards uint8 (M,4), weights float32 (M,)). Cards are rank*4 + suit with
rank 0..12 = 2..A and suit 0..3 = s,h,d,c.

Every answer is an EXACT weighted count over all range combos that don't
conflict with the flop -- no Monte Carlo -- because each question only depends
on the flop:

    answer = sum(weight of matching combos) / sum(weight of board-compatible combos)

Definitions (the wording shown to players must match these):
  * set            -- a hole pair matching a board rank (unpaired boards only)
  * straight       -- villain's best hand is a straight or straight flush
  * flush draw     -- two-tone flop; villain holds >= 2 cards of the board suit
  * nut flush draw -- as above, one of them the highest card of that suit not on the board
  * straight outs  -- turn cards (not in villain's hand or on the flop) that give a
                      straight villain doesn't already have; 8+ = OESD-class, 13+ = big wrap
  * nuts           -- villain's hand ties the best hand possible on this flop
  * air            -- no pair or better, no flush draw, fewer than 8 straight outs
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path

import numpy as np

RANKS = "23456789TJQKA"
SUITS = "shdc"
DATA = Path(__file__).parent / "data" / "gto_rfi_plo50_100bb.npz"

POSITIONS = {
    "EP": "UTG",
    "MP": "MP",
    "CO": "CO",
    "BU": "BTN",
    "SB": "SB",
}

HIGH, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)

# 5-bit straight masks over ranks 0..12 (wheel = A,2,3,4,5).
STRAIGHT_MASKS = {(0b11111 << lo): lo + 4 for lo in range(9)}
STRAIGHT_MASKS[(1 << 12) | 0b1111] = 3  # wheel, high card = 5 (rank idx 3)
_SMASK_ARR = np.array(sorted(STRAIGHT_MASKS), dtype=np.int64)
_SHIGH_ARR = np.array([STRAIGHT_MASKS[m] for m in sorted(STRAIGHT_MASKS)], dtype=np.int64)


def card(s: str) -> int:
    return RANKS.index(s[0].upper()) * 4 + SUITS.index(s[1].lower())


def card_str(c: int) -> str:
    return RANKS[c // 4] + SUITS[c % 4]


def load_ranges(path: Path = DATA) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    z = np.load(path)
    return {p: (z[f"{p}_cards"].astype(np.int64), z[f"{p}_weights"].astype(np.float64)) for p in POSITIONS}


# --------------------------------------------------------------------------
# Vectorized 5-card scoring
# --------------------------------------------------------------------------

def score5(cards: np.ndarray) -> np.ndarray:
    """cards: (N,5) ints -> (N,) int64 scores; higher is better. Category = score // 15**5."""
    r = cards // 4
    s = cards % 4
    flush = (s == s[:, :1]).all(axis=1)
    mask = np.bitwise_or.reduce(np.left_shift(1, r), axis=1)
    distinct = (r[:, :, None] == r[:, None, :]).sum(axis=2)  # count of each card's rank
    idx = np.searchsorted(_SMASK_ARR, mask)
    idx = np.clip(idx, 0, len(_SMASK_ARR) - 1)
    is_straight = _SMASK_ARR[idx] == mask
    straight_high = np.where(is_straight, _SHIGH_ARR[idx], 0)

    # Order ranks by (count desc, rank desc) for kicker comparison.
    key = distinct * 16 + r
    order = np.argsort(-key, axis=1, kind="stable")
    ordered = np.take_along_axis(r, order, axis=1)
    counts = np.sort(distinct, axis=1)[:, ::-1]  # e.g. [3,3,3,1,1]
    kick = np.zeros(len(cards), dtype=np.int64)
    for k in range(5):
        kick = kick * 15 + ordered[:, k]

    c0, c3 = counts[:, 0], counts[:, 3]
    cat = np.full(len(cards), HIGH, dtype=np.int64)
    cat = np.where(c0 == 2, PAIR, cat)
    cat = np.where((c0 == 2) & (counts[:, 2] == 2), TWO_PAIR, cat)
    cat = np.where(c0 == 3, TRIPS, cat)
    cat = np.where(is_straight, STRAIGHT, cat)
    cat = np.where(flush, FLUSH, cat)
    cat = np.where((c0 == 3) & (c3 == 2), FULL_HOUSE, cat)
    cat = np.where(c0 == 4, QUADS, cat)
    cat = np.where(is_straight & flush, STRAIGHT_FLUSH, cat)
    body = np.where(is_straight, straight_high, kick)
    return cat * 15 ** 5 + body


HOLE_PAIRS = list(combinations(range(4), 2))


def best_flop_score(hands: np.ndarray, flop: list[int]) -> np.ndarray:
    """Omaha: exactly 2 hole cards + the 3 flop cards. hands (M,4) -> (M,) best score."""
    f = np.broadcast_to(np.array(flop, dtype=np.int64), (len(hands), 3))
    best = None
    for i, j in HOLE_PAIRS:
        v = score5(np.concatenate([hands[:, [i, j]], f], axis=1))
        best = v if best is None else np.maximum(best, v)
    return best


def nut_score(flop: list[int]) -> int:
    """Best Omaha hand possible on this flop (any two unseen hole cards)."""
    deck = [c for c in range(52) if c not in flop]
    pairs = np.array(list(combinations(deck, 2)), dtype=np.int64)
    five = np.concatenate([pairs, np.broadcast_to(np.array(flop), (len(pairs), 3))], axis=1)
    return int(score5(five).max())


# --------------------------------------------------------------------------
# Per-combo flop features
# --------------------------------------------------------------------------

def straight_outs(hands: np.ndarray, flop: list[int], made_straight: np.ndarray) -> np.ndarray:
    """Number of unseen turn cards giving villain a straight they don't already have."""
    hr = hands // 4
    fr = [c // 4 for c in flop]
    outs = np.zeros(len(hands), dtype=np.int64)
    for t in range(13):
        board4 = fr + [t]
        triples = list(combinations(board4, 3))
        makes = np.zeros(len(hands), dtype=bool)
        for i, j in HOLE_PAIRS:
            hm = np.left_shift(1, hr[:, i]) | np.left_shift(1, hr[:, j])
            for tri in triples:
                bm = (1 << tri[0]) | (1 << tri[1]) | (1 << tri[2])
                m = hm | bm
                # 5 distinct ranks and a straight shape
                five_distinct = (hr[:, i] != hr[:, j]) & ((hm & bm) == 0) & (len(set(tri)) == 3)
                idx = np.clip(np.searchsorted(_SMASK_ARR, m), 0, len(_SMASK_ARR) - 1)
                makes |= five_distinct & (_SMASK_ARR[idx] == m)
        live = 4 - (hr == t).sum(axis=1) - fr.count(t)
        outs += np.where(makes & ~made_straight, np.maximum(live, 0), 0)
    return outs


def flop_features(hands: np.ndarray, flop: list[int], nuts: int | None = None) -> dict[str, np.ndarray]:
    """Boolean/int feature arrays for every combo (hands must not conflict with the flop)."""
    hr, hs = hands // 4, hands % 4
    fr, fs = [c // 4 for c in flop], [c % 4 for c in flop]
    best = best_flop_score(hands, flop)
    cat = best // 15 ** 5
    made_straight = (cat == STRAIGHT) | (cat == STRAIGHT_FLUSH)

    # Set: a hole pair whose rank is on the (unpaired) flop.
    has_set = np.zeros(len(hands), dtype=bool)
    top_set = np.zeros(len(hands), dtype=bool)
    for r in set(fr):
        n = (hr == r).sum(axis=1)
        has_set |= n >= 2
        if r == max(fr):
            top_set |= n >= 2

    # Flush draws (two-tone flops only).
    suit_counts = {s: fs.count(s) for s in set(fs)}
    fd = np.zeros(len(hands), dtype=bool)
    nfd = np.zeros(len(hands), dtype=bool)
    for s, n in suit_counts.items():
        if n != 2:
            continue
        held = (hs == s)
        fd |= held.sum(axis=1) >= 2
        on_board = {fr[k] for k in range(3) if fs[k] == s}
        top = max(r for r in range(13) if r not in on_board)
        has_top = (held & (hr == top)).any(axis=1)
        nfd |= has_top & (held.sum(axis=1) >= 2)

    outs = straight_outs(hands, flop, made_straight)
    if nuts is None:
        nuts = nut_score(flop)
    return {
        "category": cat,
        "set": has_set,
        "top_set": top_set,
        "straight": made_straight,
        "two_pair_plus": cat >= TWO_PAIR,
        "flush": cat >= FLUSH,
        "flush_draw": fd & (cat < FLUSH),
        "nut_flush_draw": nfd & (cat < FLUSH),
        "straight_outs": outs,
        "nuts": best == nuts,
        "air": (cat == HIGH) & ~fd & (outs < 8),
        "combo_draw": fd & (outs >= 8) & (cat < STRAIGHT),
    }


# --------------------------------------------------------------------------
# Board textures + templates
# --------------------------------------------------------------------------

def texture(flop: list[int]) -> dict:
    fr = sorted((c // 4 for c in flop), reverse=True)
    suits = [c % 4 for c in flop]
    ns = len(set(suits))
    paired = len(set(fr)) < 3
    # A straight is possible if some 5-rank window contains all 3 distinct ranks.
    bits = 0
    for r in fr:
        bits |= 1 << r
    straight_possible = not paired and any((m & bits) == bits for m in STRAIGHT_MASKS)
    return {
        "suits": {1: "monotone", 2: "two-tone", 3: "rainbow"}[ns],
        "paired": paired,
        "straight_possible": straight_possible,
        "high": RANKS[fr[0]],
    }


@dataclass(frozen=True)
class Template:
    id: str
    prompt: str            # "{pos}" is filled with the position label
    feature: str
    explain: str
    applies: callable = lambda t: True
    threshold: int | None = None  # for integer features (straight outs)


TEMPLATES = [
    Template("flush_draw", "How often does {pos}'s range have a flush draw?", "flush_draw",
             "Any two cards of the board's two-tone suit (made flushes excluded).",
             lambda t: t["suits"] == "two-tone"),
    Template("nut_flush_draw", "How often does {pos}'s range have the NUT flush draw?", "nut_flush_draw",
             "Two+ cards of the board suit including the highest one not on the board.",
             lambda t: t["suits"] == "two-tone"),
    Template("set", "How often has {pos}'s range flopped a set?", "set",
             "A pocket pair matching one of the board cards.",
             lambda t: not t["paired"]),
    Template("top_set", "How often has {pos}'s range flopped top set?", "top_set",
             "A pocket pair matching the highest board card.",
             lambda t: not t["paired"]),
    Template("straight", "How often has {pos}'s range flopped a straight?", "straight",
             "Best hand is already a straight (or straight flush).",
             lambda t: t["straight_possible"]),
    Template("two_pair_plus", "How often does {pos}'s range have two pair or better?", "two_pair_plus",
             "Two pair, sets, straights, flushes and better.",
             lambda t: not t["paired"]),
    Template("made_flush", "How often has {pos}'s range flopped a flush?", "flush",
             "Two hole cards of the board's suit (or better).",
             lambda t: t["suits"] == "monotone"),
    Template("oesd_plus", "How often does {pos}'s range have 8+ straight outs?", "straight_outs",
             "8 or more unseen turn cards make a straight it doesn't already have (OESD, wraps).",
             lambda t: t["straight_possible"] or not t["paired"], threshold=8),
    Template("big_wrap", "How often does {pos}'s range have a 13+ out wrap?", "straight_outs",
             "13 or more unseen turn cards make a straight it doesn't already have.",
             lambda t: not t["paired"], threshold=13),
    Template("combo_draw", "How often does {pos}'s range have a flush draw AND 8+ straight outs?", "combo_draw",
             "Flush draw plus 8+ straight outs (no made straight yet).",
             lambda t: t["suits"] == "two-tone" and not t["paired"]),
    Template("nuts", "How often does {pos}'s range have the nuts right now?", "nuts",
             "Ties the best possible hand on this flop.",
             lambda t: True),
    Template("air", "How often does {pos}'s range have nothing (no pair, no flush draw, under 8 straight outs)?", "air",
             "No pair or better, no flush draw, fewer than 8 straight outs.",
             lambda t: True),
]


def answer(feat: np.ndarray, weights: np.ndarray, threshold: int | None) -> float:
    hit = feat >= threshold if threshold is not None else feat.astype(bool)
    return float(weights[hit].sum() / weights.sum())


def questions_for_flop(ranges, flop: list[int]) -> list[dict]:
    t = texture(flop)
    nuts = nut_score(flop)
    out = []
    for pos, (cards, weights) in ranges.items():
        ok = ~np.isin(cards, flop).any(axis=1)
        h, w = cards[ok], weights[ok]
        feats = flop_features(h, flop, nuts)
        for tpl in TEMPLATES:
            if not tpl.applies(t):
                continue
            a = answer(feats[tpl.feature], w, tpl.threshold)
            flop_s = "".join(card_str(c) for c in flop)
            out.append({
                "id": f"plo4-rfi-{pos}-{flop_s}-{tpl.id}",
                "game": "PLO4",
                "spot": "PLO50 100BB 6-max, raise first in",
                "position": POSITIONS[pos],
                "board": [card_str(c) for c in flop],
                "template": tpl.id,
                "prompt": tpl.prompt.format(pos=POSITIONS[pos]),
                "definition": tpl.explain,
                "answer": round(a * 100, 2),
                "texture": t,
            })
    return out


def sample_flops(n: int, seed: int = 7) -> list[list[int]]:
    """Deterministic, texture-balanced flops: rainbow / two-tone / monotone / paired."""
    rng = np.random.default_rng(seed)
    want = {"rainbow": 0.35, "two-tone": 0.40, "monotone": 0.10, "paired": 0.15}
    quota = {k: max(1, round(v * n)) for k, v in want.items()}
    seen, out = set(), []
    while any(quota.values()):
        flop = sorted(rng.choice(52, size=3, replace=False).tolist(), reverse=True)
        key = tuple(flop)
        if key in seen:
            continue
        t = texture(flop)
        bucket = "paired" if t["paired"] else t["suits"]
        if quota.get(bucket, 0) <= 0:
            continue
        quota[bucket] -= 1
        seen.add(key)
        out.append(flop)
    return out


def build_bank(n_flops: int = 60, seed: int = 7) -> dict:
    ranges = load_ranges()
    qs = []
    for flop in sample_flops(n_flops, seed):
        qs.extend(questions_for_flop(ranges, flop))
    # Drop degenerate questions whose answer is 0 -- they teach nothing and
    # read as trick questions (e.g. a set on a board where none is possible).
    qs = [q for q in qs if q["answer"] > 0]
    return {
        "version": 1,
        "source": "GTO RFI ranges, PLO50 100BB 6-max (exact weighted combo counts)",
        "questions": qs,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--flops", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bank = build_bank(a.flops, a.seed)
    Path(a.out).write_text(json.dumps(bank, separators=(",", ":")))
    print(f"wrote {len(bank['questions'])} questions to {a.out}")
