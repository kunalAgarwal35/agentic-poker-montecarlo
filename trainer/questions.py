"""Question generators for the PLO Trainer's training modes.

Every mode emits questions in one shape (see `_q`):
  kind='estimate': the player guesses `answer` (a percent)
  kind='choice'  : the player picks `choices[correct]`

Modes
  range_reads    How often does {pos}'s opening range have X on this flop?   exact
  card_removal   Same, but you hold a hand: villain combos using your cards are
                 removed. Shows the no-blocker baseline after answering.        exact
  range_vs_range Which of two opening ranges has X more often on this flop?   exact
  hero_equity    Your hand's equity vs {pos}'s opening range on this flop.     Monte Carlo
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from trainer.fasteval import evaluate
from trainer.features import card_str, flop_features, nut_score, texture
from trainer.ranges import POSITIONS, SPOT, ranges

# --------------------------------------------------------------------------
# Hand-type templates
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Template:
    id: str
    noun: str              # "a flush draw" -> used in every mode's wording
    feature: str
    explain: str
    applies: Callable[[dict], bool] = lambda t: True
    threshold: int | None = None

    def hit(self, feats: dict) -> np.ndarray:
        f = feats[self.feature]
        return f >= self.threshold if self.threshold is not None else f.astype(bool)


TEMPLATES = [
    Template("flush_draw", "a flush draw", "flush_draw",
             "Two or more cards of the board's two-tone suit (made flushes excluded).",
             lambda t: t["suits"] == "two-tone"),
    Template("nut_flush_draw", "the nut flush draw", "nut_flush_draw",
             "Two or more cards of the board suit, including the highest one not on the board.",
             lambda t: t["suits"] == "two-tone"),
    Template("set", "a set", "set", "A pocket pair matching one of the board cards.",
             lambda t: not t["paired"]),
    Template("top_set", "top set", "top_set", "A pocket pair matching the highest board card.",
             lambda t: not t["paired"]),
    Template("straight", "a straight", "straight", "Best hand is already a straight (or straight flush).",
             lambda t: t["straight_possible"]),
    Template("two_pair_plus", "two pair or better", "two_pair_plus",
             "Two pair, sets, straights, flushes and better.", lambda t: not t["paired"]),
    Template("made_flush", "a flush", "flush", "Two hole cards of the board's suit (or better).",
             lambda t: t["suits"] == "monotone"),
    Template("oesd_plus", "8+ straight outs", "straight_outs",
             "8 or more unseen turn cards make a straight it doesn't already have (OESD, wraps).",
             lambda t: not t["paired"], threshold=8),
    Template("big_wrap", "a 13+ out wrap", "straight_outs",
             "13 or more unseen turn cards make a straight it doesn't already have.",
             lambda t: not t["paired"], threshold=13),
    Template("combo_draw", "a flush draw plus 8+ straight outs", "combo_draw",
             "Flush draw plus 8+ straight outs (no made straight yet).",
             lambda t: t["suits"] == "two-tone" and not t["paired"]),
    Template("nuts", "the nuts", "nuts", "Ties the best possible hand on this flop."),
    Template("air", "nothing (no pair, no flush draw, under 8 straight outs)", "air",
             "No pair or better, no flush draw, fewer than 8 straight outs."),
]
TEMPLATE_BY_ID = {t.id: t for t in TEMPLATES}

# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _board(flop):
    return [card_str(c) for c in flop]


def _hand(h):
    return [card_str(int(c)) for c in sorted(h, key=lambda c: (-(c // 4), c % 4))]


def _q(mode, game, flop, key, **kw) -> dict:
    t = texture(flop)
    return {
        "id": f"{mode}-{game.lower()}-{''.join(_board(flop))}-{key}",
        "mode": mode,
        "game": game,
        "spot": SPOT[game],
        "board": _board(flop),
        "texture": t["suits"] + (" paired" if t["paired"] else ""),
        **kw,
    }


class FlopRanges:
    """Per-(game, flop) cache: board-compatible combos, weights and features per position."""

    def __init__(self, game: str, flop: list[int]):
        self.game, self.flop = game, flop
        self.nuts = nut_score(flop)
        self.data = {}
        rs = ranges(game)
        widest = max(rs, key=lambda p: len(rs[p][0]))
        wc, ww_all = rs[widest]
        nested = all(np.array_equal(rs[p][0], wc[:len(rs[p][0])]) for p in rs)
        if nested:
            # Assumed PLO5 ranges are prefixes of one ranking: analyse the widest
            # range once and slice it for the others.
            ok = ~np.isin(wc, flop).any(axis=1)
            feats = flop_features(wc[ok], flop, self.nuts)
            n_ok = np.cumsum(ok)
            for pos, (cards, w) in rs.items():
                k = int(n_ok[len(cards) - 1])
                self.data[pos] = (wc[ok][:k], ww_all[ok][:k], {f: v[:k] for f, v in feats.items()})
            return
        for pos, (cards, w) in rs.items():
            ok = ~np.isin(cards, flop).any(axis=1)
            h, ww = cards[ok], w[ok]
            self.data[pos] = (h, ww, flop_features(h, flop, self.nuts))

    def freq(self, pos: str, tpl: Template, block: list[int] | None = None) -> float:
        h, w, f = self.data[pos]
        hit = tpl.hit(f)
        if block:
            keep = ~np.isin(h, block).any(axis=1)
            w = w * keep
        return float(w[hit].sum() / w.sum() * 100)


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------


def range_reads(fr: FlopRanges) -> list[dict]:
    out = []
    t = texture(fr.flop)
    for pos in POSITIONS:
        for tpl in TEMPLATES:
            if not tpl.applies(t):
                continue
            a = fr.freq(pos, tpl)
            if a < 0.1:  # unguessably rare (or impossible): teaches nothing
                continue
            out.append(_q("range_reads", fr.game, fr.flop, f"{pos}-{tpl.id}",
                          kind="estimate", template=tpl.id, position=POSITIONS[pos],
                          prompt=f"How often does {POSITIONS[pos]}'s range have {tpl.noun}?",
                          definition=tpl.explain, answer=round(a, 2)))
    return out


def card_removal(fr: FlopRanges, rng: np.random.Generator, per_flop: int = 6, n_heroes: int = 120) -> list[dict]:
    """Pick hero hands whose cards move villain's frequency the most.

    The search runs on a <=200k-combo subsample per position (one blocker mask
    per hero, shared by every template); the chosen hand's numbers are then
    recomputed exactly on the full range."""
    t = texture(fr.flop)
    tpls = [tp for tp in TEMPLATES if tp.applies(t) and tp.id != "air"]
    hero_pool, _ = ranges(fr.game)["CO"]
    hero_pool = hero_pool[~np.isin(hero_pool, fr.flop).any(axis=1)]
    heroes = hero_pool[rng.choice(len(hero_pool), size=min(n_heroes, len(hero_pool)), replace=False)]
    cands = []
    for pos in POSITIONS:
        h, w, f = fr.data[pos]
        if len(h) > 200_000:
            pick = rng.choice(len(h), size=200_000, replace=False)
            h, w, f = h[pick], w[pick], {k: v[pick] for k, v in f.items()}
        hits = {tp.id: tp.hit(f) for tp in tpls}
        base = {k: w[v].sum() / w.sum() * 100 for k, v in hits.items()}
        best: dict[str, tuple] = {}
        for hero in heroes:
            keep = ~np.isin(h, hero).any(axis=1)
            wk = w * keep
            tot = wk.sum()
            for tp in tpls:
                b = base[tp.id]
                if b < 1:
                    continue
                d = wk[hits[tp.id]].sum() / tot * 100 - b
                if abs(d) < 1.5 or abs(d) / b < 0.25:
                    continue
                # Partial effects teach more than total blockouts (holding the
                # A of the suit -> 0% nut flush draws), so those rank lower.
                score = abs(d) / b if (b + d) > 0.05 else 0.3
                if score > best.get(tp.id, (0,))[0]:
                    best[tp.id] = (score, hero)
        for tid, (score, hero) in best.items():
            cands.append((score, pos, TEMPLATE_BY_ID[tid], hero))
    cands.sort(key=lambda c: -c[0])
    out, seen, used_heroes, blockouts = [], set(), set(), 0
    for _, pos, tpl, hero in cands:
        key = tuple(hero.tolist())
        if tpl.id in seen or key in used_heroes or len(out) >= per_flop:
            continue
        base, with_ = fr.freq(pos, tpl), fr.freq(pos, tpl, hero.tolist())
        if abs(with_ - base) < 1.5:  # subsample noise; skip
            continue
        if with_ < 0.05:
            if blockouts:
                continue
            blockouts += 1
        seen.add(tpl.id)
        used_heroes.add(key)
        hero_s = _hand(hero)
        out.append(_q("card_removal", fr.game, fr.flop, f"{pos}-{tpl.id}-{''.join(hero_s)}",
                      kind="estimate", template=tpl.id, position=POSITIONS[pos], hero=hero_s,
                      prompt=f"You hold this hand. How often does {POSITIONS[pos]}'s range have {tpl.noun} now?",
                      definition=tpl.explain + " Villain combos that use any of your cards are impossible.",
                      answer=round(with_, 2), baseline=round(base, 2)))
    return out


RVR_PAIRS = [("EP", "BU"), ("MP", "BU"), ("EP", "CO"), ("CO", "BU"), ("SB", "BU"), ("EP", "SB")]


def range_vs_range(fr: FlopRanges, rng: np.random.Generator, per_flop: int = 6) -> list[dict]:
    t = texture(fr.flop)
    cands = []
    for a, b in RVR_PAIRS:
        for tpl in TEMPLATES:
            if not tpl.applies(t):
                continue
            va, vb = fr.freq(a, tpl), fr.freq(b, tpl)
            hi, lo = max(va, vb), min(va, vb)
            # Must be a real difference, not a coin flip.
            if hi >= 1 and (hi - lo) >= 1 and (hi - lo) / hi >= 0.2:
                cands.append(((hi - lo) / hi, a, b, tpl, va, vb))
    rng.shuffle(cands)
    out, seen = [], set()
    for _, a, b, tpl, va, vb in cands:
        if tpl.id in seen or len(out) >= per_flop:
            continue
        seen.add(tpl.id)
        if rng.random() < 0.5:  # don't always list the tighter range first
            a, b, va, vb = b, a, vb, va
        names = [POSITIONS[a], POSITIONS[b]]
        out.append(_q("range_vs_range", fr.game, fr.flop, f"{a}-{b}-{tpl.id}",
                      kind="choice", template=tpl.id, positions=names,
                      prompt=f"Whose opening range has {tpl.noun} more often on this flop?",
                      definition=tpl.explain, choices=names, correct=0 if va > vb else 1,
                      values=[round(va, 2), round(vb, 2)]))
    return out


def equity_vs_range(hero: np.ndarray, cards: np.ndarray, w: np.ndarray, flop: list[int],
                    trials: int, rng: np.random.Generator) -> float:
    """Hero equity (percent, ties split) vs a weighted range on a flop, Monte Carlo."""
    ok = ~np.isin(cards, np.concatenate([hero, flop])).any(axis=1)
    cards, w = cards[ok], w[ok]
    vill = cards[rng.choice(len(cards), size=trials, p=w / w.sum())]
    used = np.zeros((trials, 52), dtype=bool)
    used[:, list(hero) + list(flop)] = True
    np.put_along_axis(used, vill, True, axis=1)
    keys = rng.random((trials, 52))
    keys[used] = 2.0
    runout = np.argpartition(keys, 2, axis=1)[:, :2]
    boards = np.concatenate([np.broadcast_to(np.array(flop), (trials, 3)), runout], axis=1)
    hs, _ = evaluate(np.broadcast_to(hero, (trials, len(hero))), boards)
    vs, _ = evaluate(vill, boards)
    return float(((hs > vs) + 0.5 * (hs == vs)).mean() * 100)


def hero_equity(fr: FlopRanges, rng: np.random.Generator, per_flop: int = 4, trials: int = 40000) -> list[dict]:
    """Hero hands drawn from the BTN opening range, spread across equity bands."""
    hero_pool, _ = ranges(fr.game)["BU"]
    hero_pool = hero_pool[~np.isin(hero_pool, fr.flop).any(axis=1)]
    positions = list(POSITIONS)
    rng.shuffle(positions)
    bands = [(0, 35), (35, 55), (55, 75), (75, 101)]
    out = []
    for pos in positions:
        if len(out) >= per_flop:
            break
        cards, w = ranges(fr.game)[pos]
        for _ in range(12):  # find a hero hand in an unfilled band
            hero = hero_pool[rng.integers(len(hero_pool))]
            eq = equity_vs_range(hero, cards, w, fr.flop, 4000, rng)
            band = next(i for i, (lo, hi) in enumerate(bands) if lo <= eq < hi)
            if band in {q["band"] for q in out}:
                continue
            eq = equity_vs_range(hero, cards, w, fr.flop, trials, rng)
            hero_s = _hand(hero)
            q = _q("hero_equity", fr.game, fr.flop, f"{pos}-{''.join(hero_s)}",
                   kind="estimate", template="equity", position=POSITIONS[pos], hero=hero_s,
                   prompt=f"{POSITIONS[pos]} opened and you see this flop. What is your equity against their range?",
                   definition=f"All-in equity vs {POSITIONS[pos]}'s whole opening range, turn and river to come "
                              f"(Monte Carlo, {trials // 1000}k runouts, about ±0.5 pts).",
                   answer=round(eq, 1))
            q["band"] = band
            out.append(q)
            break
    for q in out:
        q.pop("band")
    return out


MODES = {
    "range_reads": lambda fr, rng: range_reads(fr),
    "card_removal": card_removal,
    "range_vs_range": range_vs_range,
    "hero_equity": hero_equity,
}
