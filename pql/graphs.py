"""Equity *graph* computations: per-street series, hero-vs-range distribution, and
hero equity grouped by the villain's made category. Built on the existing sampler /
enumerator / evaluator. Consumed by the /pql-graph endpoint."""
from __future__ import annotations
import itertools
from math import comb
import numpy as np
from numba import njit

from hand_indexing import BINOMIAL, HOLDEM_7CARD_COMBOS
from optimized_evaluator import get_score_array, best_score_numba, best5_of_7_numba
from hand_categories import CATEGORY_TOKENS, CATEGORY_NAMES
from pql.scenario import Scenario
from pql.runtime.enumerate import enumerate_boards
from pql.runtime.sampler import sample_trials
from pql.runtime.evaluator import player_scores, player_categories
from pql.runtime.context import EvalContext
from pql.functions.values import river_equity

MAX_DIST_COMBOS = 4000
EXACT_RUNOUTS = 200
MC_RUNOUTS = 300


def _sample_or_enumerate(scenario: Scenario, trials: int, seed: int | None):
    """Return (player_hands (N,P,H), boards (N,5)) for a scenario, mirroring the
    executor: exact enumeration when there is no range and the runout is small, else
    Monte Carlo."""
    has_range = any(pl.pool is not None for pl in scenario.players)
    boards_only = None if has_range else enumerate_boards(scenario)
    if boards_only is not None:
        eff = boards_only.shape[0]
        H = scenario.game.num_hole
        hands = np.stack([pl.cards for pl in scenario.players])          # (P, H)
        player_hands = np.broadcast_to(hands, (eff, len(scenario.players), H)).copy()
        return player_hands, boards_only
    return sample_trials(scenario, trials, seed)


def _avg_equities(scenario: Scenario, trials: int, seed: int | None) -> np.ndarray:
    """Average riverEquity for every player on `scenario` (P,)."""
    player_hands, boards = _sample_or_enumerate(scenario, trials, seed)
    scores = player_scores(scenario, player_hands, boards)
    ctx = EvalContext(scenario=scenario, player_hands=player_hands, boards=boards, scores=scores)
    P = len(scenario.players)
    return np.array([float(river_equity(ctx, j, []).mean()) for j in range(P)])


def equity_by_street(scenario: Scenario, trials: int, seed: int | None) -> list[dict]:
    """One line series per player: all-in equity at each board prefix the board provides
    (preflop always; flop/turn/river as the board has >=3/4/5 cards)."""
    nb = len(scenario.board)
    prefixes = [("preflop", 0)]
    if nb >= 3:
        prefixes.append(("flop", 3))
    if nb >= 4:
        prefixes.append(("turn", 4))
    if nb >= 5:
        prefixes.append(("river", 5))
    series = [{"name": pl.name, "points": []} for pl in scenario.players]
    for label, k in prefixes:
        sub = Scenario(game=scenario.game, players=scenario.players,
                       board=scenario.board[:k], dead=scenario.dead)
        eqs = _avg_equities(sub, trials, seed)
        for j in range(len(scenario.players)):
            series[j]["points"].append({"street": label, "equity": float(eqs[j])})
    return series


@njit(cache=True)
def _equity_over_boards(hero, villain, boards, holdem, hand_combos, board_combos, sa, binom, combos21):
    """Hero's heads-up equity (win=1, tie=0.5) vs one villain hand over a set of 5-card boards."""
    R = boards.shape[0]
    acc = 0.0
    c7h = np.empty(7, dtype=np.int32)
    c7v = np.empty(7, dtype=np.int32)
    if holdem:
        c7h[0] = hero[0]; c7h[1] = hero[1]
        c7v[0] = villain[0]; c7v[1] = villain[1]
    for r in range(R):
        b = boards[r]
        if holdem:
            for k in range(5):
                c7h[2 + k] = b[k]
                c7v[2 + k] = b[k]
            hs = best5_of_7_numba(c7h, combos21, sa, binom)
            vs = best5_of_7_numba(c7v, combos21, sa, binom)
        else:
            hs = best_score_numba(hero, b, hand_combos, board_combos, sa, binom)
            vs = best_score_numba(villain, b, hand_combos, board_combos, sa, binom)
        if hs > vs:
            acc += 1.0
        elif hs == vs:
            acc += 0.5
    return acc / R


def _make_boards(board, deck, need, rng):
    """(R,5) completed boards: exact enumeration when comb(deck,need) <= EXACT_RUNOUTS,
    else MC_RUNOUTS random completions."""
    if need == 0:
        return board.reshape(1, 5).astype(np.int32)
    if comb(len(deck), need) <= EXACT_RUNOUTS:
        combs = np.array(list(itertools.combinations(deck.tolist(), need)), dtype=np.int32)
    else:
        keys = rng.random((MC_RUNOUTS, len(deck)))
        combs = deck[np.argsort(keys, axis=1)[:, :need]]
    nb = len(board)
    fixed = np.tile(board, (combs.shape[0], 1)).astype(np.int32) if nb else \
        np.empty((combs.shape[0], 0), dtype=np.int32)
    return np.concatenate([fixed, combs], axis=1).astype(np.int32)


def _bucketize(eqs: np.ndarray) -> list[dict]:
    buckets = []
    for b in range(10):
        lo, hi = b * 10, (b + 1) * 10
        if b < 9:
            m = (eqs >= lo / 100.0) & (eqs < hi / 100.0)
        else:
            m = (eqs >= lo / 100.0) & (eqs <= 1.0)   # last bucket includes exactly 100%
        buckets.append({"lo": lo, "hi": hi, "pct": float(m.mean()) if eqs.size else 0.0})
    return buckets


def equity_distribution(scenario: Scenario, hero_idx: int, trials: int, seed: int | None) -> dict:
    """Histogram of hero's equity across a villain range's combos (heads-up, hero fixed)."""
    g = scenario.game
    holdem = g.eval_kind == "holdem"
    hero_pl = scenario.players[hero_idx]
    if hero_pl.cards is None:
        raise ValueError("equity-distribution needs a hero with fixed hole cards")
    villains = [i for i in range(len(scenario.players)) if i != hero_idx]
    range_villains = [i for i in villains if scenario.players[i].pool is not None]
    if len(villains) != 1 or len(range_villains) != 1:
        raise ValueError("equity-distribution is heads-up and needs exactly one range opponent")
    pool = scenario.players[range_villains[0]].pool
    hero = hero_pl.cards.astype(np.int32)
    board = scenario.board.astype(np.int32)
    blocked = set(hero.tolist()) | set(board.tolist()) | set(scenario.dead.tolist())
    keep = [r for r in range(pool.shape[0]) if not (set(pool[r].tolist()) & blocked)]
    combos = pool[keep]
    if combos.shape[0] == 0:
        raise ValueError("the range has no combos left after removing hero/board/dead cards")
    total_combos = combos.shape[0]
    rng = np.random.default_rng(seed)
    sampled = None
    if combos.shape[0] > MAX_DIST_COMBOS:
        idx = rng.choice(combos.shape[0], MAX_DIST_COMBOS, replace=False)
        combos = combos[idx]
        sampled = MAX_DIST_COMBOS
    sa = get_score_array()
    need = 5 - len(board)
    base_deck_excl = set(hero.tolist()) | set(board.tolist()) | set(scenario.dead.tolist())
    eqs = np.empty(combos.shape[0], dtype=np.float64)
    for i in range(combos.shape[0]):
        c = combos[i].astype(np.int32)
        used = base_deck_excl | set(c.tolist())
        deck = np.array([x for x in range(52) if x not in used], dtype=np.int32)
        boards = _make_boards(board, deck, need, rng)
        eqs[i] = _equity_over_boards(hero, c, boards, holdem, g.hand_combos, g.board_combos, sa, BINOMIAL, HOLDEM_7CARD_COMBOS)
    return {
        "buckets": _bucketize(eqs),
        "mean": float(eqs.mean()),
        "combos": total_combos,
        "sampled_combos": sampled,
    }


def equity_vs_class(scenario: Scenario, hero_idx: int, trials: int, seed: int | None) -> dict:
    """Hero's heads-up equity grouped by the villain's river made category (heads-up)."""
    if len(scenario.players) != 2:
        raise ValueError("equity-vs-class is heads-up (exactly two players)")
    villain_idx = 1 - hero_idx
    player_hands, boards = _sample_or_enumerate(scenario, trials, seed)
    scores = player_scores(scenario, player_hands, boards)         # (N, 2)
    cats = player_categories(scenario, player_hands, boards)       # (N, 2)
    hs = scores[:, hero_idx]
    vs = scores[:, villain_idx]
    outcome = np.where(hs > vs, 1.0, np.where(hs == vs, 0.5, 0.0))
    vcat = cats[:, villain_idx]
    n = outcome.shape[0]
    rows = []
    for cat in range(9):
        m = vcat == cat
        cnt = int(m.sum())
        if cnt == 0:
            continue
        rows.append({
            "category": CATEGORY_TOKENS[cat],
            "label": CATEGORY_NAMES[cat],
            "equity": float(outcome[m].mean()),
            "freq": cnt / n,
        })
    return {"rows": rows}
