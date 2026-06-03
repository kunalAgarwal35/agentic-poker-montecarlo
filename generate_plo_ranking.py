"""Rank suit-isomorphic PLO starting-hand classes by Monte-Carlo equity vs one random hand,
and write a premium-first class-order JSON per variant (mirrors generate_holdem_ranking.py).

Usage:
  python generate_plo_ranking.py --variant plo4 --trials 1500
  python generate_plo_ranking.py --variant plo5 --trials 1200
  python generate_plo_ranking.py --variant plo6 --trials 1000   # the long pole
"""
import argparse
import json
import os
from itertools import combinations
import numpy as np

from card_encoding import int_to_card
from hand_indexing import BINOMIAL
from optimized_evaluator import get_score_array, best_score_numba
from pql.games import get_game
from pql.ranges.plo_classes import canon_key

VARIANT_GAME = {'plo4': 'omahahi', 'plo5': 'omahahi5', 'plo6': 'omahahi6'}


def enumerate_classes(num_hole):
    keys = {}
    for combo in combinations(range(52), num_hole):
        k = canon_key(combo)
        if k not in keys:
            keys[k] = True
    return list(keys)


def equity(rep, g, trials, rng, sa):
    hero = np.array(rep, dtype=np.int32)
    repset = set(rep)
    deck0 = [c for c in range(52) if c not in repset]
    H = g.num_hole
    wins = 0.0
    for _ in range(trials):
        pick = rng.choice(len(deck0), size=H + 5, replace=False)
        cards = [deck0[k] for k in pick]
        opp = np.array(cards[:H], dtype=np.int32)
        board = np.array(cards[H:H + 5], dtype=np.int32)
        hs = best_score_numba(hero, board, g.hand_combos, g.board_combos, sa, BINOMIAL)
        oscore = best_score_numba(opp, board, g.hand_combos, g.board_combos, sa, BINOMIAL)
        wins += 1.0 if hs > oscore else (0.5 if hs == oscore else 0.0)
    return wins / trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=list(VARIANT_GAME))
    ap.add_argument('--trials', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    g = get_game(VARIANT_GAME[args.variant])
    sa = get_score_array()
    rng = np.random.default_rng(args.seed)
    classes = enumerate_classes(g.num_hole)
    print(f'{args.variant}: {len(classes)} classes; ranking at {args.trials} trials...')
    scored = [(rep, equity(rep, g, args.trials, rng, sa)) for rep in classes]
    scored.sort(key=lambda t: t[1], reverse=True)
    order = [''.join(int_to_card(c) for c in rep) for rep, _ in scored]
    out = os.path.join(os.path.dirname(__file__), f'{args.variant}_class_order.json')
    with open(out, 'w') as f:
        json.dump(order, f)
    print('wrote', out, 'top:', order[:5])


if __name__ == '__main__':
    main()
