"""Rank the 169 holdem starting-hand classes by Monte-Carlo equity vs a random hand."""
import json
import os
import numpy as np
from card_encoding import RANKS
from hand_indexing import BINOMIAL, HOLDEM_7CARD_COMBOS
from optimized_evaluator import get_score_array, best5_of_7_numba

ORDER = 'AKQJT98765432'  # high -> low


def all_classes():
    out = []
    for i in range(13):
        out.append(ORDER[i] + ORDER[i])
        for k in range(i + 1, 13):
            out.append(ORDER[i] + ORDER[k] + 's')
            out.append(ORDER[i] + ORDER[k] + 'o')
    return out


def rep(cls):
    a = RANKS.index(cls[0]); b = RANKS.index(cls[1])
    if cls[0] == cls[1]:
        return [a * 4 + 0, a * 4 + 1]
    if cls.endswith('s'):
        return [a * 4 + 0, b * 4 + 0]
    return [a * 4 + 0, b * 4 + 1]


def equity(rep_cards, trials, rng, sa):
    hero = np.array(rep_cards, dtype=np.int32)
    deck0 = [c for c in range(52) if c not in rep_cards]
    wins = 0.0
    for _ in range(trials):
        pick = rng.choice(len(deck0), size=7, replace=False)
        cards = [deck0[k] for k in pick]
        opp = np.array(cards[:2], dtype=np.int32)
        board = np.array(cards[2:], dtype=np.int32)
        hs = best5_of_7_numba(np.concatenate((hero, board)), HOLDEM_7CARD_COMBOS, sa, BINOMIAL)
        os_ = best5_of_7_numba(np.concatenate((opp, board)), HOLDEM_7CARD_COMBOS, sa, BINOMIAL)
        wins += 1.0 if hs > os_ else (0.5 if hs == os_ else 0.0)
    return wins / trials


if __name__ == '__main__':
    sa = get_score_array()
    rng = np.random.default_rng(42)
    classes = all_classes()
    assert len(classes) == 169 and len(set(classes)) == 169
    scored = [(c, equity(rep(c), 2000, rng, sa)) for c in classes]
    scored.sort(key=lambda t: t[1], reverse=True)
    order = [c for c, _ in scored]
    out = os.path.join(os.path.dirname(__file__), 'holdem_class_order.json')
    with open(out, 'w') as f:
        json.dump(order, f)
    print('wrote', out, 'top:', order[:5])
