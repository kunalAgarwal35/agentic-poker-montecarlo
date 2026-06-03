import numpy as np
from pql.functions.registry import get_value_fn
from pql.functions import values
from pql import run_pql


class _Ctx:
    def __init__(self, scores, cats):
        self.scores = np.asarray(scores, dtype=float)
        self._c = np.asarray(cats, dtype=np.int64)
    def categories(self):
        return self._c


def test_winning_hand_type_unit():
    # 2 trials, 2 players. T0: P0 wins with flush(5). T1: P1 wins with pair(1).
    ctx = _Ctx(scores=[[10.0, 5.0], [3.0, 9.0]], cats=[[5, 1], [2, 1]])
    win_flush = get_value_fn('winninghandtype')(ctx, 0, ['river', 'flush'])
    assert win_flush.tolist() == [1.0, 0.0]
    win_pair = get_value_fn('winninghandtype')(ctx, 0, ['river', 'pair'])
    assert win_pair.tolist() == [0.0, 1.0]


def test_players_with_best_hi_unit():
    ctx = _Ctx(scores=[[10.0, 5.0], [7.0, 7.0]], cats=[[1, 1], [1, 1]])
    pw = get_value_fn('playerswithbesthi')(ctx, 0, [])
    assert pw.tolist() == [1.0, 2.0]   # T0 sole winner, T1 a 2-way tie


def test_winning_distribution_sums_to_trials():
    from hand_categories import CATEGORY_TOKENS
    cols = ", ".join(f"count(winningHandType(PLAYER_1, river, {t})) as win_{t}" for t in CATEGORY_TOKENS)
    q = (f"select {cols} from game='holdem', board='2c3c4c5c', "
         f"PLAYER_1='AsKs', PLAYER_2='QdQh'")
    r = run_pql(q, seed=1)
    assert sum(r.values[f'win_{t}'] for t in CATEGORY_TOKENS) == r.trials
