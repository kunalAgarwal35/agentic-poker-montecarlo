import numpy as np
from itertools import combinations
from pql import run_pql


def test_nut_score_kernel_matches_bruteforce_holdem():
    from optimized_evaluator import get_score_array, best5_of_7_numba, nut_score_holdem_numba
    from hand_indexing import BINOMIAL, HOLDEM_7CARD_COMBOS
    sa = get_score_array()
    board = np.array([0, 5, 10, 20, 30], dtype=np.int32)
    deck = np.array([c for c in range(52) if c not in board.tolist()], dtype=np.int32)
    got = nut_score_holdem_numba(board, deck, HOLDEM_7CARD_COMBOS, sa, BINOMIAL)
    brute = 0.0
    for i, j in combinations(range(len(deck)), 2):
        c7 = np.array([deck[i], deck[j], *board.tolist()], dtype=np.int32)
        brute = max(brute, best5_of_7_numba(c7, HOLDEM_7CARD_COMBOS, sa, BINOMIAL))
    assert got == brute


def test_nuthi_deterministic_holdem():
    # Board AhKhQhJh2c. P1=Th9h makes the royal flush (the nuts). P2=AsAd makes trips.
    q = ("select count(nutHi(PLAYER_1)) as p1_nuts, count(nutHi(PLAYER_2)) as p2_nuts "
         "from game='holdem', board='AhKhQhJh2c', PLAYER_1='Th9h', PLAYER_2='AsAd'")
    r = run_pql(q, seed=1)
    assert r.mode == 'enumeration' and r.trials == 1
    assert r.values['p1_nuts'] == 1.0
    assert r.values['p2_nuts'] == 0.0


def test_nuthi_omaha_runs_and_bounded():
    q = ("select count(nutHi(PLAYER_1)) as p1_nuts "
         "from game='omahahi5', board='2c3c4c5c', PLAYER_1='AsAhKsKhQs', PLAYER_2='JdTd9d8d7d'")
    r = run_pql(q, seed=1)
    assert 0 <= r.values['p1_nuts'] <= r.trials
