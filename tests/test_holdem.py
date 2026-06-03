import numpy as np
from itertools import combinations
from hand_indexing import BINOMIAL, hand_to_index, HOLDEM_7CARD_COMBOS
from optimized_evaluator import get_score_array, best5_of_7_numba
from pql import run_pql


def test_holdem_combos_shape():
    assert HOLDEM_7CARD_COMBOS.shape == (21, 5)  # C(7,5)


def test_best5_of_7_matches_bruteforce():
    sa = get_score_array()
    rng = np.random.default_rng(0)
    for _ in range(50):
        cards7 = rng.choice(52, size=7, replace=False).astype(np.int32)
        brute = max(
            sa[hand_to_index(np.array(c, dtype=np.int32))]
            for c in combinations(cards7.tolist(), 5)
        )
        got = best5_of_7_numba(cards7, HOLDEM_7CARD_COMBOS, sa, BINOMIAL)
        assert got == brute


from pql.games import get_game
from pql.parser.parse import parse
from pql.scenario import build_scenario
from pql.runtime.sampler import complete_boards, sample_trials
from pql.runtime.evaluator import player_scores, player_score_scalar


def test_holdem_game_spec():
    g = get_game('holdem')
    assert g.name == 'holdem'
    assert g.num_hole == 2
    assert g.eval_kind == 'holdem'


def test_omaha_specs_default_to_omaha_eval_kind():
    assert get_game('omahahi5').eval_kind == 'omaha'


def test_holdem_player_scores_match_scalar():
    q = parse("select avg(riverEquity(PLAYER_1)) from game='holdem', "
              "board='2c3c4c', PLAYER_1='AsKs', PLAYER_2='QdQh'")
    s = build_scenario(q)
    player_hands, boards = sample_trials(s, num_trials=40, seed=4)
    scores = player_scores(s, player_hands, boards)
    assert scores.shape == (40, 2)
    for t in range(40):
        for p in range(2):
            assert scores[t, p] == player_score_scalar(s, p, boards[t])


def test_holdem_full_board_deterministic_winner():
    # Full 5-card board -> enumeration, trials=1. P1 (AsKs) makes aces full of kings;
    # P2 (QdQh) only two pair -> P1 wins outright.
    q = ("select avg(riverEquity(PLAYER_1)) as p1, avg(riverEquity(PLAYER_2)) as p2 "
         "from game='holdem', board='AhAd7c2dKd', PLAYER_1='AsKs', PLAYER_2='QdQh'")
    r = run_pql(q, seed=1)
    assert r.mode == 'enumeration'
    assert r.trials == 1
    assert r.values['p1'] == 1.0
    assert r.values['p2'] == 0.0


def test_holdem_equities_sum_to_one_preflop():
    q = ("select avg(riverEquity(PLAYER_1)) as p1, avg(riverEquity(PLAYER_2)) as p2 "
         "from game='holdem', PLAYER_1='AsKs', PLAYER_2='QdQh'")
    r = run_pql(q, trials=8000, seed=2)
    assert r.mode == 'monte_carlo'
    assert abs(r.values['p1'] + r.values['p2'] - 1.0) < 1e-9
    assert 0.4 < r.values['p1'] < 0.6   # AKs vs QQ is a near-coinflip (~46% for AKs)
