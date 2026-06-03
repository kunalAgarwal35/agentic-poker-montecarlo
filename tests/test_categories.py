import os
import numpy as np
from hand_categories import CATEGORY_NAMES, CATEGORY_TOKENS, NAME_TO_INDEX


def test_category_constants_aligned():
    assert len(CATEGORY_NAMES) == 9
    assert len(CATEGORY_TOKENS) == 9
    assert NAME_TO_INDEX['Straight Flush'] == 8
    assert NAME_TO_INDEX['High Card'] == 0


def test_category_array_exists_and_classifies_known_hands():
    from optimized_evaluator import get_category_array
    from hand_indexing import hand_to_index
    from card_encoding import hand_str_to_ints
    ca = get_category_array()
    assert ca.shape == (2598960,)
    assert ca[hand_to_index(hand_str_to_ints('AsKsQsJsTs'))] == 8     # royal -> straight flush
    assert ca[hand_to_index(hand_str_to_ints('AsAhAdKsKh'))] == 6     # full house
    assert ca[hand_to_index(hand_str_to_ints('AsKhQdJc9s'))] == 0     # high card


def test_player_categories_omaha_and_holdem():
    from pql.parser.parse import parse
    from pql.scenario import build_scenario
    from pql.runtime.sampler import sample_trials
    from pql.runtime.evaluator import player_categories

    q = parse("select avg(riverEquity(PLAYER_1)) from game='holdem', "
              "board='2s7s9sKd', PLAYER_1='AsQs', PLAYER_2='Kc2d'")
    s = build_scenario(q)
    player_hands, boards = sample_trials(s, num_trials=10, seed=1)
    cats = player_categories(s, player_hands, boards)
    assert cats.shape == (10, 2)
    # P1 AsQs on a 3-spade board -> always at least a flush (idx 5) or better
    assert (cats[:, 0] >= 5).all()


def test_player_categories_match_bruteforce_holdem():
    import numpy as np
    from itertools import combinations
    from hand_indexing import hand_to_index
    from optimized_evaluator import get_score_array, get_category_array
    from card_encoding import hand_str_to_ints
    from pql.parser.parse import parse
    from pql.scenario import build_scenario
    from pql.runtime.sampler import sample_trials
    from pql.runtime.evaluator import player_categories

    q = parse("select avg(riverEquity(PLAYER_1)) from game='holdem', "
              "board='2c3d', PLAYER_1='AsKs', PLAYER_2='QdQh'")
    s = build_scenario(q)
    player_hands, boards = sample_trials(s, num_trials=20, seed=2)
    cats = player_categories(s, player_hands, boards)
    sa, ca = get_score_array(), get_category_array()
    hole = hand_str_to_ints('AsKs')
    for t in range(20):
        seven = np.concatenate((hole, boards[t]))
        best_score, best_cat = -1.0, 0
        for c in combinations(seven.tolist(), 5):
            i = hand_to_index(np.array(c, dtype=np.int32))
            if sa[i] > best_score:
                best_score, best_cat = sa[i], ca[i]
        assert cats[t, 0] == best_cat


from pql import run_pql
from hand_categories import CATEGORY_TOKENS


def test_exact_hand_type_and_distribution_sum_to_trials():
    cols = ", ".join(
        f"count(exactHandType(PLAYER_1, river, {tok})) as p1_{tok}" for tok in CATEGORY_TOKENS
    )
    q = (f"select {cols} from game='holdem', board='AsKsQsJsTs', "
         f"PLAYER_1='2c3d', PLAYER_2='4h5h'")
    r = run_pql(q, seed=1)
    # Board is a royal flush; P1 plays the board -> straight flush bucket = 1, rest 0.
    assert r.values['p1_straightflush'] == 1.0
    assert sum(r.values[f'p1_{t}'] for t in CATEGORY_TOKENS) == r.trials


def test_exact_hand_type_monte_carlo_distribution():
    cols = ", ".join(
        f"count(exactHandType(PLAYER_1, river, {tok})) as p1_{tok}" for tok in CATEGORY_TOKENS
    )
    q = (f"select {cols} from game='holdem', PLAYER_1='AsKs', PLAYER_2='QdQh'")
    r = run_pql(q, trials=4000, seed=3)
    total = sum(r.values[f'p1_{t}'] for t in CATEGORY_TOKENS)
    assert total == r.trials            # every trial lands in exactly one bucket
    assert r.values['p1_pair'] > 0      # AK makes a pair often
