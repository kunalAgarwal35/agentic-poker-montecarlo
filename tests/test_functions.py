import numpy as np
from pql.parser.parse import parse
from pql.scenario import build_scenario
from pql.runtime.sampler import complete_boards, sample_trials
from pql.runtime.evaluator import player_scores, player_score_scalar


def _scn():
    q = parse(
        "select avg(riverEquity(PLAYER_1)) from game='omahahi5', "
        "board='2c3c4c', PLAYER_1='AsAhKsKhQs', PLAYER_2='JdTd9d8d7d'"
    )
    return build_scenario(q)


def test_player_scores_shape():
    s = _scn()
    player_hands, boards = sample_trials(s, num_trials=200, seed=7)
    scores = player_scores(s, player_hands, boards)
    assert scores.shape == (200, 2)             # (trials, players)
    assert (scores > 0).all()                   # every best-5 score is positive


def test_vector_matches_scalar_per_trial():
    s = _scn()
    player_hands, boards = sample_trials(s, num_trials=50, seed=7)
    scores = player_scores(s, player_hands, boards)
    for t in range(boards.shape[0]):
        for p in range(len(s.players)):
            assert scores[t, p] == player_score_scalar(s, p, boards[t])


from pql.functions.registry import get_value_fn
from pql.functions import values  # registers fns on import
from pql.runtime.context import EvalContext


def _ctx(scores):
    # category-free context; category fns aren't exercised here
    return EvalContext(scenario=None, player_hands=None, boards=None, scores=np.asarray(scores, dtype=float))


def test_river_equity_tie_split():
    ctx = _ctx([[10.0, 5.0], [5.0, 10.0], [7.0, 7.0]])
    eq = get_value_fn("riverequity")(ctx, 0, [])
    assert eq.tolist() == [1.0, 0.0, 0.5]


def test_wins_hi_sole_only():
    ctx = _ctx([[10.0, 5.0], [7.0, 7.0]])
    assert get_value_fn("winshi")(ctx, 0, []).tolist() == [1.0, 0.0]


def test_ties_hi_only_on_ties():
    ctx = _ctx([[10.0, 5.0], [7.0, 7.0]])
    assert get_value_fn("tieshi")(ctx, 0, []).tolist() == [0.0, 1.0]


def test_river_equity_against_evaluator_matches_scalar():
    s = _scn()
    player_hands, boards = sample_trials(s, num_trials=40, seed=3)
    scores = player_scores(s, player_hands, boards)
    ctx = EvalContext(scenario=s, player_hands=player_hands, boards=boards, scores=scores)
    eq_vec = get_value_fn("riverequity")(ctx, 0, [])
    for t in range(boards.shape[0]):
        row = scores[t]
        mx = row.max()
        winners = int((row == mx).sum())
        share = (1.0 / winners) if row[0] == mx else 0.0
        assert eq_vec[t] == share
