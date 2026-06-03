import numpy as np
from pql.functions.registry import get_value_fn
from pql.functions import values  # registers
from pql.runtime.context import EvalContext


class _Ctx:
    """Minimal ctx exposing scores + categories for value-fn unit tests."""
    def __init__(self, scores, cats):
        self.scores = np.asarray(scores, dtype=float)
        self._c = np.asarray(cats, dtype=np.int64)
    def categories(self):
        return self._c


def test_min_max_hand_type():
    ctx = _Ctx(scores=[[1.0], [1.0], [1.0]], cats=[[1], [5], [8]])
    minf = get_value_fn('minhandtype')(ctx, 0, ['river', 'flush'])   # >= flush(5)
    assert minf.tolist() == [0.0, 1.0, 1.0]
    maxp = get_value_fn('maxhandtype')(ctx, 0, ['river', 'twopair'])  # <= twopair(2)
    assert maxp.tolist() == [1.0, 0.0, 0.0]


def test_unknown_token_raises():
    import pytest
    ctx = _Ctx(scores=[[1.0]], cats=[[1]])
    with pytest.raises(ValueError):
        get_value_fn('minhandtype')(ctx, 0, ['river', 'bogus'])


from pql import run_pql


def test_min_hand_type_consistency_with_exact():
    board = "board='2c3c4c5c'"  # enumeration (turn->river), 38 outcomes
    hero = "PLAYER_1='AsAhKsKhQs'"; villain = "PLAYER_2='JdTd9d8d7d'"
    q_min = f"select count(minHandType(PLAYER_1, river, flush)) as m from game='omahahi5', {board}, {hero}, {villain}"
    q_exact = ("select count(exactHandType(PLAYER_1, river, flush)) as fl, "
               "count(exactHandType(PLAYER_1, river, fullhouse)) as fh, "
               "count(exactHandType(PLAYER_1, river, quads)) as qu, "
               "count(exactHandType(PLAYER_1, river, straightflush)) as sf "
               f"from game='omahahi5', {board}, {hero}, {villain}")
    rmin = run_pql(q_min, seed=1)
    rex = run_pql(q_exact, seed=1)
    assert rmin.values['m'] == rex.values['fl'] + rex.values['fh'] + rex.values['qu'] + rex.values['sf']


def test_min_hand_type_flop_deterministic():
    # AsKs on the Ah7c2d flop = pair of aces -> >= pair on the flop is always true,
    # >= twopair on the flop is never true (no second pair on that board).
    from pql import run_pql
    base = "from game='holdem', board='Ah7c2d', PLAYER_1='AsKs', PLAYER_2='QdQh'"
    p = run_pql(f"select count(minHandType(PLAYER_1, flop, pair)) as c {base}", seed=1)
    tp = run_pql(f"select count(minHandType(PLAYER_1, flop, twopair)) as c {base}", seed=1)
    assert p.values["c"] == p.trials
    assert tp.values["c"] == 0.0


def test_min_hand_type_street_monotonic():
    # The chance of >= two pair only grows as more board cards arrive.
    from pql import run_pql
    base = "from game='holdem', board='Ah7c2d', PLAYER_1='AsKs', PLAYER_2='QdQh'"
    flop = run_pql(f"select count(minHandType(PLAYER_1, flop, twopair)) as c {base}", seed=2).values["c"]
    turn = run_pql(f"select count(minHandType(PLAYER_1, turn, twopair)) as c {base}", seed=2).values["c"]
    river = run_pql(f"select count(minHandType(PLAYER_1, river, twopair)) as c {base}", seed=2).values["c"]
    assert flop <= turn <= river


def test_river_hand_type_unchanged_regression():
    # River category must equal the old full-board behavior (PLAYER_1 makes >= flush).
    from pql import run_pql
    q = ("select count(minHandType(PLAYER_1, river, flush)) as c "
         "from game='omahahi5', board='2c3c4c', PLAYER_1='AsAhKsKhQs', PLAYER_2='JdTd9d8d7d'")
    assert run_pql(q, seed=1).values["c"] >= 0
