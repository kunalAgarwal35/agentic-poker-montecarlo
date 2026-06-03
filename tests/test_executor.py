import numpy as np
from pql.functions.aggregates import get_aggregate
from pql.result import PQLResult


def test_avg_and_count():
    col = np.array([1.0, 0.0, 0.5, 1.0])
    assert get_aggregate("avg")(col) == 0.625
    assert get_aggregate("count")(col) == 3   # count of non-zero rows (1.0, 0.5, 1.0)


def test_pqlresult_holds_columns():
    r = PQLResult(values={"p1_eq": 0.62}, trials=1000, mode="monte_carlo", seed=1)
    assert r.values["p1_eq"] == 0.62
    assert r.mode == "monte_carlo"


from pql import run_pql


def test_run_pql_equities_sum_to_one_monte_carlo():
    # Preflop-ish: empty board -> Monte Carlo (large completion space)
    q = (
        "select avg(riverEquity(PLAYER_1)) as p1, avg(riverEquity(PLAYER_2)) as p2 "
        "from game='omahahi5', board='', "
        "PLAYER_1='AsAhKsKhQs', PLAYER_2='JdTd9d8d7d'"
    )
    r = run_pql(q, trials=5000, seed=123)
    assert r.mode == "monte_carlo"
    assert abs(r.values["p1"] + r.values["p2"] - 1.0) < 1e-9   # heads-up equities sum to 1
    assert 0.0 < r.values["p1"] < 1.0


def test_run_pql_enumerates_one_card_board():
    # 4 board cards given -> need 1 -> enumerate all completions -> exact, mode=enumeration
    q = (
        "select avg(riverEquity(PLAYER_1)) as p1, avg(riverEquity(PLAYER_2)) as p2 "
        "from game='omahahi5', board='2c3c4c5c', "
        "PLAYER_1='AsAhKsKhQs', PLAYER_2='JdTd9d8d7d'"
    )
    r = run_pql(q, trials=10000, seed=1)
    assert r.mode == "enumeration"
    assert abs(r.values["p1"] + r.values["p2"] - 1.0) < 1e-9


def test_run_pql_count_winsHi_le_trials():
    q = (
        "select count(winsHi(PLAYER_1)) as w1 "
        "from game='omahahi5', board='', "
        "PLAYER_1='AsAhKsKhQs', PLAYER_2='JdTd9d8d7d'"
    )
    r = run_pql(q, trials=3000, seed=5)
    assert 0 <= r.values["w1"] <= 3000


def test_enumeration_exact_value_known_rational():
    # 4-card board -> 38 river completions enumerated exactly. PLAYER_1 wins exactly
    # 34 of 38 (no ties), so equity is 34/38 (= 17/19), hand-verifiable & Java-independent.
    q = (
        "select avg(riverEquity(PLAYER_1)) as p1, avg(riverEquity(PLAYER_2)) as p2 "
        "from game='omahahi5', board='2c3c4c5c', "
        "PLAYER_1='AsAhKsKhQs', PLAYER_2='JdTd9d8d7d'"
    )
    r = run_pql(q, seed=1)
    assert r.mode == "enumeration"
    assert r.trials == 38
    assert r.values["p1"] == 34 / 38
    assert r.values["p2"] == 4 / 38


def test_dead_cards_change_equity():
    base = (
        "select avg(riverEquity(PLAYER_1)) as p1 from game='omahahi5', board='2c3c4c', "
        "PLAYER_1='AsAhKsKhQs', PLAYER_2='JdTd9d8d7d'"
    )
    with_dead = (
        "select avg(riverEquity(PLAYER_1)) as p1 from game='omahahi5', board='2c3c4c', "
        "dead='JsJhTsTh', PLAYER_1='AsAhKsKhQs', PLAYER_2='JdTd9d8d7d'"
    )
    r1 = run_pql(base, seed=7)
    r2 = run_pql(with_dead, seed=7)
    # both enumerate exactly over different live decks -> exact equities must differ
    assert r1.values["p1"] != r2.values["p1"]


def test_duplicate_card_rejected_via_run_pql():
    import pytest
    q = (
        "select avg(riverEquity(PLAYER_1)) as p1 from game='omahahi5', "
        "PLAYER_1='AsAhKsKhQs', PLAYER_2='AsTd9d8d7d'"   # shared As
    )
    with pytest.raises(ValueError):
        run_pql(q, seed=1)
