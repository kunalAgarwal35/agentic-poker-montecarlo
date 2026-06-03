from pql import run_pql


def test_holdem_hand_vs_range_equity():
    r = run_pql("select avg(riverEquity(PLAYER_1)) as p1, avg(riverEquity(PLAYER_2)) as p2 "
                "from game='holdem', PLAYER_1='AsKs', PLAYER_2='QQ+'", trials=8000, seed=1)
    assert r.mode == 'monte_carlo'
    assert abs(r.values['p1'] + r.values['p2'] - 1.0) < 1e-9
    assert 0.28 < r.values['p1'] < 0.42      # AKs vs QQ+ ~ 0.34 (AA/KK combos dominate)


def test_range_vs_range_runs():
    r = run_pql("select avg(riverEquity(PLAYER_1)) as p1 from game='holdem', "
                "PLAYER_1='AA,KK', PLAYER_2='JJ+,AKs'", trials=4000, seed=2)
    assert 0.0 < r.values['p1'] < 1.0


def test_empty_range_raises():
    import pytest
    with pytest.raises(ValueError):
        # all four aces dead -> 'AA' expands to no combos -> empty pool
        run_pql("select avg(riverEquity(PLAYER_1)) as p1 from game='holdem', "
                "dead='AsAhAdAc', PLAYER_1='KsKh', PLAYER_2='AA'", trials=10)


def test_plo4_hand_vs_25pct_runs():
    r = run_pql("select avg(riverEquity(PLAYER_1)) as p1, avg(riverEquity(PLAYER_2)) as p2 "
                "from game='omahahi', PLAYER_1='AsAhKsQh', PLAYER_2='25%'", trials=4000, seed=1)
    assert r.mode == 'monte_carlo'
    assert abs(r.values['p1'] + r.values['p2'] - 1.0) < 1e-9
    assert 0.5 < r.values['p1'] < 0.85
