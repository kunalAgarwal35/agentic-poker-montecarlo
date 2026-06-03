from pql import run_pql


def test_single_player_category_query_runs():
    # No opponent needed: how often does one hand make >= a pair by the river.
    r = run_pql("select count(minHandType(PLAYER_1, river, pair)) as c "
                "from game='holdem', PLAYER_1='AsKs'", trials=3000, seed=1)
    assert "c" in r.values
    assert 0.0 <= r.values["c"] <= r.trials


def test_single_player_flush_draw_runs():
    # KhQh always has a flush draw on the Ah7c2h flop -> one player is enough.
    r = run_pql("select count(flushDraw(PLAYER_1)) as fd "
                "from game='holdem', board='Ah7c2h', PLAYER_1='KhQh'", seed=1)
    assert r.values["fd"] == r.trials
