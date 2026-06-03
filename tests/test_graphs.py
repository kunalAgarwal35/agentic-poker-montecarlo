import numpy as np
from pql.scenario import build_scenario
from pql.parser.ast import Query
from pql.graphs import equity_by_street
from pql import run_pql


def _scenario(game, board, players, dead=""):
    params = {"game": game}
    if board:
        params["board"] = board
    if dead:
        params["dead"] = dead
    for name, val in players.items():
        params[name.lower()] = val
    return build_scenario(Query(select=[], params=params))


def test_street_prefixes_match_board_length():
    # A flop board -> exactly preflop + flop points.
    sc = _scenario("holdem", "Ah7c2d", {"PLAYER_1": "AsKs", "PLAYER_2": "QdQh"})
    series = equity_by_street(sc, trials=4000, seed=1)
    assert {p["street"] for p in series[0]["points"]} == {"preflop", "flop"}
    assert [s["name"] for s in series] == ["PLAYER_1", "PLAYER_2"]


def test_street_river_point_matches_river_equity():
    # The 'river' prefix on a full board equals avg(riverEquity) for that board.
    board = "Ah7c2dKsQc"
    sc = _scenario("holdem", board, {"PLAYER_1": "AsKh", "PLAYER_2": "QdJd"})
    series = equity_by_street(sc, trials=4000, seed=1)
    river_pt = next(p["equity"] for p in series[0]["points"] if p["street"] == "river")
    direct = run_pql(
        f"select avg(riverEquity(PLAYER_1)) as p1 from game='holdem', board='{board}', "
        "PLAYER_1='AsKh', PLAYER_2='QdJd'", seed=1).values["p1"]
    assert abs(river_pt - direct) < 1e-9   # both enumerate the (here empty) runout -> exact


def test_street_equities_sum_to_one_per_street():
    sc = _scenario("holdem", "Ah7c2d", {"PLAYER_1": "AsKs", "PLAYER_2": "QdQh"})
    series = equity_by_street(sc, trials=6000, seed=2)
    for i, _ in enumerate(series[0]["points"]):
        total = sum(s["points"][i]["equity"] for s in series)
        assert abs(total - 1.0) < 1e-6


def test_street_plo_runs():
    # Exercises the omaha eval path through _sample_or_enumerate / player_scores.
    sc = _scenario("omahahi", "Ah7c2d", {"PLAYER_1": "AsKsQsJs", "PLAYER_2": "AdKdQdJd"})
    series = equity_by_street(sc, trials=3000, seed=1)
    assert {p["street"] for p in series[0]["points"]} == {"preflop", "flop"}
    for i, _ in enumerate(series[0]["points"]):
        assert abs(sum(s["points"][i]["equity"] for s in series) - 1.0) < 1e-6


from pql.graphs import equity_distribution


def test_distribution_mean_matches_overall_equity():
    # Hero AsKs vs a top-ish holdem range, on a flop. The distribution mean must equal
    # hero's overall equity vs that range (a built-in cross-check).
    sc = _scenario("holdem", "Ah7c2d", {"PLAYER_1": "AsKs", "PLAYER_2": "QQ+"})
    dist = equity_distribution(sc, hero_idx=0, trials=6000, seed=3)
    direct = run_pql(
        "select avg(riverEquity(PLAYER_1)) as p1 from game='holdem', board='Ah7c2d', "
        "PLAYER_1='AsKs', PLAYER_2='QQ+'", trials=40000, seed=3).values["p1"]
    assert abs(dist["mean"] - direct) < 0.03
    assert len(dist["buckets"]) == 10
    assert abs(sum(b["pct"] for b in dist["buckets"]) - 1.0) < 1e-9


def test_distribution_requires_a_range_villain():
    import pytest
    sc = _scenario("holdem", "Ah7c2d", {"PLAYER_1": "AsKs", "PLAYER_2": "QdQh"})
    with pytest.raises(ValueError):
        equity_distribution(sc, hero_idx=0, trials=2000, seed=1)


def test_distribution_plo_mean_matches_overall():
    # PLO4: AAxx is a large range -> exercises the omaha kernel and the combo-cap path.
    sc = _scenario("omahahi", "Ah7c2d", {"PLAYER_1": "AsKsQsJs", "PLAYER_2": "AAxx"})
    dist = equity_distribution(sc, hero_idx=0, trials=4000, seed=5)
    direct = run_pql(
        "select avg(riverEquity(PLAYER_1)) as p1 from game='omahahi', board='Ah7c2d', "
        "PLAYER_1='AsKsQsJs', PLAYER_2='AAxx'", trials=30000, seed=5).values["p1"]
    assert abs(dist["mean"] - direct) < 0.04


from pql.graphs import equity_vs_class


def test_vsclass_freq_weighted_equity_matches_overall():
    sc = _scenario("holdem", "Ah7c2d", {"PLAYER_1": "AsKs", "PLAYER_2": "QQ+"})
    vc = equity_vs_class(sc, hero_idx=0, trials=8000, seed=4)
    assert abs(sum(r["freq"] for r in vc["rows"]) - 1.0) < 1e-9
    weighted = sum(r["freq"] * r["equity"] for r in vc["rows"])
    direct = run_pql(
        "select avg(riverEquity(PLAYER_1)) as p1 from game='holdem', board='Ah7c2d', "
        "PLAYER_1='AsKs', PLAYER_2='QQ+'", trials=40000, seed=4).values["p1"]
    assert abs(weighted - direct) < 0.03


def test_vsclass_requires_two_players():
    import pytest
    sc = _scenario("holdem", "Ah7c2d",
                   {"PLAYER_1": "AsKs", "PLAYER_2": "QdQh", "PLAYER_3": "JsJh"})
    with pytest.raises(ValueError):
        equity_vs_class(sc, hero_idx=0, trials=2000, seed=1)
