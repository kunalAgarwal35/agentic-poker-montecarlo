import numpy as np
from pql.games import get_game, GameSpec


def test_omahahi5_spec():
    g = get_game("omahahi5")
    assert isinstance(g, GameSpec)
    assert g.name == "omahahi5"
    assert g.num_hole == 5
    assert g.hand_combos.shape == (10, 2)   # C(5,2)
    assert g.board_combos.shape == (10, 3)  # C(5,3)


def test_unknown_game_raises():
    import pytest
    with pytest.raises(ValueError):
        get_game("badugi")


from pql.scenario import build_scenario, Scenario
from pql.parser.parse import parse


def test_build_scenario_fixed_hands():
    q = parse(
        "select avg(riverEquity(PLAYER_1)) "
        "from game='omahahi5', board='2c3c4c', dead='9h', "
        "PLAYER_1='AsAhKsKhQs', PLAYER_2='JdTd9d8d7d'"
    )
    s = build_scenario(q)
    assert isinstance(s, Scenario)
    assert s.game.name == "omahahi5"
    assert len(s.players) == 2
    assert s.players[0].name == "PLAYER_1"
    # Encoding rank*4+suit, RANKS='23456789TJQKA', SUITS='shdc':
    # As=48, Ah=49, Ks=44, Kh=45, Qs=40
    assert s.players[0].cards.tolist() == [48, 49, 44, 45, 40]
    # board '2c3c4c': 2c=3, 3c=7, 4c=11
    assert s.board.tolist() == [3, 7, 11]
    # dead '9h': 9=idx7, h=idx1 -> 7*4+1=29
    assert s.dead.tolist() == [29]


def test_range_value_now_supported():
    # Ranges are now supported; percentile range expands to a pool of combos.
    # Use PLO4 (omahahi), whose class-order artifact is committed; PLO5's artifact
    # is generated separately and may be absent.
    q = parse(
        "select avg(riverEquity(PLAYER_1)) "
        "from game='omahahi', PLAYER_1='25%', PLAYER_2='JdTd9d8d'"
    )
    s = build_scenario(q)
    assert s.players[0].pool is not None
    assert s.players[0].pool.shape[0] > 0
    assert s.players[0].cards is None


def test_duplicate_card_across_players_raises():
    import pytest
    q = parse(
        "select avg(riverEquity(PLAYER_1)) from game='omahahi5', "
        "PLAYER_1='AsAhKsKhQs', PLAYER_2='AsTd9d8d7d'"   # shared As
    )
    with pytest.raises(ValueError):
        build_scenario(q)


def test_board_too_long_raises():
    import pytest
    q = parse(
        "select avg(riverEquity(PLAYER_1)) from game='omahahi5', "
        "board='2c3c4c5c6c7c', PLAYER_1='AsAhKsKhQs', PLAYER_2='JdTd9d8d7d'"
    )
    with pytest.raises(ValueError):
        build_scenario(q)
