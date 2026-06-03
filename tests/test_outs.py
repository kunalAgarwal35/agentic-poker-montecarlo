import numpy as np
from pql import run_pql


def test_flush_draw_outs_holdem_deterministic():
    # KhQh on the Ah7c2h flop: 4 hearts in hand+board; 9 hearts remain.
    # outsToHandType(P1, flop, flush) counts single-card outs to a flush by the turn.
    # Board is exactly the flop, so there is no sampling -> deterministic 9.
    q = ("select avg(outsToHandType(PLAYER_1, flop, flush)) as o "
         "from game='holdem', board='Ah7c2h', PLAYER_1='KhQh', PLAYER_2='AsAd'")
    r = run_pql(q, seed=1)
    assert r.values["o"] == 9.0


def test_min_outs_predicate_holdem():
    base = "from game='holdem', board='Ah7c2h', PLAYER_1='KhQh', PLAYER_2='AsAd'"
    nine = run_pql(f"select count(minOutsToHandType(PLAYER_1, flop, flush, 9)) as x {base}", seed=1)
    ten = run_pql(f"select count(minOutsToHandType(PLAYER_1, flop, flush, 10)) as x {base}", seed=1)
    # Enumeration over the 2 remaining streets -> every outcome has exactly 9 flush outs
    # at the flop, so >=9 is always true and >=10 never true.
    assert nine.values["x"] == nine.trials
    assert ten.values["x"] == 0.0


def test_already_made_has_zero_outs():
    # On a 4-heart flush board the player already has a flush -> 0 outs to a flush.
    q = ("select avg(outsToHandType(PLAYER_1, flop, flush)) as o "
         "from game='holdem', board='Kh7h2h', PLAYER_1='AhQh', PLAYER_2='AsAd'")
    r = run_pql(q, seed=1)
    assert r.values["o"] == 0.0


def test_river_street_rejected_for_outs():
    import pytest
    with pytest.raises(ValueError):
        run_pql("select avg(outsToHandType(PLAYER_1, river, flush)) as o "
                "from game='holdem', board='Ah7c2h', PLAYER_1='KhQh', PLAYER_2='AsAd'", seed=1)


def test_flush_draw_outs_omaha_plo4_deterministic():
    # PLO4: AhKh + 2 offsuit, on the Qh7h2c flop. Using exactly two hole cards (Ah,Kh)
    # plus three board cards, the player has 4 hearts (Ah Kh Qh 7h) -> a one-card flush
    # draw. Remaining hearts = 13 - 4 = 9 outs to a flush by the turn.
    q = ("select avg(outsToHandType(PLAYER_1, flop, flush)) as o "
         "from game='omahahi', board='Qh7h2c', PLAYER_1='AhKh3s4d', PLAYER_2='AsAd5c6c'")
    r = run_pql(q, seed=1)
    assert r.values["o"] == 9.0


def test_outs_omaha_plo5_runs_and_is_bounded():
    q = ("select avg(outsToHandType(PLAYER_1, turn, straight)) as o "
         "from game='omahahi5', board='9h8c2d', PLAYER_1='7h6sJdTc5h', PLAYER_2='AsAdKsKd2c'")
    r = run_pql(q, trials=5000, seed=3)
    assert 0.0 <= r.values["o"] <= 47.0
