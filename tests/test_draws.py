from pql import run_pql


def test_flush_draw_true_on_four_flush():
    q = ("select count(flushDraw(PLAYER_1)) as x "
         "from game='holdem', board='Ah7c2h', PLAYER_1='KhQh', PLAYER_2='AsAd'")
    r = run_pql(q, seed=1)
    assert r.values["x"] == r.trials          # always a flush draw on this flop


def test_gutshot_excludes_oesd():
    # JhTh on 8c7h2d: needs a 9 (4 outs) -> an inside (gut-shot) straight draw, not open-ended.
    base = "from game='holdem', board='8c7h2d', PLAYER_1='JhTh', PLAYER_2='AsAd'"
    gs = run_pql(f"select count(gutshot(PLAYER_1)) as x {base}", seed=1)
    oesd = run_pql(f"select count(oesd(PLAYER_1)) as x {base}", seed=1)
    assert gs.values["x"] == gs.trials
    assert oesd.values["x"] == 0.0


def test_oesd_true_for_open_ended():
    # 9h8h on 7c6h2d: 5s and Ts complete the straight -> 8 outs -> open-ended.
    q = ("select count(oesd(PLAYER_1)) as x "
         "from game='holdem', board='7c6h2d', PLAYER_1='9h8h', PLAYER_2='AsAd'")
    r = run_pql(q, seed=1)
    assert r.values["x"] == r.trials


def test_straight_draw_not_inflated_by_flush_cards():
    # 9h8h on 7h6h2d also has a flush draw; a heart that makes a flush must NOT be
    # counted as a straight out. oesd stays true (8 clean straight outs: the four 5s
    # and four Ts, where 5h/Th are straight flushes and still count as straights-or-better).
    q = ("select count(oesd(PLAYER_1)) as x "
         "from game='holdem', board='7h6h2d', PLAYER_1='9h8h', PLAYER_2='AsAd'")
    r = run_pql(q, seed=1)
    assert r.values["x"] == r.trials


def test_nut_hi_for_flush_true_with_nut_flush():
    # AhKh makes the A-high (nut) flush -> holds the nut of their own category. 2-arg PPT form.
    q = ("select count(nutHiForHandType(PLAYER_1, river)) as x "
         "from game='holdem', board='Qh7h2h9c4d', PLAYER_1='AhKh', PLAYER_2='JsTd'")
    r = run_pql(q, seed=1)
    assert r.values["x"] == r.trials


def test_nut_hi_for_flush_false_with_second_nut():
    # KhJh makes only the second-nut flush -> not the nut of their category.
    q = ("select count(nutHiForHandType(PLAYER_1, river)) as x "
         "from game='holdem', board='Qh7h2h9c4d', PLAYER_1='KhJh', PLAYER_2='AsTd'")
    r = run_pql(q, seed=1)
    assert r.values["x"] == 0.0
