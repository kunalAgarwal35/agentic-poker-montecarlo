from pql.parser.parse import parse
from pql.parser.ast import (
    Query, SelectItem, Call, Ident, IntLit, BinOp, Compare, Logical, Not,
)


def _agg_arg(q: Query):
    """The single expression argument of the first select item's aggregate."""
    return q.select[0].expr.args[0]


def test_plain_value_call_unchanged():
    q = parse("select avg(riverEquity(PLAYER_1)) as e from game='holdem'")
    assert q.select[0].expr.name == "avg"
    arg = _agg_arg(q)
    assert isinstance(arg, Call) and arg.name == "riverEquity"
    assert isinstance(arg.args[0], Ident) and arg.args[0].name == "PLAYER_1"
    assert q.where is None


def test_comparison_builds_compare_node():
    q = parse("select count(outsToHandType(PLAYER_1,flop,flush) > 5) as x from game='holdem'")
    arg = _agg_arg(q)
    assert isinstance(arg, Compare) and arg.op == ">"
    assert isinstance(arg.left, Call) and arg.left.name == "outsToHandType"
    # value-fn token args stay Idents (so the executor's token plumbing still works)
    assert [a.name for a in arg.left.args] == ["PLAYER_1", "flop", "flush"]
    assert isinstance(arg.right, IntLit) and arg.right.value == 5


def test_arithmetic_builds_binop():
    q = parse("select avg(outsToHandType(PLAYER_1,flop,flush) + 1) as x from game='holdem'")
    arg = _agg_arg(q)
    assert isinstance(arg, BinOp) and arg.op == "+"
    assert isinstance(arg.right, IntLit) and arg.right.value == 1


def test_precedence_add_below_compare():
    # a + b > c  parses as  (a + b) > c
    q = parse("select count(minOutsToHandType(PLAYER_1,flop,flush) + 1 > 2) as x from game='holdem'")
    arg = _agg_arg(q)
    assert isinstance(arg, Compare) and arg.op == ">"
    assert isinstance(arg.left, BinOp) and arg.left.op == "+"


def test_where_with_and_or_not():
    q = parse(
        "select count(winsHi(PLAYER_1)) as w from game='holdem' "
        "where minHandType(PLAYER_1,river,flush) and not winsHi(PLAYER_2)"
    )
    assert isinstance(q.where, Logical) and q.where.op == "and"
    assert isinstance(q.where.left, Call) and q.where.left.name == "minHandType"
    assert isinstance(q.where.right, Not)
    assert isinstance(q.where.right.operand, Call) and q.where.right.operand.name == "winsHi"


def test_equality_and_neq_ops():
    q = parse("select count(outsToHandType(PLAYER_1,flop,flush) = 9) as x from game='holdem'")
    assert _agg_arg(q).op == "="
    q2 = parse("select count(outsToHandType(PLAYER_1,flop,flush) != 9) as x from game='holdem'")
    assert _agg_arg(q2).op == "!="


def test_value_fn_int_arg_stays_ident():
    # minOutsToHandType(P,flop,flush,9): the 9 is a value-fn token arg, NOT an IntLit
    q = parse("select avg(minOutsToHandType(PLAYER_1,flop,flush,9)) as x from game='holdem'")
    call = _agg_arg(q)
    assert isinstance(call, Call) and call.name == "minOutsToHandType"
    assert all(isinstance(a, Ident) for a in call.args)
    assert call.args[-1].name == "9"


import numpy as np
from pql import run_pql


SCEN = "game='holdem', board='Ah7c2h', PLAYER_1='KhQh', PLAYER_2='QdQc'"
SEED = 7


def test_arithmetic_avg_shifts_by_constant():
    base = run_pql(f"select avg(outsToHandType(PLAYER_1,flop,flush)) as o from {SCEN}",
                   trials=4000, seed=SEED).values["o"]
    plus = run_pql(f"select avg(outsToHandType(PLAYER_1,flop,flush) + 1) as o from {SCEN}",
                   trials=4000, seed=SEED).values["o"]
    assert abs(plus - (base + 1)) < 1e-9


def test_comparison_count_matches_manual_mask():
    # P1 has exactly 9 flush outs on this flop, so (>5) is true on every trial.
    r = run_pql(f"select count(outsToHandType(PLAYER_1,flop,flush) > 5) as x from {SCEN}",
                trials=3000, seed=SEED)
    assert r.values["x"] == r.trials  # 100%


def test_sum_aggregate():
    r = run_pql(f"select sum(winsHi(PLAYER_1)) as s, count(winsHi(PLAYER_1)) as c from {SCEN}",
                trials=3000, seed=SEED)
    # winsHi is 0/1, so sum == count.
    assert r.values["s"] == r.values["c"]


def test_where_filters_and_conditions_aggregate():
    # avg(riverEquity) given P1 makes a flush should exceed the unconditional avg.
    cond = run_pql(
        f"select avg(riverEquity(PLAYER_1)) as e from {SCEN} where minHandType(PLAYER_1,river,flush)",
        trials=4000, seed=SEED)
    uncond = run_pql(f"select avg(riverEquity(PLAYER_1)) as e from {SCEN}",
                     trials=4000, seed=SEED).values["e"]
    assert cond.values["e"] > uncond
    assert cond.trials < 4000  # trials is the matching-trial denominator


def test_where_denominator_is_filtered():
    # count(winsHi) where winsHi  ==  trials (every matching trial wins -> 100%)
    r = run_pql(
        f"select count(winsHi(PLAYER_1)) as w from {SCEN} where winsHi(PLAYER_1)",
        trials=3000, seed=SEED)
    assert r.values["w"] == r.trials and r.trials > 0


def test_empty_where_yields_null_avg_zero_trials():
    # A condition that is never true -> 0 matching trials.
    r = run_pql(
        f"select avg(riverEquity(PLAYER_1)) as e, count(winsHi(PLAYER_1)) as w "
        f"from {SCEN} where minHandType(PLAYER_1,flop,straightflush)",
        trials=2000, seed=SEED)
    assert r.trials == 0
    assert r.values["e"] is None
    assert r.values["w"] == 0.0


def test_handtype_returns_category_index():
    # On Ah7c2h with KhQh, by the river P1's category is in 0..8; histogram-ready.
    r = run_pql(f"select avg(handType(PLAYER_1,river)) as h, "
                f"max(handType(PLAYER_1,river)) as mx, min(handType(PLAYER_1,river)) as mn from {SCEN}",
                trials=2000, seed=SEED)
    assert 0.0 <= r.values["mn"] <= r.values["h"] <= r.values["mx"] <= 8.0


def test_handtype_defaults_to_river():
    a = run_pql(f"select avg(handType(PLAYER_1)) as h from {SCEN}", trials=1500, seed=SEED).values["h"]
    b = run_pql(f"select avg(handType(PLAYER_1,river)) as h from {SCEN}", trials=1500, seed=SEED).values["h"]
    assert abs(a - b) < 1e-9


def test_histogram_pairs_sum_to_trials_with_labels():
    r = run_pql(f"select histogram(handType(PLAYER_1,river)) as h from {SCEN}",
                trials=3000, seed=SEED)
    assert "h" not in r.values            # histogram is NOT a scalar column
    hist = r.histograms["h"]
    assert sum(p["count"] for p in hist["pairs"]) == r.trials
    # category labels attached (handType is a category-returning value fn)
    assert hist["labels"] is not None
    # every value key is a 0..8 category index
    assert all(0 <= p["value"] <= 8 for p in hist["pairs"])
    # labels map the integer values to category names
    some = hist["pairs"][0]["value"]
    assert str(some) in hist["labels"]


def test_histogram_numeric_has_no_labels():
    r = run_pql(f"select histogram(outsToHandType(PLAYER_1,flop,flush)) as h from {SCEN}",
                trials=2000, seed=SEED)
    hist = r.histograms["h"]
    assert hist["labels"] is None
    assert sum(p["count"] for p in hist["pairs"]) == r.trials


def test_histogram_and_scalar_columns_coexist():
    r = run_pql(f"select avg(riverEquity(PLAYER_1)) as e, histogram(handType(PLAYER_1,river)) as h from {SCEN}",
                trials=1500, seed=SEED)
    assert "e" in r.values and "h" in r.histograms
    assert r.columns == ["e", "h"]
