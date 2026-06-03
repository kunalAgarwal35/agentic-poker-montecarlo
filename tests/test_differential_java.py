import os
import re
import shutil
import subprocess

import pytest
from pql import run_pql
from testing.java_oracle import java_available, run_java_pql

pytestmark = pytest.mark.skipif(not java_available(), reason="java or p2.jar not available")


def test_equity_matches_java_enumerated():
    # 4 board cards -> our engine enumerates (EXACT). The jar still samples (-mt),
    # so we compare within a small tolerance; our exact value is pinned separately
    # in tests/test_executor.py::test_enumeration_exact_value_known_rational.
    q = (
        "select avg(riverEquity(PLAYER_1)) as p1, avg(riverEquity(PLAYER_2)) as p2 "
        "from game='omahahi5', syntax='Generic', board='2c3c4c5c', dead='', "
        "PLAYER_1='AsAhKsKhQs', PLAYER_2='JdTd9d8d7d'"
    )
    ours = run_pql(q, seed=1)
    theirs = run_java_pql(q)
    assert ours.mode == "enumeration"
    assert abs(ours.values["p1"] - theirs["p1"]) < 0.005
    assert abs(ours.values["p2"] - theirs["p2"]) < 0.005


def test_equity_matches_java_monte_carlo_within_tolerance():
    q = (
        "select avg(riverEquity(PLAYER_1)) as p1, avg(riverEquity(PLAYER_2)) as p2 "
        "from game='omahahi5', syntax='Generic', board='', dead='', "
        "PLAYER_1='AsAhKsKhQs', PLAYER_2='JdTd9d8d7d'"
    )
    ours = run_pql(q, trials=50000, seed=99)
    theirs = run_java_pql(q)
    # Monte Carlo on both sides -> statistical tolerance (~0.6%).
    assert abs(ours.values["p1"] - theirs["p1"]) < 0.006


def test_holdem_equity_matches_java():
    # AsKs vs QdQh on Ah7c2d -> our engine enumerates the 2 remaining cards (exact);
    # Java samples -> compare within tolerance.
    q = ("select avg(riverEquity(PLAYER_1)) as p1, avg(riverEquity(PLAYER_2)) as p2 "
         "from game='holdem', syntax='Generic', board='Ah7c2d', dead='', "
         "PLAYER_1='AsKs', PLAYER_2='QdQh'")
    ours = run_pql(q, seed=1)
    theirs = run_java_pql(q)
    assert ours.mode == 'enumeration'
    assert abs(ours.values['p1'] - theirs['p1']) < 0.02


def test_holdem_hand_vs_range_matches_java():
    q = ("select avg(riverEquity(PLAYER_1)) as p1, avg(riverEquity(PLAYER_2)) as p2 "
         "from game='holdem', syntax='Generic', PLAYER_1='AsKs', PLAYER_2='QQ+'")
    ours = run_pql(q, trials=80000, seed=1)
    theirs = run_java_pql(q)
    assert abs(ours.values['p1'] - theirs['p1']) < 0.01


def test_plo5_hand_vs_percentile_matches_java():
    import os
    plo5 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plo5_class_order.json')
    if not os.path.exists(plo5):
        pytest.skip('plo5_class_order.json not generated yet')
    q = ("select avg(riverEquity(PLAYER_1)) as p1, avg(riverEquity(PLAYER_2)) as p2 "
         "from game='omahahi5', syntax='Generic', PLAYER_1='AsAhKsKhQs', PLAYER_2='25%'")
    ours = run_pql(q, trials=60000, seed=2)
    theirs = run_java_pql(q)
    assert abs(ours.values['p1'] - theirs['p1']) < 0.012


def test_omaha_pattern_vs_pattern_matches_java():
    # KQJTs = single-suited KQJT (valid PPT Omaha suitedness suffix).
    # KQJTds is not valid PPT syntax for 4-card Omaha (rejected as "5 cards").
    q = ("select avg(riverEquity(PLAYER_1)) as p1 "
         "from game='omahahi', syntax='Generic', PLAYER_1='AAxx', PLAYER_2='KQJTs'")
    ours = run_pql(q, trials=60000, seed=3)
    theirs = run_java_pql(q)
    assert abs(ours.values['p1'] - theirs['p1']) < 0.012


def test_nuthi_matches_java():
    # JcTc has a royal flush (the nuts) on AcKcQc — our engine and Java should both
    # report 1.0 (100%).  PPT's nutHi is a boolean predicate; Java only accepts
    # count() on it (returns a % which the oracle normalises to [0,1]), while our
    # engine returns a 0/1 float column compatible with avg().
    q_ours = ("select avg(nutHi(PLAYER_1)) as p1_nuts "
              "from game='holdem', syntax='Generic', board='AcKcQc', dead='', "
              "PLAYER_1='JcTc', PLAYER_2='AdKd'")
    q_java = ("select count(nutHi(PLAYER_1, flop)) as p1_nuts "
              "from game='holdem', syntax='Generic', board='AcKcQc', dead='', "
              "PLAYER_1='JcTc', PLAYER_2='AdKd'")
    ours = run_pql(q_ours, seed=1)
    theirs = run_java_pql(q_java)
    assert abs(ours.values['p1_nuts'] - theirs['p1_nuts']) < 0.03


def test_min_hand_type_matches_java():
    # how often is PLAYER_1's best-5 at least a flush on the river?
    # PPT's minHandType is a boolean predicate; Java only accepts count() on it
    # (returns a % which the oracle normalises to [0,1]), while our engine returns
    # a 0/1 float column compatible with avg().
    q_ours = ("select avg(minHandType(PLAYER_1, river, flush)) as m "
              "from game='omahahi5', syntax='Generic', board='2c3c4c', dead='', "
              "PLAYER_1='AsAhKsKhQs', PLAYER_2='JdTd9d8d7d'")
    q_java = ("select count(minHandType(PLAYER_1, river, flush)) as m "
              "from game='omahahi5', syntax='Generic', board='2c3c4c', dead='', "
              "PLAYER_1='AsAhKsKhQs', PLAYER_2='JdTd9d8d7d'")
    ours = run_pql(q_ours, seed=1)
    theirs = run_java_pql(q_java)
    assert abs(ours.values['m'] - theirs['m']) < 0.02


def test_plo4_hand_vs_percentile_matches_java():
    q = ("select avg(riverEquity(PLAYER_1)) as p1, avg(riverEquity(PLAYER_2)) as p2 "
         "from game='omahahi', syntax='Generic', PLAYER_1='AsAhKsQh', PLAYER_2='25%'")
    ours = run_pql(q, trials=60000, seed=1)
    theirs = run_java_pql(q)
    assert abs(ours.values['p1'] - theirs['p1']) < 0.015


# Differential tests use queries where the requested street equals the given board
# length, so the out count is fully determined by the visible board (no dependence on
# how board completions are labeled turn vs river). That keeps ours (enumeration) and
# the jar (Monte Carlo) in exact agreement. Future-street outs (e.g. street=turn on a
# flop board) are an expectation over the unseen turn and are best queried in Monte
# Carlo mode; see pql/functions/_outs.py.
def test_outs_to_flush_matches_java_holdem():
    # KhQh on the Ah7c2h flop: 9 flush outs to the turn, deterministic.
    q = ("select avg(outsToHandType(PLAYER_1, flop, flush)) as o "
         "from game='holdem', board='Ah7c2h', PLAYER_1='KhQh', PLAYER_2='AsAd'")
    ours = run_pql(q, seed=7)
    theirs = run_java_pql(q)
    assert abs(ours.values["o"] - theirs["o"]) < 0.05


def test_outs_to_straight_gutshot_matches_java_holdem():
    # JhTh on the 8c7h2d flop: only a 9 completes the J-T-9-8-7 straight -> 4 outs.
    q = ("select avg(outsToHandType(PLAYER_1, flop, straight)) as o "
         "from game='holdem', board='8c7h2d', PLAYER_1='JhTh', PLAYER_2='AsAd'")
    ours = run_pql(q, seed=7)
    theirs = run_java_pql(q)
    assert abs(ours.values["o"] - theirs["o"]) < 0.05


def test_min_outs_flush_draw_fraction_matches_java_holdem():
    # On the Ah7c2hKs turn, KhQh always has exactly 9 river flush outs -> >=9 is always true.
    q = ("select count(minOutsToHandType(PLAYER_1, turn, flush, 9)) as fd "
         "from game='holdem', board='Ah7c2hKs', PLAYER_1='KhQh', PLAYER_2='AsAd'")
    ours = run_pql(q, seed=7)
    theirs = run_java_pql(q)
    # Java reports count as a fraction-of-trials percentage; normalize ours the same way.
    ours_frac = ours.values["fd"] / ours.trials
    assert abs(ours_frac - theirs["fd"]) < 0.02


def test_outs_to_flush_matches_java_plo4():
    # PLO4 AhKh3s4d on the Qh7h2c flop: 9 flush outs to the turn, deterministic.
    q = ("select avg(outsToHandType(PLAYER_1, flop, flush)) as o "
         "from game='omahahi', board='Qh7h2c', PLAYER_1='AhKh3s4d', PLAYER_2='AsAd5c6c'")
    ours = run_pql(q, seed=11)
    theirs = run_java_pql(q)
    assert abs(ours.values["o"] - theirs["o"]) < 0.05


def test_nut_hi_for_hand_type_matches_java():
    # AhKh holds the nut (A-high) flush on a three-heart river -> nut of its own category.
    # PPT's nutHiForHandType is a boolean predicate taking (player, street); Java accepts
    # count() (a % the oracle normalises), while our engine returns a 0/1 column for avg().
    board = "board='Qh7h2h9c4d'"
    q_ours = (f"select avg(nutHiForHandType(PLAYER_1, river)) as x "
              f"from game='holdem', syntax='Generic', {board}, PLAYER_1='AhKh', PLAYER_2='JsTd'")
    q_java = (f"select count(nutHiForHandType(PLAYER_1, river)) as x "
              f"from game='holdem', syntax='Generic', {board}, PLAYER_1='AhKh', PLAYER_2='JsTd'")
    ours = run_pql(q_ours, seed=1)
    theirs = run_java_pql(q_java)
    assert abs(ours.values["x"] - theirs["x"]) < 0.03


def test_min_hand_type_single_player_river_matches_java():
    # Single player: how often a top-10% range makes >= two pair by the river.
    q_ours = ("select count(minHandType(PLAYER_1, river, twopair)) as c "
              "from game='holdem', syntax='Generic', PLAYER_1='10%'")
    ours = run_pql(q_ours, trials=40000, seed=5)
    theirs = run_java_pql(q_ours)
    assert abs(ours.values["c"] / ours.trials - theirs["c"]) < 0.02


def test_min_hand_type_flop_and_turn_match_java():
    base = "from game='holdem', syntax='Generic', board='Ah7c2d', PLAYER_1='10%'"
    for street in ("flop", "turn"):
        q = f"select count(minHandType(PLAYER_1, {street}, twopair)) as c {base}"
        ours = run_pql(q, trials=40000, seed=5)
        theirs = run_java_pql(q)
        assert abs(ours.values["c"] / ours.trials - theirs["c"]) < 0.02, street


def test_exclusion_range_matches_java_holdem():
    q = ("select avg(riverEquity(PLAYER_1)) as p1 "
         "from game='holdem', syntax='Generic', PLAYER_1='AsKs', PLAYER_2='30%!5%'")
    ours = run_pql(q, trials=80000, seed=4)
    theirs = run_java_pql(q)
    assert abs(ours.values['p1'] - theirs['p1']) < 0.01    # was 0.03, now PPT-exact order


def test_holdem_percentile_cases_match_java():
    for r in ('10%', '30%', '5%-30%'):
        q = (f"select avg(riverEquity(PLAYER_1)) as p1 "
             f"from game='holdem', syntax='Generic', PLAYER_1='AsKs', PLAYER_2='{r}'")
        ours = run_pql(q, trials=80000, seed=4)
        theirs = run_java_pql(q)
        assert abs(ours.values['p1'] - theirs['p1']) < 0.01, r


def test_plo4_percentile_30_matches_java():
    q = ("select avg(riverEquity(PLAYER_1)) as p1 "
         "from game='omahahi', syntax='Generic', PLAYER_1='AsAhKsKh', PLAYER_2='30%'")
    ours = run_pql(q, trials=60000, seed=4)
    theirs = run_java_pql(q)
    assert abs(ours.values['p1'] - theirs['p1']) < 0.01


def test_plo5_percentile_30_matches_java():
    import os
    plo5 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plo5_class_order.json')
    if not os.path.exists(plo5):
        import pytest
        pytest.skip('plo5_class_order.json not generated yet')
    q = ("select avg(riverEquity(PLAYER_1)) as p1 "
         "from game='omahahi5', syntax='Generic', PLAYER_1='AsAhKsKhQs', PLAYER_2='30%'")
    ours = run_pql(q, trials=60000, seed=4)
    theirs = run_java_pql(q)
    assert abs(ours.values['p1'] - theirs['p1']) < 0.02


# --- PQL where / comparison / histogram differentials (2026-06-03) ---
_SCEN_E = "game='holdem', board='Ah7c2h', PLAYER_1='KhQh', PLAYER_2='QdQc'"

# The jar prints histogram buckets labelled by lowercase short category TOKENS, e.g.
#   H = [nothing:24.47% (4893)],[pair:31.85% (6369)],...,[str. fl.:0.12% (24)]
# in poker-strength order (index 0..8).  Map those tokens to the same integer category
# indices our engine uses in hist["pairs"][i]["value"].  run_java_pql() only parses
# scalar "NAME = FLOAT" lines, so we run the jar directly and parse the H = [...] line.
_JAR_HIST_TOKEN_TO_INDEX = {
    "nothing": 0,    # High Card
    "pair": 1,       # One Pair
    "twopair": 2,    # Two Pair
    "trips": 3,      # Three of a Kind
    "strt.": 4,      # Straight
    "flush": 5,      # Flush
    "full h.": 6,    # Full House
    "quads": 7,      # Four of a Kind
    "str. fl.": 8,   # Straight Flush
}


def _run_java_histogram(query, max_trials=20000, max_seconds=60):
    """Run a histogram PQL query through the jar and return {category_index: fraction}.

    Parses the jar's  H = [token:PCT% (count)],[...]  line.  Returns fractions in
    [0, 1] keyed by the integer category index that matches our engine's
    hist["pairs"][i]["value"].
    """
    jar_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "java_files")
    jar = os.path.join(jar_dir, "p2.jar")
    cmd = [
        "java", "-XX:TieredStopAtLevel=1", "-Xshare:auto",
        "-cp", jar, "propokertools.cli.RunPQL",
        "-mt", str(max_trials), "-ms", str(max_seconds), query,
    ]
    out = subprocess.check_output(cmd, cwd=jar_dir, text=True, timeout=max_seconds + 30)
    result = {}
    # Each bucket: [label:PCT% (count)]  -- label may contain spaces/dots (e.g. "str. fl.").
    for label, pct in re.findall(r"\[([^:\]]+):([0-9]*\.?[0-9]+)%", out):
        idx = _JAR_HIST_TOKEN_TO_INDEX.get(label.strip())
        if idx is not None:
            result[idx] = float(pct) / 100.0
    return result


def test_diff_where_conditional_equity():
    q = f"select avg(riverEquity(PLAYER_1)) as e from {_SCEN_E} where minHandType(PLAYER_1,river,flush)"
    ours = run_pql(q, trials=20000, seed=123).values["e"]
    theirs = run_java_pql(q)["e"]
    assert abs(ours - theirs) < 0.01


def test_diff_where_denominator_is_filtered():
    q = f"select count(winsHi(PLAYER_1)) as w from {_SCEN_E} where winsHi(PLAYER_1)"
    r = run_pql(q, trials=20000, seed=123)
    # 100% within the filtered subset (engine-internal invariant): every surviving
    # trial passed the WHERE predicate winsHi(PLAYER_1), so the count equals trials.
    assert r.values["w"] == r.trials


def test_diff_histogram_category_percentages():
    q = f"select histogram(handType(PLAYER_1,river)) as h from {_SCEN_E}"
    r = run_pql(q, trials=20000, seed=123)
    ours_pct = {p["value"]: p["count"] / r.trials for p in r.histograms["h"]["pairs"]}
    theirs = _run_java_histogram(q)
    # Compare buckets present in both within 0.015; skip any we cannot confidently match.
    compared = 0
    for idx, frac in ours_pct.items():
        match = theirs.get(idx)
        if match is not None:
            assert abs(frac - match) < 0.015, (idx, frac, match)
            compared += 1
    assert compared >= 3  # ensure the comparison actually exercised real buckets


def test_diff_arithmetic_constant_shift():
    base = run_pql(
        f"select avg(outsToHandType(PLAYER_1,flop,flush)) as o from {_SCEN_E}",
        trials=8000, seed=123,
    ).values["o"]
    shifted = run_pql(
        f"select avg(outsToHandType(PLAYER_1,flop,flush) + 1) as o from {_SCEN_E}",
        trials=8000, seed=123,
    ).values["o"]
    assert abs(shifted - (base + 1)) < 1e-9
