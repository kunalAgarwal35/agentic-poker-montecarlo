from itertools import combinations

import numpy as np

from card_encoding import generate_deck_ints, hand_str_to_ints, ints_to_hand_str
from hand_rank_evaluator import get_score_array
from range_ladder import (
    DEFAULT_BUCKETS,
    _bucket_slices_and_edges,
    compute_range_ladder,
    rank_hands,
    sample_runouts,
    sample_villains,
    score_hands,
)

# Fixture deliberately differs from the task-7 brief's literal example
# (board="6s7s4s", dead=["AsKs9h2c", "QhJhTd3d"]). That fixture is legal
# (no card collisions -- compute_range_ladder accepts it) and four of the
# five properties below pass on it cleanly. But
# test_equity_is_monotonic_in_bucket_width fails on it, reproducibly, for
# the "nuts" ladder:
#
#   seed=5  nuts: [0.6508, 0.8469, 0.8534, 0.8501, 0.8690, 0.9135]   dip at 25%->40%
#   seed=1  nuts: [0.7043, 0.8662, 0.8460, 0.8519, 0.8631, 0.9106]   dip at 15%->25%
#   seed=3  nuts: [0.7032, 0.8653, 0.8542, 0.8623, 0.8738, 0.9160]   dip at 15%->25%
#   seed=4  nuts: [0.6839, 0.8526, 0.8192, 0.8398, 0.8525, 0.9008]   dip at 15%->25%
#
# ("air" is monotonic in every trial -- only "nuts" dips.) Investigation
# (see task-7-report.md) ruled out the three implementation bugs the brief
# names as likely causes: the ranking reduction's beat_sum/beat_cnt indexing
# is aligned (verified by inspection), the bucket slicer sorts `-strength`
# (strongest first, correct), and `take` never rounds to 0 at n~=3000. (That
# investigation predates Task 13's two-pass rewrite, so it names symbols
# that no longer exist: `evaluate_population` is now `rank_hands` +
# `_reduce_rank_scores`, which keeps beat_sum/beat_cnt but has no
# `hero_sum` at all, and the sort moved from `build_rungs` into
# `_bucket_slices_and_edges`. The conclusion still holds; the names are
# corrected here per final review, Finding 9.) It
# also ruled out plain Monte Carlo noise: holding the fixture's cards fixed
# and increasing hands/runouts by 10x (3000/300 -> 30000/2000) does not
# shrink the dip below ~0.3-3.4 percentage points; it is not a 1/sqrt(N)
# effect, and it reproduces at the *same* bucket transition (15%->25% or
# 25%->40%) across every seed tried.
#
# The cause is structural, not a bug: AsKs9h2c on a monotone 6s7s4s flop is
# the *effective* nuts (best reachable flush) but not the *mathematical*
# nuts -- it is still beaten by any runout that fills a boat, quads or
# straight flush. "Villain strength" ranks by equity against the whole
# population (all hand types); "hero equity" is nuts's specific matchup
# equity. A villain that ranks only moderately by population strength can
# still carry disproportionate boat-redraw equity specifically against a
# flush, so hero equity vs. bucket B is not guaranteed monotonic in B even
# with unlimited samples. This is a fixture problem, not an implementation
# bug: no monotone-flop "made flush" hero can serve as a `nuts` fixture for
# this property, because it is never literally unbeatable.
#
# The fixture below sidesteps the ambiguity by making "nuts" *provably*
# unbeatable, not just probably so. Board 3s4s5s is monotone spades; hero
# holds 6s7s8s2d. Using 6s+7s against the board makes 3s4s5s6s7s -- a made
# straight flush -- and 8s is also dead, which blocks the only higher
# same-suit window (4s5s6s7s8s). No other window through ranks 6/7/8 is
# possible (all need a dead card), the only remaining same-suit window is
# the wheel (As2s3s4s5s), which is a *lower* straight flush, and no
# different-suit straight flush can ever form: the board's 3 fixed cards
# are all spades, so a different suit needs 3 board cards of that suit and
# the 2-card runout cannot supply more than 2. So no villain hand and no
# runout, ever, beats hero "nuts" here -- equity is exactly 1.0 against
# every villain on every runout, by construction, not by sampling luck.
# That makes its ladder trivially, deterministically flat (monotonic with
# zero slack). "air" (QcJc9d2h) has no pair, no flush cards and no
# straight-relevant ranks against this board and was confirmed monotonic
# across 10 independent seeds (5,1,2,3,4,6,7,8,9,10) at hands=3000,
# runouts=300 -- see task-7-report.md for the full sweep, including
# bucket-5 nuts-air separation of ~1.00 in every trial (well over the 0.4
# floor) and 6 distinct boundary hands per ladder.
# Task 13: two passes replace Task 11's single `trials` count.
# `hands`/`rank_runouts` (Pass 1, ranks villain hands on shared runouts) and
# `trials_per_bucket` (Pass 2, stratified equity on independent runouts) are
# both far smaller than the old grid needed: these properties need
# discrimination between a provably-1.0 hero and a near-0.0 hero, not
# fine-grained precision, so hands=3000/rank_runouts=20/trials_per_bucket=1500
# is comfortable headroom while running in a fraction of the old grid's time.
FLOP = "3s4s5s"
DEAD = ["6s7s8s2d", "QcJc9d2h"]
HEROES = [{"id": "nuts", "cards": "6s7s8s2d"}, {"id": "air", "cards": "QcJc9d2h"}]


def _ladder(**kw):
    return compute_range_ladder(board=FLOP, dead=DEAD, heroes=HEROES,
                                hands=3000, rank_runouts=20, trials_per_bucket=1500,
                                seed=kw.pop("seed", 5), **kw)


def test_equity_is_monotonic_in_bucket_width():
    """NOT a general property of this feature -- see Task 13 fix-round-2
    Finding 3. Task 13's hand ranking means equity vs. bucket B is not
    guaranteed non-decreasing in B even with unlimited samples: a hero's
    made hand can have disproportionate redraw equity against villains
    that rank only moderately by population strength, so real (small,
    reproducible) dips exist at the shipped defaults for ordinary heroes
    (see range_ladder.py's compute_range_ladder docstring for a measured
    example). A dip there would NOT mean the ranking is inverted or the
    slices are wrong.

    This assertion is still meaningful HERE, and only here, because the
    fixture below makes "nuts" *provably* unbeatable (equity exactly 1.0
    against every villain on every runout, by construction -- see the
    fixture comment above), not just usually strong. A perfectly flat 1.0
    line cannot dip, so a failure on `nuts` really would mean the pipeline
    is broken (ranking inverted, slices misapplied, wrong hero scored --
    not the structural non-monotonicity Finding 3 describes). "air" was
    separately confirmed monotonic across 10 independent seeds (see above)
    for the same reason `nuts` is guaranteed to be: it is nowhere near a
    strong hand's own redraw-equity regime on this board.
    """
    out = _ladder()
    for lad in out["ladders"]:
        eq = [r["equity"] for r in lad["rungs"]]
        assert eq == sorted(eq), f"{lad['id']}: {eq}"


# ---------------------------------------------------------------------------
# Spec section 9, "Identity": hero equity vs the top 100% bucket must equal
# plain equity against a random hand, computed independently.
#
# The previous version of this test asserted `0.0 <= equity <= 1.0` and
# called it a recomputation. `equity` is the mean of a hero_results row whose
# entries come from {0.0, 0.5, 1.0}, so that assertion cannot fail for any
# value the code can produce -- the final review killed it by mutation
# (scaling bucket-100 equity by 0.5 and by 0.8 both left it PASSING). This
# version recomputes the quantity for real.
#
# Fixture: the live deck is shrunk to 12 cards, so that the villain
# POPULATION is the complete C(12,4) = 495-hand space (sample_villains'
# exhaustive branch) and every one of its C(8,2) = 28 compatible runouts can
# be enumerated. The reference is therefore EXACT -- a full enumeration with
# no Monte Carlo error of its own -- and it is exactly the quantity Pass 2
# estimates: draw a villain uniformly from the bucket, then a runout
# uniformly from the ones compatible with it. Every hand has the same 28
# compatible runouts, so that two-stage average is well defined.
#
# The board is 3s4s5s and the live cards hold no spades, which makes the
# three heroes span the range instead of all sitting at an extreme:
# "straight" is exactly 1.0, "pair" is ~0.49 (the value that actually
# constrains), "low" is ~0.33.
# ---------------------------------------------------------------------------

IDENT_BOARD = "3s4s5s"
IDENT_LIVE = ["Ah", "Kh", "Qh", "9h", "8h", "7h", "Ad", "Kd", "9d", "8d", "7d", "2c"]
IDENT_HEROES = [
    {"id": "straight", "cards": "6d7cQcJs"},
    {"id": "pair", "cards": "TcTdJhJd"},
    {"id": "low", "cards": "6h6cKcQs"},
]
# Pass-2 trials per bucket for the Identity fixture. Sets the tolerance
# below; see its derivation there.
IDENT_TRIALS = 8000


def _identity_dead():
    """`dead` for the Identity fixture: every hero, plus one blocker entry
    holding every card that is neither on the board, nor in a hero, nor in
    IDENT_LIVE -- which is what shrinks the live deck to exactly 12 cards."""
    used = {IDENT_BOARD[i:i + 2] for i in range(0, len(IDENT_BOARD), 2)}
    for h in IDENT_HEROES:
        used |= {h["cards"][i:i + 2] for i in range(0, len(h["cards"]), 2)}
    full = ints_to_hand_str(generate_deck_ints([]))
    blocker = [full[i:i + 2] for i in range(0, len(full), 2)
              if full[i:i + 2] not in used and full[i:i + 2] not in IDENT_LIVE]
    return [h["cards"] for h in IDENT_HEROES] + ["".join(blocker)]


def _exact_equity_versus_the_whole_population(hero_cards):
    """Hero equity against a uniformly random villain hand on a uniformly
    random compatible runout, by EXHAUSTIVE enumeration -- no sampling.

    Deliberately independent of Pass 2: it builds every (villain hand,
    compatible runout) pair itself and averages hero's win/tie/loss, rather
    than reusing sample_pass2_trials/evaluate_pass2 (which would only prove
    the ladder agrees with itself). It does share `score_hands`, whose
    numba-vs-numpy equivalence is pinned separately in test_fast_score.py --
    what is under test here is the equity denominator and the per-bucket
    aggregation, not the scoring kernel.
    """
    board_ints = hand_str_to_ints(IDENT_BOARD)
    deck = sorted(hand_str_to_ints("".join(IDENT_LIVE)).tolist())
    hero = hand_str_to_ints(hero_cards)
    all_runouts = list(combinations(deck, 2))

    hands, boards, per_hand_counts = [], [], []
    for villain in combinations(deck, 4):
        held = set(villain)
        compatible = [r for r in all_runouts if not held & set(r)]
        per_hand_counts.append(len(compatible))
        for runout in compatible:
            hands.append(villain)
            boards.append(board_ints.tolist() + list(runout))
    # Every hand blocks the same number of runouts (hole_count is constant),
    # so the per-hand averages below all carry equal weight.
    assert len(set(per_hand_counts)) == 1, per_hand_counts[:5]

    hands = np.array(hands, dtype=np.int32)
    boards = np.array(boards, dtype=np.int32)
    hero_tile = np.broadcast_to(hero, (hands.shape[0], hero.size))
    score_array = get_score_array()
    villain_scores = score_hands(hands, boards, "plo4", score_array)
    hero_scores = score_hands(hero_tile, boards, "plo4", score_array)
    results = np.where(hero_scores > villain_scores, 1.0,
                      np.where(hero_scores == villain_scores, 0.5, 0.0))
    per_hand = results.reshape(-1, per_hand_counts[0]).mean(axis=1)
    return float(per_hand.mean())


def test_top_100_percent_equals_equity_versus_a_random_hand():
    """Spec section 9's "Identity" property, recomputed independently.

    Tolerance derivation: the bucket-100 rung is the mean of IDENT_TRIALS
    i.i.d. Pass-2 trials, each a draw from {0.0, 0.5, 1.0}. Any random
    variable supported on [0, 1] has variance <= 1/4, so sd <= 0.5 and the
    standard error of that mean is at most 0.5/sqrt(IDENT_TRIALS)
    (= 0.0056 at 8,000). The reference is an exhaustive enumeration and
    contributes no error of its own, so 4 standard errors --
    2/sqrt(IDENT_TRIALS) = 0.0224 -- is the whole budget. Four sigma rather
    than three because this is a merge gate, not a research result; the
    fixture is seeded end to end, so the test is deterministic anyway and
    the sigma count only says how much genuine sampling slack is allowed
    before a real discrepancy is called.

    Verified to bite: scaling the production bucket equities by 0.9 moves
    "pair" by 0.049 and "straight" by 0.100, both well outside 0.0224, and
    the test fails on both. The old tautology survived a 0.5x scaling.
    """
    tolerance = 4 * 0.5 / np.sqrt(IDENT_TRIALS)
    out = compute_range_ladder(
        board=IDENT_BOARD, dead=_identity_dead(), heroes=IDENT_HEROES,
        hands=1000, rank_runouts=40, trials_per_bucket=IDENT_TRIALS, seed=5,
    )
    # The population must really be the whole enumerated space -- otherwise
    # the reference below is answering a different question.
    assert out["population"] == 495, out["population"]

    for hero, lad in zip(IDENT_HEROES, out["ladders"]):
        assert lad["id"] == hero["id"]
        measured = [r["equity"] for r in lad["rungs"] if r["bucket"] == 100][0]
        expected = _exact_equity_versus_the_whole_population(hero["cards"])
        assert abs(measured - expected) <= tolerance, (
            f"{hero['id']}: bucket-100 equity {measured:.5f} vs exhaustive "
            f"head-to-head {expected:.5f} (tolerance {tolerance:.5f})"
        )

    # The three heroes must also be spread out, so the check above is a
    # comparison of real numbers and not three coincidences near 0 or 1.
    by_id = {l["id"]: {r["bucket"]: r["equity"] for r in l["rungs"]}
             for l in out["ladders"]}
    assert by_id["straight"][100] > by_id["pair"][100] > by_id["low"][100]


def test_boundary_hands_are_distinct_across_buckets():
    # The weak half of spec section 9's "Boundary ordering": the edges must
    # at least not all be the same hand. The ordering itself -- the property
    # that actually matters -- is checked directly below, in
    # test_edge_hand_strength_is_non_increasing_as_buckets_widen. This one
    # stays because it is the only check that runs against
    # compute_range_ladder's PUBLIC response rather than against the
    # module-level functions, so it would catch an edge that never reaches
    # the rungs. (Its earlier comment claimed the ordering was untestable
    # because "strength lives only inside build_rungs" -- that was wrong on
    # two counts: build_rungs never receives `strength` at all, and
    # `_bucket_slices_and_edges`, which does, is module-level and directly
    # callable. Final review, Finding 5.)
    out = _ladder()
    rungs = out["ladders"][0]["rungs"]
    assert len({r["edge"]["cards"] for r in rungs}) > 1


def test_edge_hand_strength_is_non_increasing_as_buckets_widen():
    """Spec section 9, "Boundary ordering": each bucket's edge hand is the
    WEAKEST hand in a cumulative top-pct slice, so widening the bucket can
    only admit weaker hands -- the edge's own Pass-1 strength must never go
    UP as pct grows.

    Runs Pass 1 through its module-level entry points (`sample_villains` /
    `sample_runouts` / `rank_hands`) and then hands the resulting
    `strength` to the very function compute_range_ladder uses to slice it,
    `_bucket_slices_and_edges`, so this asserts against production code
    rather than a reimplementation of it. That is also why the edges'
    reported cards are cross-checked against the slice indices below: it
    pins that the strength being asserted on belongs to the hand the
    response actually names.
    """
    board_ints = hand_str_to_ints(FLOP)
    dead_cards = [d[i:i + 2] for d in DEAD for i in range(0, len(d), 2)]
    board_cards = [FLOP[i:i + 2] for i in range(0, len(FLOP), 2)]
    deck = generate_deck_ints(dead_cards + board_cards)

    rng = np.random.default_rng(5)
    villain_hands = sample_villains(deck, 4, 3000, rng)
    runouts = sample_runouts(deck, len(board_ints), 20, rng)
    strength, counts = rank_hands(villain_hands, board_ints, runouts,
                                  "plo4", get_score_array())
    ranked = counts > 0
    villain_hands, strength = villain_hands[ranked], strength[ranked]

    buckets = list(DEFAULT_BUCKETS)                # 5, 15, 25, 40, 60, 100
    assert buckets == sorted(buckets), "this test assumes widening buckets"
    bucket_indices, edges = _bucket_slices_and_edges(
        strength, villain_hands, buckets, board_ints, runouts,
    )

    edge_strengths = [float(strength[idx[-1]]) for idx in bucket_indices]
    assert all(a >= b for a, b in zip(edge_strengths, edge_strengths[1:])), (
        f"edge strengths rose as buckets widened: "
        f"{list(zip(buckets, edge_strengths))}"
    )
    # ...and it really is the reported edge hand's strength.
    for idx, edge in zip(bucket_indices, edges):
        assert ints_to_hand_str(villain_hands[idx[-1]]) == edge["cards"]
    # Not a flat line: at least one strictly decreasing step, so the
    # assertion above is doing work rather than comparing a constant.
    assert edge_strengths[0] > edge_strengths[-1], edge_strengths


def test_ranking_is_hero_independent():
    """Both heroes must see identical boundary hands. This is architecturally
    guaranteed, not proven here -- build_rungs is called with the same
    `strength` array for both heroes in this fixture, so equal edges are
    expected by construction, not evidence of a deeper independence
    property. It stays as a regression guard: it is the one property in
    this file that would catch a regression to per-hero ranking (e.g. if
    compute_range_ladder started sorting by hero_equity instead of the
    shared `strength`)."""
    out = _ladder()
    edges = [[r["edge"]["cards"] for r in lad["rungs"]] for lad in out["ladders"]]
    assert edges[0] == edges[1]


def test_nuts_and_air_are_distinguishable_against_the_tightest_slice():
    """If this fails the feature is pointless -- see spec section 6.

    Tightened from the brief's literal `> 0.4` floor: `nuts` is exactly 1.0
    here by construction (see the fixture comment above), so `> 0.4` reduces
    to `air[5] < 0.6` and never actually exercises `nuts`'s side of the
    comparison. Under review's five mutants `air[5]` peaked at 0.507, so the
    0.4 floor left a 0.49 margin of slack -- loose enough that a broken
    ranking still cleared it. `air`'s real bucket-5 baseline is exactly 0.0
    in every seed tried (see task-7-report.md), so a floor of 0.9 (i.e.
    requiring air[5] < 0.1, a 10x-wider-than-observed tolerance around that
    true 0.0) leaves ample room for sampling noise while actually failing
    against the 0.507 mutant.
    """
    out = _ladder()
    by_id = {l["id"]: {r["bucket"]: r["equity"] for r in l["rungs"]} for l in out["ladders"]}
    assert by_id["nuts"][5] - by_id["air"][5] > 0.9


def test_bucket_equity_is_per_slice_not_a_repeated_population_wide_number():
    """Defends against a ladder that ignores bucket slicing entirely and
    reports the whole-population (bucket-100) equity for every rung -- a
    mutant that killed none of the other four properties here, because
    `nuts` is 1.0 at every bucket (by construction of this fixture) whether
    or not slicing works at all, and the hero-independence/discrimination/
    boundary-distinctness checks don't look at how `equity` changes across
    buckets for a single hero. `air`'s tightest-slice equity (0.0, see the
    fixture comment above) versus its whole-population equity (~0.12-0.16
    across seeds, always > 0.12) is the one place in this file that number
    actually has to move. Margin (0.05, well under the observed ~0.12
    minimum) is there so ordinary sampling noise can't flip this by luck.
    """
    out = _ladder()
    by_id = {l["id"]: {r["bucket"]: r["equity"] for r in l["rungs"]} for l in out["ladders"]}
    assert by_id["air"][100] - by_id["air"][5] > 0.05
