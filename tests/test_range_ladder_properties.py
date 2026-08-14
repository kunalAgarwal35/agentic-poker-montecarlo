from range_ladder import compute_range_ladder

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
# names as likely causes: `evaluate_population`'s beat_cnt/hero_sum indexing
# is aligned (verified by inspection), `build_rungs` sorts `-strength`
# (strongest first, correct), and `take` never rounds to 0 at n~=3000. It
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


def test_top_100_percent_equals_equity_versus_a_random_hand():
    """Independent check of the equity denominator."""
    out = _ladder()
    for lad in out["ladders"]:
        rung100 = [r for r in lad["rungs"] if r["bucket"] == 100][0]
        # recompute directly: mean hero equity over the whole population
        assert 0.0 <= rung100["equity"] <= 1.0
    nuts = [l for l in out["ladders"] if l["id"] == "nuts"][0]["rungs"]
    air = [l for l in out["ladders"] if l["id"] == "air"][0]["rungs"]
    top100 = {r["bucket"]: r["equity"] for r in nuts}[100]
    assert top100 > {r["bucket"]: r["equity"] for r in air}[100]


def test_boundary_hands_are_distinct_across_buckets():
    # Named for what it actually checks, not for what the brief's original
    # name ("...strength_falls_as_buckets_widen") promised. The returned
    # rungs expose each boundary hand's cards/category but not its
    # `strength` value -- that number lives only inside build_rungs' local
    # `strength` array and is never returned by compute_range_ladder -- so
    # there is nothing in the public response to assert non-increasing
    # strength against without reimplementing (part of) the sampling
    # pipeline separately from compute_range_ladder and hoping it lines up.
    # Per review guidance this is the documented fallback: a renamed test
    # matching its weaker, actually-checked assertion, flagged in
    # task-7-report.md, rather than inventing a new accessor on
    # range_ladder.py to satisfy the original name.
    out = _ladder()
    rungs = out["ladders"][0]["rungs"]
    assert len({r["edge"]["cards"] for r in rungs}) > 1


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
