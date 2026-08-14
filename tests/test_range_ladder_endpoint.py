"""Task 9: POST /range_ladder -- the Flask endpoint over range_ladder.compute_range_ladder.

Payload sizes here are deliberately tiny (small hands/rank_runouts/
trials_per_bucket) so the suite stays fast; the shipped defaults are ~5-6s
per call and are exercised by range_ladder.py's own test suite, not here.
"""
import pytest

import server
from server import (
    app,
    DEFAULT_HANDS,
    DEFAULT_RANK_RUNOUTS,
    DEFAULT_TRIALS_PER_BUCKET,
    MAX_HEROES,
    MAX_BUCKETS,
)

client = app.test_client()

_RANKS = "23456789TJQKA"
_SUITS = "shdc"
_ALL_CARDS = [r + s for r in _RANKS for s in _SUITS]  # 52 distinct two-char cards


def _distinct_cards(n, exclude=()):
    """First `n` two-char cards not in `exclude`, in a fixed deterministic order."""
    excluded = set(exclude)
    out = [c for c in _ALL_CARDS if c not in excluded]
    assert len(out) >= n, "not enough cards left in the deck for this fixture"
    return out[:n]


def test_range_ladder_returns_one_rung_per_bucket_in_requested_order():
    # Non-default, non-sorted bucket order -- proves the response echoes
    # the CALLER's order rather than always emitting DEFAULT_BUCKETS order.
    # River board (5 cards): cheap, and reports exact:False because
    # hands=5000 subsamples the C(39,4)=82,251 legal hands this deck allows
    # (final review, Finding 1 -- exactness is about the POPULATION too, not
    # just about having no runout left to draw).
    # hands=5000 x heroes=2 is also comfortably above the pool's parallel
    # dispatch threshold, so this test incidentally warms the process pool
    # for the rest of the module.
    resp = client.post("/range_ladder", json={
        "board": "6s7s4s2h9d",
        "dead": ["AsKs9h2c", "3dTc5s8h"],
        "heroes": [{"id": "nuts", "cards": "AsKs9h2c"},
                   {"id": "air", "cards": "3dTc5s8h"}],
        "buckets": [100, 5, 50],
        "hands": 5000,
        "seed": 3,
    })
    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()
    assert body["exact"] is False
    assert body["population"] > 0
    assert len(body["ladders"]) == 2
    for ladder in body["ladders"]:
        rungs = ladder["rungs"]
        assert [r["bucket"] for r in rungs] == [100, 5, 50]
        for rung in rungs:
            assert 0.0 <= rung["equity"] <= 1.0
            assert "cards" in rung["edge"] and "category" in rung["edge"]


def test_river_with_an_enumerated_population_reports_exact_true():
    # The other direction of final review Finding 1, end to end through the
    # route: same river board as above, but the live deck is shrunk to 12
    # cards (everything else parked in `dead`) so C(12,4)=495 < hands and
    # sample_villains enumerates the whole population instead of sampling
    # it. Only then may the response claim exactness.
    board = "6s7s4s2h9d"
    hero = "AsKs9h2c"
    used = {board[i:i + 2] for i in range(0, len(board), 2)}
    used |= {hero[i:i + 2] for i in range(0, len(hero), 2)}
    live = _distinct_cards(12, exclude=used)
    blocker = "".join(c for c in _ALL_CARDS if c not in used and c not in live)
    resp = client.post("/range_ladder", json={
        "board": board,
        "dead": [hero, blocker],
        "heroes": [{"id": "nuts", "cards": hero}],
        "hands": 5000,
        "rank_runouts": 5,
        "trials_per_bucket": 50,
        "seed": 3,
    })
    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()
    assert body["population"] == 495
    assert body["exact"] is True


def test_missing_board_is_a_400():
    resp = client.post("/range_ladder", json={
        "heroes": [{"id": "u1", "cards": "AsKs9h2c"}],
        "dead": ["AsKs9h2c"],
    })
    assert resp.status_code == 400, resp.get_data(as_text=True)


def test_missing_heroes_is_a_400():
    resp = client.post("/range_ladder", json={"board": "6s7s4s", "dead": []})
    assert resp.status_code == 400, resp.get_data(as_text=True)


def test_hero_not_in_dead_is_a_422():
    resp = client.post("/range_ladder", json={
        "board": "6s7s4s",
        "dead": ["AsKs9h2c"],
        "heroes": [{"id": "u1", "cards": "QsJs8c3h"}],   # not in dead
        "hands": 300, "rank_runouts": 10, "trials_per_bucket": 50, "seed": 1,
    })
    assert resp.status_code == 422, resp.get_data(as_text=True)
    body = resp.get_json()
    assert "details" in body


def test_oversized_hands_is_clamped_not_accepted_or_rejected():
    # A wildly oversized `hands` must neither be accepted at face value
    # (which would run a request costing minutes on this box) nor rejected
    # outright (400/422) -- it should be silently clamped down to
    # DEFAULT_HANDS and the request should still succeed.
    resp = client.post("/range_ladder", json={
        "board": "6s7s4s",
        "dead": ["AsKs9h2c"],
        "heroes": [{"id": "u1", "cards": "AsKs9h2c"}],
        "hands": 10_000_000,
        "rank_runouts": 5,
        "trials_per_bucket": 50,
        "seed": 1,
    })
    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()
    # `population` reflects however many villain hands actually got
    # sampled and ranked -- proof the endpoint never tried to sample
    # 10,000,000 hands.
    assert 0 < body["population"] <= DEFAULT_HANDS


def test_unauthorized_when_engine_key_set(monkeypatch):
    monkeypatch.setenv("ENGINE_KEY", "testkey")
    resp = client.post("/range_ladder", json={
        "board": "6s7s4s",
        "dead": ["AsKs9h2c"],
        "heroes": [{"id": "u1", "cards": "AsKs9h2c"}],
        "hands": 300, "rank_runouts": 10, "trials_per_bucket": 50, "seed": 1,
    })
    assert resp.status_code == 401, resp.get_data(as_text=True)


# ---------------------------------------------------------------------------
# Fix round 1, IMPORTANT 1: `raw or DEFAULT` treated a literal 0 as absent,
# so {"hands": 0} silently bought the CEILING (the most expensive
# computation) for a caller who asked for the least. Each clamp must floor
# a literal 0 -- and int 0 / string "0" must agree, since the bug's other
# half was that they didn't.
# ---------------------------------------------------------------------------

def test_hands_zero_clamps_to_the_floor_not_the_default():
    payload = {
        "board": "6s7s4s",
        "dead": ["AsKs9h2c"],
        "heroes": [{"id": "u1", "cards": "AsKs9h2c"}],
        "rank_runouts": 5, "trials_per_bucket": 50, "seed": 1,
    }
    resp_int = client.post("/range_ladder", json={**payload, "hands": 0})
    resp_str = client.post("/range_ladder", json={**payload, "hands": "0"})
    assert resp_int.status_code == 200, resp_int.get_data(as_text=True)
    assert resp_str.status_code == 200, resp_str.get_data(as_text=True)
    pop_int = resp_int.get_json()["population"]
    pop_str = resp_str.get_json()["population"]
    # Floor is 100 -- nowhere near DEFAULT_HANDS (60,000), which is what the
    # pre-fix `raw or DEFAULT_HANDS` bug would have bought instead.
    assert pop_int <= 100, pop_int
    assert pop_str <= 100, pop_str
    assert pop_int == pop_str, "int 0 and string \"0\" must clamp identically"


def test_rank_runouts_zero_clamps_to_the_floor_not_the_default():
    # A flop (not river): `rank_runouts` is actually consumed here, unlike
    # on the river where sample_runouts always returns a single empty
    # completion regardless of the requested count.
    payload = {
        "board": "6s7s4s",
        "dead": ["AsKs9h2c"],
        "heroes": [{"id": "u1", "cards": "AsKs9h2c"}],
        "hands": 300, "trials_per_bucket": 50, "seed": 1,
    }
    resp_int = client.post("/range_ladder", json={**payload, "rank_runouts": 0})
    resp_str = client.post("/range_ladder", json={**payload, "rank_runouts": "0"})
    assert resp_int.status_code == 200, resp_int.get_data(as_text=True)
    assert resp_str.status_code == 200, resp_str.get_data(as_text=True)
    assert resp_int.get_json()["rank_runouts"] == 5
    assert resp_str.get_json()["rank_runouts"] == 5


def test_trials_per_bucket_zero_clamps_to_the_floor_not_the_default():
    payload = {
        "board": "6s7s4s2h9d",
        "dead": ["AsKs9h2c"],
        "heroes": [{"id": "u1", "cards": "AsKs9h2c"}],
        "hands": 300, "rank_runouts": 5, "seed": 1,
    }
    resp_int = client.post("/range_ladder", json={**payload, "trials_per_bucket": 0})
    resp_str = client.post("/range_ladder", json={**payload, "trials_per_bucket": "0"})
    assert resp_int.status_code == 200, resp_int.get_data(as_text=True)
    assert resp_str.status_code == 200, resp_str.get_data(as_text=True)
    assert resp_int.get_json()["trials_per_bucket"] == 50
    assert resp_str.get_json()["trials_per_bucket"] == 50


# ---------------------------------------------------------------------------
# Fix round 1, MINOR: the original suite only ever exercised the `hands`
# ceiling; `rank_runouts` and `trials_per_bucket` share the same clamp
# shape and deserve the same coverage.
# ---------------------------------------------------------------------------

def test_oversized_rank_runouts_is_clamped_not_accepted_or_rejected():
    resp = client.post("/range_ladder", json={
        "board": "6s7s4s",
        "dead": ["AsKs9h2c"],
        "heroes": [{"id": "u1", "cards": "AsKs9h2c"}],
        "hands": 300,
        "rank_runouts": 10_000_000,
        "trials_per_bucket": 50,
        "seed": 1,
    })
    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()
    assert body["rank_runouts"] <= DEFAULT_RANK_RUNOUTS


def test_oversized_trials_per_bucket_is_clamped_not_accepted_or_rejected():
    resp = client.post("/range_ladder", json={
        "board": "6s7s4s2h9d",   # river: trials_per_bucket just echoes the
                                  # (already-clamped) input, and stays cheap
                                  # regardless of the requested value here.
        "dead": ["AsKs9h2c"],
        "heroes": [{"id": "u1", "cards": "AsKs9h2c"}],
        "hands": 300,
        "rank_runouts": 5,
        "trials_per_bucket": 10_000_000,
        "seed": 1,
    })
    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()
    assert body["trials_per_bucket"] <= DEFAULT_TRIALS_PER_BUCKET


# ---------------------------------------------------------------------------
# Fix round 1, IMPORTANT 2: Pass-2 cost scales with
# len(buckets) x (1 + len(heroes)), so both need a ceiling -- but as
# rejections (400), not silent clamps, since quietly dropping a caller's
# 11th hero would answer a different, smaller question than the one asked.
# Empty `heroes`/`buckets` are degenerate requests and also rejected.
# ---------------------------------------------------------------------------

def test_too_many_heroes_is_a_400():
    resp = client.post("/range_ladder", json={
        "board": "6s7s4s",
        "dead": ["AsKs9h2c"],
        "heroes": [{"id": f"h{i}", "cards": "AsKs9h2c"} for i in range(MAX_HEROES + 1)],
        "hands": 100, "rank_runouts": 5, "trials_per_bucket": 50, "seed": 1,
    })
    assert resp.status_code == 400, resp.get_data(as_text=True)


def test_empty_heroes_is_a_400():
    resp = client.post("/range_ladder", json={
        "board": "6s7s4s", "dead": [], "heroes": [],
    })
    assert resp.status_code == 400, resp.get_data(as_text=True)


def test_too_many_buckets_is_a_400():
    resp = client.post("/range_ladder", json={
        "board": "6s7s4s",
        "dead": ["AsKs9h2c"],
        "heroes": [{"id": "u1", "cards": "AsKs9h2c"}],
        "buckets": list(range(1, MAX_BUCKETS + 2)),
        "hands": 100, "rank_runouts": 5, "trials_per_bucket": 50, "seed": 1,
    })
    assert resp.status_code == 400, resp.get_data(as_text=True)


def test_empty_buckets_is_a_400():
    resp = client.post("/range_ladder", json={
        "board": "6s7s4s",
        "dead": ["AsKs9h2c"],
        "heroes": [{"id": "u1", "cards": "AsKs9h2c"}],
        "buckets": [],
        "hands": 100, "rank_runouts": 5, "trials_per_bucket": 50, "seed": 1,
    })
    assert resp.status_code == 400, resp.get_data(as_text=True)


def test_exactly_max_heroes_still_succeeds():
    board = "6s7s4s"
    board_cards = {board[i:i + 2] for i in range(0, len(board), 2)}
    cards = _distinct_cards(MAX_HEROES * 4, exclude=board_cards)
    heroes, dead = [], []
    for i in range(MAX_HEROES):
        hand = "".join(cards[i * 4:(i + 1) * 4])
        heroes.append({"id": f"h{i}", "cards": hand})
        dead.append(hand)
    resp = client.post("/range_ladder", json={
        "board": board,
        "dead": dead,
        "heroes": heroes,
        "hands": 150, "rank_runouts": 5, "trials_per_bucket": 50, "seed": 1,
    })
    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()
    assert len(body["ladders"]) == MAX_HEROES


# ---------------------------------------------------------------------------
# Fix round 1 follow-up: `int(seed)` sat outside the try/except, so a
# non-numeric `seed` raised an uncaught ValueError/TypeError -- Flask turned
# that into a bare, unlogged 500 for what is plainly a caller error. Fixed
# to return 400 (distinguishable from compute_range_ladder's own ValueError,
# which this route maps to 422).
# ---------------------------------------------------------------------------

def test_non_numeric_seed_is_a_400_not_a_500_or_422():
    resp = client.post("/range_ladder", json={
        "board": "6s7s4s",
        "dead": ["AsKs9h2c"],
        "heroes": [{"id": "u1", "cards": "AsKs9h2c"}],
        "hands": 100, "rank_runouts": 5, "trials_per_bucket": 50,
        "seed": "abc",
    })
    assert resp.status_code == 400, resp.get_data(as_text=True)


def test_string_seed_still_works_and_matches_integer_seed():
    payload = {
        "board": "6s7s4s",
        "dead": ["AsKs9h2c"],
        "heroes": [{"id": "u1", "cards": "AsKs9h2c"}],
        "hands": 300, "rank_runouts": 10, "trials_per_bucket": 100,
    }
    resp_int = client.post("/range_ladder", json={**payload, "seed": 7})
    resp_str = client.post("/range_ladder", json={**payload, "seed": "7"})
    assert resp_int.status_code == 200, resp_int.get_data(as_text=True)
    assert resp_str.status_code == 200, resp_str.get_data(as_text=True)
    # Determinism must not depend on which JSON type the caller sent the
    # seed as -- same seed value in, same ladder out.
    assert resp_int.get_json() == resp_str.get_json()


# ---------------------------------------------------------------------------
# Final review, Finding 2: card-syntax errors escaped as 500s. `hand_str_to_ints`
# raises KeyError on a malformed or non-canonical card and malformed hero
# objects raise TypeError/KeyError -- neither is a ValueError, so both fell
# past the route's `except ValueError -> 422` into the bare 500 handler. Cards
# are canonical `Rank`+`suit` (uppercase rank, lowercase suit) and every real
# caller already sends them that way, so these are plain input errors: 422,
# with a message naming what was wrong.
# ---------------------------------------------------------------------------

_CARD_SYNTAX_PAYLOADS = {
    "lowercase rank": {"board": "6s7s4s", "dead": ["asks9h2c"],
                       "heroes": [{"id": "u1", "cards": "asks9h2c"}]},
    "bogus rank letter": {"board": "6s7s4s", "dead": ["ZZKs9h2c"],
                          "heroes": [{"id": "u1", "cards": "AsKs9h2c"}]},
    "bogus suit letter": {"board": "6s7s4s", "dead": ["AxKs9h2c"],
                          "heroes": [{"id": "u1", "cards": "AsKs9h2c"}]},
    "odd-length board": {"board": "6s7s4", "dead": ["AsKs9h2c"],
                         "heroes": [{"id": "u1", "cards": "AsKs9h2c"}]},
    "odd-length hero": {"board": "6s7s4s", "dead": ["AsKs9h2c"],
                        "heroes": [{"id": "u1", "cards": "AsKs9h2"}]},
    "hero missing cards": {"board": "6s7s4s", "dead": ["AsKs9h2c"],
                           "heroes": [{"id": "u1"}]},
    "hero missing id": {"board": "6s7s4s", "dead": ["AsKs9h2c"],
                        "heroes": [{"cards": "AsKs9h2c"}]},
    "hero not an object": {"board": "6s7s4s", "dead": ["AsKs9h2c"],
                           "heroes": ["AsKs9h2c"]},
    "hero cards not a string": {"board": "6s7s4s", "dead": ["AsKs9h2c"],
                                "heroes": [{"id": "u1", "cards": 42}]},
    "board not a string": {"board": 42, "dead": ["AsKs9h2c"],
                           "heroes": [{"id": "u1", "cards": "AsKs9h2c"}]},
}


@pytest.mark.parametrize("label", sorted(_CARD_SYNTAX_PAYLOADS))
def test_malformed_card_input_is_a_422_with_a_useful_message(label):
    payload = dict(_CARD_SYNTAX_PAYLOADS[label])
    payload.update(hands=100, rank_runouts=5, trials_per_bucket=50, seed=1)
    resp = client.post("/range_ladder", json=payload)
    assert resp.status_code == 422, f"{label}: {resp.get_data(as_text=True)}"
    details = resp.get_json()["details"]
    # Useful == it names the offending field or the offending value, not a
    # bare repr of some KeyError from three frames down.
    assert details, label
    assert any(t in details for t in ("board", "dead", "hero", "heroes")), (
        f"{label}: unhelpful message {details!r}"
    )


def test_non_list_heroes_is_a_400_not_a_500():
    # `heroes` is len()'d before the try block, so a non-sized value used to
    # raise TypeError outside every handler.
    resp = client.post("/range_ladder", json={
        "board": "6s7s4s", "dead": ["AsKs9h2c"], "heroes": 5,
    })
    assert resp.status_code == 400, resp.get_data(as_text=True)


def test_an_internal_failure_is_still_a_500(monkeypatch):
    # The 422 above must come from validating card syntax at the boundary,
    # NOT from broadening the route's catch. A KeyError raised from inside
    # the computation is a genuine internal fault and must still be a 500.
    def boom(**kwargs):
        raise KeyError("some internal lookup")

    monkeypatch.setattr(server, "compute_range_ladder", boom)
    resp = client.post("/range_ladder", json={
        "board": "6s7s4s",
        "dead": ["AsKs9h2c"],
        "heroes": [{"id": "u1", "cards": "AsKs9h2c"}],
        "hands": 100, "rank_runouts": 5, "trials_per_bucket": 50, "seed": 1,
    })
    assert resp.status_code == 500, resp.get_data(as_text=True)


def test_a_valid_request_still_succeeds_after_syntax_validation():
    # Guard against over-eager validation: canonical cards, an 8-card `dead`
    # entry and a 5-card board must all still be accepted.
    resp = client.post("/range_ladder", json={
        "board": "6s7s4s2h9d",
        "dead": ["AsKs9h2c", "3dTc5s8h"],
        "heroes": [{"id": "u1", "cards": "AsKs9h2c"}],
        "hands": 100, "rank_runouts": 5, "trials_per_bucket": 50, "seed": 1,
    })
    assert resp.status_code == 200, resp.get_data(as_text=True)


# ---------------------------------------------------------------------------
# Final review, Finding 3: only the LENGTH of `buckets` was checked, so the
# values passed straight through -- [0] and [-5] returned 200 with a rung
# labelled 0 or -5 whose equity came from a single hand, [500] labelled the
# whole population 500%, and ["x"] / "abc" were 500s. A bucket is a
# percentile: it must be a number in (0, 100], rejected with 400 otherwise
# (consistent with the MAX_BUCKETS/MAX_HEROES rejections).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("buckets", [
    [0],                  # selects no hands; engine's max(1, ...) made it one hand
    [-5],                 # negative percentile
    [500],                # >100%: the whole population, mislabelled
    [100.5],              # just over the top
    ["x"],                # not a number at all
    [5, 0, 100],          # one bad value among good ones
    [None],
    [True],               # bool is an int in Python; not a percentile
    "abc",                # a bare string: passes len(), tuple()s to ['a','b','c']
    5,                    # not a sequence at all
])
def test_invalid_bucket_values_are_a_400(buckets):
    resp = client.post("/range_ladder", json={
        "board": "6s7s4s",
        "dead": ["AsKs9h2c"],
        "heroes": [{"id": "u1", "cards": "AsKs9h2c"}],
        "buckets": buckets,
        "hands": 100, "rank_runouts": 5, "trials_per_bucket": 50, "seed": 1,
    })
    assert resp.status_code == 400, resp.get_data(as_text=True)
    assert "buckets" in resp.get_json()["details"]


def test_valid_bucket_values_are_accepted_including_duplicates_and_descending():
    # Duplicates and non-ascending order are ALLOWED on purpose: buckets are
    # independent cumulative top-pct slices computed from `strength` alone,
    # so order carries no meaning and a duplicate just asks the same
    # question twice. The response echoes the caller's order, duplicates
    # included.
    resp = client.post("/range_ladder", json={
        "board": "6s7s4s",
        "dead": ["AsKs9h2c"],
        "heroes": [{"id": "u1", "cards": "AsKs9h2c"}],
        "buckets": [100, 5, 5, 12.5, 0.5],
        "hands": 100, "rank_runouts": 5, "trials_per_bucket": 50, "seed": 1,
    })
    assert resp.status_code == 200, resp.get_data(as_text=True)
    rungs = resp.get_json()["ladders"][0]["rungs"]
    assert [r["bucket"] for r in rungs] == [100, 5, 5, 12.5, 0.5]
    # The two identical 5% rungs must describe the same slice, not two
    # different ones -- same edge hand, and (the river/flop equity is
    # sampled per block, so equity may differ) the same boundary.
    assert rungs[1]["edge"] == rungs[2]["edge"]
