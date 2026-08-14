"""Task 9: POST /range_ladder -- the Flask endpoint over range_ladder.compute_range_ladder.

Payload sizes here are deliberately tiny (small hands/rank_runouts/
trials_per_bucket) so the suite stays fast; the shipped defaults are ~5-6s
per call and are exercised by range_ladder.py's own test suite, not here.
"""
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
    # River board (5 cards): cheap and gives exact:True to check too.
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
    assert body["exact"] is True
    assert body["population"] > 0
    assert len(body["ladders"]) == 2
    for ladder in body["ladders"]:
        rungs = ladder["rungs"]
        assert [r["bucket"] for r in rungs] == [100, 5, 50]
        for rung in rungs:
            assert 0.0 <= rung["equity"] <= 1.0
            assert "cards" in rung["edge"] and "category" in rung["edge"]


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
