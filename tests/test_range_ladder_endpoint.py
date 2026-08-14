"""Task 9: POST /range_ladder -- the Flask endpoint over range_ladder.compute_range_ladder.

Payload sizes here are deliberately tiny (small hands/rank_runouts/
trials_per_bucket) so the suite stays fast; the shipped defaults are ~5-6s
per call and are exercised by range_ladder.py's own test suite, not here.
"""
from server import app, DEFAULT_HANDS

client = app.test_client()


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
