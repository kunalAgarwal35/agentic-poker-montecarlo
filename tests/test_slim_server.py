"""Smoke tests for the slim deployable engine server (server.py).

Exercises the public contract: /health, /pql, /pql-graph. The first /pql call
triggers numba JIT compilation, so the suite may take ~30-90s on a cold run.
"""
from server import app

client = app.test_client()


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200


def test_pql_where():
    resp = client.post("/pql", json={
        "query": "select avg(riverEquity(PLAYER_1)) as e from game='holdem', "
                 "board='Ah7c2h', PLAYER_1='KhQh', PLAYER_2='QdQc' "
                 "where minHandType(PLAYER_1,river,flush)",
        "trials": 2000,
        "seed": 7,
    })
    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()
    assert 0.95 <= body["values"]["e"] <= 1.0, body
    assert body["trials"] > 0
    assert "mode" in body


def test_pql_histogram():
    resp = client.post("/pql", json={
        "query": "select histogram(handType(PLAYER_1,river)) as dist from "
                 "game='holdem', board='Ah7c2h', PLAYER_1='KhQh', PLAYER_2='QdQc'",
    })
    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()
    pairs = body["histograms"]["dist"]["pairs"]
    assert isinstance(pairs, list) and len(pairs) > 0, body


def test_graph_street():
    resp = client.post("/pql-graph", json={
        "game": "holdem",
        "board": "Ah7c2h",
        "players": {"PLAYER_1": "KhQh", "PLAYER_2": "QdQc"},
        "kind": "street",
        "hero": "PLAYER_1",
        "trials": 1500,
    })
    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()
    assert "series" in body, body


def test_bad_query_400():
    resp = client.post("/pql", json={
        "query": "select avg(bogusFn(PLAYER_1)) as x from game='holdem', "
                 "PLAYER_1='AsKs', PLAYER_2='QdQc'",
    })
    assert resp.status_code == 400, resp.get_data(as_text=True)


_VALID_QUERY = (
    "select avg(riverEquity(PLAYER_1)) as e from game='holdem', "
    "PLAYER_1='AsKs', PLAYER_2='QdQc'"
)


def test_pql_requires_engine_key_when_set(monkeypatch):
    """With ENGINE_KEY set, a /pql POST without the X-Engine-Key header is 401."""
    monkeypatch.setenv("ENGINE_KEY", "testkey")
    resp = client.post("/pql", json={"query": _VALID_QUERY, "trials": 200})
    assert resp.status_code == 401, resp.get_data(as_text=True)


def test_pql_accepts_correct_engine_key(monkeypatch):
    """With ENGINE_KEY set and the matching header, /pql returns 200."""
    monkeypatch.setenv("ENGINE_KEY", "testkey")
    resp = client.post(
        "/pql",
        json={"query": _VALID_QUERY, "trials": 200, "seed": 1},
        headers={"X-Engine-Key": "testkey"},
    )
    assert resp.status_code == 200, resp.get_data(as_text=True)


def test_health_not_protected(monkeypatch):
    """/health stays open even when ENGINE_KEY is set."""
    monkeypatch.setenv("ENGINE_KEY", "testkey")
    resp = client.get("/health")
    assert resp.status_code == 200


def test_graph_requires_engine_key_when_set(monkeypatch):
    monkeypatch.setenv("ENGINE_KEY", "testkey")
    resp = client.post("/pql-graph", json={
        "game": "holdem",
        "players": {"PLAYER_1": "KhQh", "PLAYER_2": "QdQc"},
        "kind": "street",
    })
    assert resp.status_code == 401, resp.get_data(as_text=True)


def test_pql_clamps_huge_trials(monkeypatch):
    """A wildly large trials value is clamped (capped at 100k), so the request
    completes quickly with 200 instead of hanging on a billion-trial compute."""
    monkeypatch.delenv("ENGINE_KEY", raising=False)
    resp = client.post("/pql", json={
        "query": _VALID_QUERY,
        "trials": 99999999,
        "seed": 1,
    })
    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()
    # Clamp ceiling is 100000; the engine may run enumeration (reporting its own
    # combo count) but never the requested 99,999,999 trials.
    assert body["trials"] <= 100000, body
