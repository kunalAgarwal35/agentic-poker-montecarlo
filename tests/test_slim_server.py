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
