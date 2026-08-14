"""Slim, deployable PQL engine server.

Serves ONLY the three endpoints the public web app uses:
  - GET  /health     -> "ok" (kicks off a one-time background numba warmup)
  - POST /pql        -> run a PQL query, return asdict(result)
  - POST /pql-graph  -> equity graph (street / distribution / vsclass)

Runs purely on `pql/` + numba. NO boto3/DynamoDB, NO JVM, NO licensed jars,
NO config.ini. The request/response contract mirrors api_host.py exactly so the
web app's web/lib/engine.ts works unchanged.
"""
from flask import Flask, request, jsonify
from dataclasses import asdict
import os
import threading
import traceback
import logging

import fast_score
from pql import run_pql
from pql.graphs import equity_by_street, equity_distribution, equity_vs_class
from pql.scenario import build_scenario
from pql.parser.ast import Query
from range_ladder import warmup_pool

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)


def _engine_authorized():
    """Shared-secret gate. If ENGINE_KEY is set in the env, the request must send
    a matching X-Engine-Key header. If ENGINE_KEY is unset (local dev), allow.
    Read per-request so tests can set the env without re-importing the app."""
    expected = os.environ.get('ENGINE_KEY')
    if not expected:
        return True
    return request.headers.get('X-Engine-Key') == expected


def _clamp_trials(raw):
    """Clamp trials into [100, 30000] so a caller can't request a huge compute that
    spikes memory (the engine runs on a small box). Defaults to 20000 when missing."""
    return max(100, min(int(raw or 20000), 30000))

# One-time numba JIT warmup, kicked off by the first /health hit. Guarded by a
# module-level bool so we only ever start the daemon thread once.
_warmup_started = False


def _warmup():
    """Trigger numba JIT compilation in the background so the first real /pql or
    /pql-graph request isn't paying the (~30-90s) compile cost. Also warms the
    persistent process pool (Task 10) that range_ladder's rank_hands (Pass 1)
    and evaluate_pass2 (Pass 2) parallelise onto, so the first range-ladder
    request doesn't pay the ~5s-per-worker process-creation cost that the
    pool exists to avoid.

    Task 12: also warms fast_score's numba scoring kernel directly, in THIS
    process, as its own try/except -- independent of warmup_pool() below
    (which also warms it, inside every pool worker) so a pool-warmup failure
    doesn't silently skip warming the serial/in-process scoring path too."""
    try:
        run_pql(
            "select avg(riverEquity(PLAYER_1)) as e from game='holdem', "
            "PLAYER_1='AsKs', PLAYER_2='QdQc'",
            trials=200,
        )
    except Exception as e:  # never let warmup crash the process
        app.logger.warning(f"[Warmup] failed: {e}")

    try:
        fast_score.warmup()
    except Exception as e:  # never let warmup crash the process
        app.logger.warning(f"[Warmup] fast_score kernel warmup failed: {e}")

    try:
        warmup_pool()
    except Exception as e:  # never let warmup crash the process
        app.logger.warning(f"[Warmup] process pool warmup failed: {e}")


@app.route('/health', methods=['GET'])
def health_check():
    """Returns immediately; first call starts a daemon warmup thread."""
    global _warmup_started
    if not _warmup_started:
        _warmup_started = True
        threading.Thread(target=_warmup, daemon=True).start()
    return "ok", 200


@app.route('/pql', methods=['POST'])
def pql_endpoint():
    """Native PQL query endpoint.

    Request body: { "query": "...", "trials": 20000, "seed": null }
    Response:     asdict(result) -> { values, columns, trials, mode, seed, ... }
    """
    if not _engine_authorized():
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    query = data.get('query')
    if not query:
        return jsonify({"error": "Missing required field: query"}), 400

    trials = _clamp_trials(data.get('trials'))
    seed = data.get('seed')
    if seed is not None:
        seed = int(seed)

    try:
        result = run_pql(query, trials=trials, seed=seed)
    except NotImplementedError as nie:
        return jsonify({"error": "Unsupported PQL feature", "details": str(nie)}), 400
    except (ValueError, KeyError) as ve:
        return jsonify({"error": "Bad query", "details": str(ve)}), 400
    except Exception as e:
        app.logger.error(f"Error in /pql: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

    return jsonify(asdict(result))


def _graph_scenario(data):
    params = {"game": data.get("game")}
    if data.get("board"):
        params["board"] = data["board"]
    if data.get("dead"):
        params["dead"] = data["dead"]
    players = data.get("players") or {}
    for name, val in players.items():
        params[name.lower()] = val
    return build_scenario(Query(select=[], params=params))


def _resolve_hero(scenario, hero_name):
    if not hero_name:
        return 0
    for i, p in enumerate(scenario.players):
        if p.name.lower() == str(hero_name).lower():
            return i
    raise ValueError(f"Graph references unknown hero '{hero_name}'")


@app.route('/pql-graph', methods=['POST'])
def pql_graph_endpoint():
    """Equity graph endpoint. Body:
      { game, board?, dead?, hero?, players:{PLAYER_1:'<cards|range>', ...},
        kind:'street'|'distribution'|'vsclass', trials?, seed? }
    Returns { kind, <payload>, trials, mode, seed }."""
    if not _engine_authorized():
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    kind = data.get("kind")
    trials = _clamp_trials(data.get("trials"))
    seed = data.get("seed")
    if seed is not None:
        seed = int(seed)
    try:
        scenario = _graph_scenario(data)
        hero_idx = _resolve_hero(scenario, data.get("hero"))
        if kind == "street":
            payload = {"series": equity_by_street(scenario, trials, seed)}
        elif kind == "distribution":
            payload = equity_distribution(scenario, hero_idx, trials, seed)
        elif kind == "vsclass":
            payload = equity_vs_class(scenario, hero_idx, trials, seed)
        else:
            return jsonify({"error": "Bad query", "details": f"unknown graph kind '{kind}'"}), 400
    except (ValueError, KeyError, NotImplementedError) as ve:
        return jsonify({"error": "Bad query", "details": str(ve)}), 400
    except Exception as e:
        app.logger.error(f"Error in /pql-graph: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500
    return jsonify({"kind": kind, **payload, "trials": trials, "mode": "monte_carlo", "seed": seed})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5050)))
