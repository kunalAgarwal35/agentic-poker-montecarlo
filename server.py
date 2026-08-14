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
from range_ladder import (
    compute_range_ladder,
    warmup_pool,
    DEFAULT_BUCKETS,
    DEFAULT_HANDS,
    DEFAULT_RANK_RUNOUTS,
    DEFAULT_TRIALS_PER_BUCKET,
)

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


def _clamp_hands(raw):
    """Clamp the Pass-1 villain population into [100, DEFAULT_HANDS].

    Ceiling is the shipped default (60,000) itself, not some multiple of it:
    range_ladder.py's own DEFAULT_HANDS comment is an extensively measured
    argument that 60,000 is already the deliberately chosen top of this
    box's ~5-6s budget (a further ~2.6-3x, to ~155,000, is an untested
    extrapolation to ~14s) -- letting a caller dial past that would blow
    past a tradeoff the user made on purpose, not just spike load. A
    caller may always ask for FEWER hands (a faster, noisier read);
    asking for more than the tuned default is not something a per-request
    override should be able to grant.

    Absent/None (key omitted, or explicit `null`) is treated as "use the
    default" -- but a literal `0` (or `"0"`) is a caller-supplied value and
    must clamp DOWN to the floor, not silently jump to the ceiling. `raw or
    DEFAULT_HANDS` got this backwards (0 is falsy, so it fell through to
    the default -- the single MOST expensive value, for a caller who
    plainly asked for the least) and disagreed with itself between int 0
    and string "0" (only the int short-circuited); checking `is None`
    explicitly fixes both."""
    if raw is None:
        return DEFAULT_HANDS
    return max(100, min(int(raw), DEFAULT_HANDS))


def _clamp_rank_runouts(raw):
    """Clamp Pass-1's shared-runout count into [5, DEFAULT_RANK_RUNOUTS].

    Same reasoning as `_clamp_hands`: Pass-1 cost is hands x rank_runouts,
    so this is just as much a cost lever as `hands` is, even though it
    wasn't named explicitly in the brief. Ceiling is the shipped default
    (300) -- range_ladder.py's DEFAULT_RANK_RUNOUTS comment records that
    300 was already chosen as the point past which spending more of the
    time budget here stopped being worth it relative to spending it on
    `hands` instead.

    Same absent-vs-zero fix as `_clamp_hands`: only `None` (key omitted or
    explicit `null`) falls back to the default; a literal 0 clamps to the
    floor instead."""
    if raw is None:
        return DEFAULT_RANK_RUNOUTS
    return max(5, min(int(raw), DEFAULT_RANK_RUNOUTS))


def _clamp_trials_per_bucket(raw):
    """Clamp Pass-2's per-bucket trial count into [50, DEFAULT_TRIALS_PER_BUCKET].

    Ceiling is the shipped default (50,000): Pass-2 cost is
    trials_per_bucket x len(buckets) x (1 + len(heroes)), and 50,000 was
    the value range_ladder.py's DEFAULT_TRIALS_PER_BUCKET comment measured
    as already inside the ~5-6s budget with real headroom on a loaded
    machine -- the same "the shipped default is the tuned ceiling, not a
    suggestion" reasoning as `_clamp_hands`.

    Same absent-vs-zero fix as `_clamp_hands`: only `None` (key omitted or
    explicit `null`) falls back to the default; a literal 0 clamps to the
    floor instead."""
    if raw is None:
        return DEFAULT_TRIALS_PER_BUCKET
    return max(50, min(int(raw), DEFAULT_TRIALS_PER_BUCKET))


# Pass-2 cost is trials_per_bucket x len(buckets) x (1 + len(heroes)) -- unlike
# hands/rank_runouts/trials_per_bucket above, heroes and buckets are rejected
# outright rather than silently clamped: quietly dropping a caller's 11th hero
# or 13th bucket would return an answer to a question they didn't ask, where a
# quietly-smaller `hands` is still an honest (if noisier) answer to the same
# question.
MAX_HEROES = 10
MAX_BUCKETS = 12

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

    try:
        # Run the full range-ladder path once on a tiny input so the first
        # real /range_ladder request doesn't pay it cold. warmup_pool()
        # above only warms fast_score's batch-scoring kernel (shared with
        # /pql) in every pool worker; it does NOT touch
        # best5_category_omaha_numba (a separate @njit function) or
        # category_array.npy, both used only by range_ladder.describe_category
        # to label each rung's edge hand. hands/rank_runouts/trials_per_bucket
        # are kept tiny on purpose -- this only needs to exercise every code
        # path once for JIT/cache purposes, not produce a meaningful ladder.
        compute_range_ladder(
            board="6s7s4s",
            dead=["AsKs9h2c"],
            heroes=[{"id": "warmup", "cards": "AsKs9h2c"}],
            hands=200, rank_runouts=10, trials_per_bucket=50, seed=0,
        )
    except Exception as e:  # never let warmup crash the process
        app.logger.warning(f"[Warmup] range ladder warmup failed: {e}")


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


@app.route('/range_ladder', methods=['POST'])
def range_ladder_endpoint():
    """Postflop range ladder: hero equity vs board-strength percentile slices
    of the villain population.

    Request body: { board, heroes, dead?, buckets?, seed?,
                     hands?, rank_runouts?, trials_per_bucket? }
    Response: compute_range_ladder's dict, unchanged --
      { population, exact, rank_runouts, trials_per_bucket, ladders }

    NOTE: task-9-brief.md predates two redesigns of range_ladder.py and is
    stale on this route's exact shape -- there is no `runouts` request
    param and no `hands` response field; the real knobs are `hands`,
    `rank_runouts` and `trials_per_bucket` (see compute_range_ladder's
    signature). `parallel` / `num_workers` are deliberately NOT exposed
    here: they size this box's worker pool, a deployment concern, not
    something a caller should tune per-request.

    This endpoint is far more expensive than /pql (roughly 5-6s at the
    shipped defaults, scaling with the number of heroes requested), so
    unlike /pql there is no artificial trials-style ceiling meant to keep
    every request cheap -- only a clamp on hands/rank_runouts/
    trials_per_bucket so a caller can't ask for MORE than the
    already-tuned default (see the _clamp_* helpers above). There is no
    timeout here: a slow response is the honest cost of this computation,
    not something to truncate into a wrong (partial) answer.

    `heroes` (1..MAX_HEROES) and `buckets` (1..MAX_BUCKETS, when provided)
    are REJECTED (400) rather than clamped when out of range -- Pass-2
    cost is trials_per_bucket x len(buckets) x (1 + len(heroes)), so both
    multiply cost the same way hands/rank_runouts/trials_per_bucket do,
    but silently dropping a caller's 11th hero or 13th bucket would answer
    a different, smaller question than the one asked instead of just
    answering it more cheaply/noisily.
    """
    if not _engine_authorized():
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(silent=True) or {}

    board = data.get('board')
    heroes = data.get('heroes')
    if not board or not heroes:
        return jsonify({"error": "Missing required field: board and heroes"}), 400
    if len(heroes) > MAX_HEROES:
        return jsonify({
            "error": "Too many heroes",
            "details": f"at most {MAX_HEROES} heroes per request, got {len(heroes)}",
        }), 400

    buckets_raw = data.get('buckets')
    if buckets_raw is None:
        buckets = tuple(DEFAULT_BUCKETS)
    elif len(buckets_raw) == 0:
        return jsonify({"error": "`buckets` must not be empty"}), 400
    elif len(buckets_raw) > MAX_BUCKETS:
        return jsonify({
            "error": "Too many buckets",
            "details": f"at most {MAX_BUCKETS} buckets per request, got {len(buckets_raw)}",
        }), 400
    else:
        buckets = tuple(buckets_raw)

    seed = data.get('seed')
    if seed is not None:
        # Coerced -- and rejected -- OUTSIDE the try/except below on purpose:
        # compute_range_ladder itself raises ValueError for bad card/board
        # input, which that block maps to 422. A non-numeric seed is a
        # different kind of caller error (a malformed field, not input the
        # computation examined and rejected) and must stay a distinguishable
        # 400, not get relabeled 422 "Invalid input" alongside genuine
        # card-validation failures.
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            return jsonify({
                "error": "Invalid input",
                "details": f"`seed` must be an integer, got {seed!r}",
            }), 400

    try:
        result = compute_range_ladder(
            board=board,
            dead=data.get('dead') or [],
            heroes=heroes,
            buckets=buckets,
            hands=_clamp_hands(data.get('hands')),
            rank_runouts=_clamp_rank_runouts(data.get('rank_runouts')),
            trials_per_bucket=_clamp_trials_per_bucket(data.get('trials_per_bucket')),
            seed=seed,
        )
    except ValueError as ve:
        return jsonify({"error": "Invalid input", "details": str(ve)}), 422
    except Exception as e:
        app.logger.error(f"Error in /range_ladder: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5050)))
