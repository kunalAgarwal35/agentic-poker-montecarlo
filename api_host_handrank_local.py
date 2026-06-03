"""
Lightweight local Flask harness for the /handrank endpoint only.

This is intentionally standalone - it does NOT pull in multi_queries, numba,
matplotlib, JVM, or AWS clients - so it can boot on any dev box (including
Windows ARM64) for local testing and stress testing of the Hand Rank API.

For production, /handrank is exposed from the full api_host.py alongside all
the other endpoints. That file is NOT this one.

Run:
    python api_host_handrank_local.py
    # listens on http://0.0.0.0:5050/handrank
"""

import traceback
from flask import Flask, request, jsonify

from hand_rank_evaluator import (
    hand_rank_batch,
    warmup_handrank,
    get_handrank_cdf,
)


app = Flask(__name__)


_warmup_started = False
_warmup_complete = False


def _background_warmup() -> None:
    global _warmup_complete
    try:
        warmup_handrank()
    except Exception as e:  # noqa: BLE001
        print(f"[Warmup] HandRank failed: {e}")
    _warmup_complete = True


@app.route('/health', methods=['GET'])
def health_check():
    global _warmup_started
    import threading
    if not _warmup_started:
        _warmup_started = True
        threading.Thread(target=_background_warmup, daemon=True).start()
        return "Application is running - Warmup started in background", 200
    if _warmup_complete:
        return "Application is running - All systems ready", 200
    return "Application is running - Warmup in progress", 200


@app.route('/handrank', methods=['POST'])
def hand_rank_endpoint():
    data = request.get_json(silent=True) or {}
    cards = data.get('cards')
    if not cards:
        return jsonify({"error": "Missing required field: cards"}), 400

    if isinstance(cards, str):
        hands = [cards]
    elif isinstance(cards, list):
        if len(cards) == 0:
            return jsonify({"error": "cards must be non-empty"}), 400
        if all(isinstance(c, str) and len(c) == 2 for c in cards):
            hands = [cards]
        else:
            hands = cards
    else:
        return jsonify({"error": "cards must be a string or an array"}), 400

    dead_cards = data.get('dead_cards') or data.get('deadcards') or []
    board = data.get('board') or []
    num_trials = int(data.get('trials') or 5000)
    num_opponents = int(data.get('num_opponents') or 2)
    seed = data.get('seed')
    if seed is not None:
        seed = int(seed)

    if num_trials < 100 or num_trials > 100_000:
        return jsonify({"error": "trials must be between 100 and 100000"}), 400
    if num_opponents < 1 or num_opponents > 5:
        return jsonify({"error": "num_opponents must be between 1 and 5"}), 400

    try:
        result = hand_rank_batch(
            hands,
            dead_cards=dead_cards,
            board=board,
            num_opponents=num_opponents,
            num_trials=num_trials,
            seed=seed,
        )
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:  # noqa: BLE001
        app.logger.error(f"Error in /handrank: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

    if not result.get('cdf_available'):
        result['warning'] = (
            f"Baseline CDF for {result['game_type']} not loaded - returning equity only "
            f"(rank is null). Run generate_handrank_cdf.py to produce it."
        )
    return jsonify(result)


@app.route('/')
def home():
    return "HandRank local harness. POST /handrank to compute ranks."


if __name__ == '__main__':
    import socket
    print(socket.gethostbyname(socket.gethostname()))
    # Threaded=True so concurrent requests actually run on separate threads
    # for realistic stress-test behavior.
    app.run(host='0.0.0.0', port=5050, threaded=True)
