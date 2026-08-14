# Slim, public-launch image for the PQL engine (server.py).
#
# Data files COPYed below are the MINIMAL set the engine loads at runtime
# (verified empirically by tracing builtins.open / np.load through run_pql + the
# graph functions). Each, and why:
#   score_array.npy           numba 5-card lookup (optimized_evaluator.get_score_array)
#   category_array.npy        numba made-hand category lookup (optimized_evaluator)
#   holdem_class_order.json   169-class order for holdem percentile ranges (pql/ranges/percentile.py)
#   plo4_class_order.json     PLO4 percentile-range class order
#   plo5_class_order.json     PLO5 percentile-range class order
#   pql/parser/grammar.lark   PQL grammar (lives inside pql/, copied with it)
# NOT copied (not on the engine path): handrank_cdf_*.npy, category_dict.pkl,
# score_dict.pkl (legacy evaluator), plo6_class_order.json (not generated; PLO6
# percentile ranges raise a clean NotImplementedError, explicit PLO6 ranges work).
#
# Support .py modules the engine imports from repo root (verified via sys.modules):
#   card_encoding.py, hand_indexing.py, hand_categories.py, optimized_evaluator.py
FROM python:3.11-slim

WORKDIR /app

COPY requirements-engine.txt .
RUN pip install --no-cache-dir -r requirements-engine.txt

# Engine package + slim server
COPY pql/ ./pql/
COPY server.py .

# Root support modules imported by pql
COPY card_encoding.py hand_indexing.py hand_categories.py optimized_evaluator.py ./

# Root modules on the /range_ladder path. This list is NOT hand-curated -- it is
# the transitive root-module closure of `import server`, and there is a test
# (tests/test_docker_image_completeness.py) that recomputes that closure and
# fails if anything here is missing. That test exists because the omission it
# guards took the whole engine down once: server.py grew `from range_ladder
# import ...` at module scope while this COPY list stayed minimal, so the image
# built fine, then died on import at container start -- taking /pql, a live
# user-facing endpoint, with it. A missing file here is not a degraded feature;
# it is a 502 on everything.
COPY range_ladder.py fast_score.py hand_rank_evaluator.py process_pool.py ./

# Runtime data files (see header)
COPY score_array.npy category_array.npy \
     holdem_class_order.json plo4_class_order.json plo5_class_order.json ./

ENV PORT=8080
# Keep memory bounded on a small (e.g. 512MB) host: single numba thread, an
# on-disk JIT cache (so restarts re-use compiled code), and limited waitress
# concurrency so simultaneous Monte-Carlo requests can't pile up allocations.
ENV NUMBA_NUM_THREADS=1
ENV NUMBA_CACHE_DIR=/tmp/numba-cache
# /range_ladder's process pool. MUST be set here: range_ladder defaults to
# os.cpu_count(), which inside a container reports the HOST's core count and
# not the cgroup limit -- 24 workers were observed on a box entitled to a
# fraction of one. Each worker is a separate interpreter holding numpy, numba
# and the 21MB score_array, so an unset value is an OOM kill. Matches the
# --threads=2 below: at most two requests are in flight, so more pools than
# that buys nothing.
ENV RANGE_LADDER_POOL_WORKERS=2
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
EXPOSE 8080
CMD ["sh", "-c", "waitress-serve --host=0.0.0.0 --port=${PORT} --threads=2 --connection-limit=25 --channel-timeout=120 server:app"]
