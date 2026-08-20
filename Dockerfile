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
# One numba thread per worker and an on-disk JIT cache (so restarts re-use
# compiled code). Parallelism lives at the PROCESS level (the pool below);
# threads inside a worker on top of that would only oversubscribe the box.
ENV NUMBA_NUM_THREADS=1
ENV NUMBA_CACHE_DIR=/tmp/numba-cache
# /range_ladder's process pool. MUST be set here: range_ladder defaults to
# os.cpu_count(), which inside a container reports the HOST's core count and
# not the cgroup limit -- 24 workers were observed on a box entitled to a
# fraction of one. Each worker is a separate interpreter holding numpy, numba
# and the 21MB score_array, so an unset value is an OOM kill.
#
# Sized to the replica's actual allocation: 8 vCPU / 8GB. At ~300MB per worker
# that peaks near 2.4GB, comfortably inside 8GB.
#
# It was 2, on the stated grounds that it should "match the --threads=2 below:
# at most two requests are in flight, so more pools than that buys nothing".
# That conflates two independent knobs. --threads is how many REQUESTS waitress
# handles at once; this is how many cores ONE request may fan out across, since
# rank_hands splits its runouts into `workers` chunks. Matching them pinned the
# engine to 2 of 8 available vCPU and made Pass 1 -- ~65% of an off-river
# request, and fully parallel -- four times longer than it needed to be.
#
# It also made the two settings fight: the pool is a single global executor
# shared by every request, so two concurrent requests contended for the same
# two workers rather than using idle cores.
ENV RANGE_LADDER_POOL_WORKERS=8
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
EXPOSE 8080
CMD ["sh", "-c", "waitress-serve --host=0.0.0.0 --port=${PORT} --threads=2 --connection-limit=25 --channel-timeout=120 server:app"]
