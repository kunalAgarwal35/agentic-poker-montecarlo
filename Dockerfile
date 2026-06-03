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

# Runtime data files (see header)
COPY score_array.npy category_array.npy \
     holdem_class_order.json plo4_class_order.json plo5_class_order.json ./

ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "waitress-serve --host=0.0.0.0 --port=${PORT} server:app"]
