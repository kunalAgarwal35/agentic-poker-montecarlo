from __future__ import annotations
from typing import Callable, Dict, Optional
import numpy as np

AggFn = Callable[[np.ndarray], Optional[float]]


def _avg(col):
    return None if col.size == 0 else float(np.mean(col))


def _min(col):
    return None if col.size == 0 else float(np.min(col))


def _max(col):
    return None if col.size == 0 else float(np.max(col))


_AGGS: Dict[str, AggFn] = {
    "avg": _avg,
    "count": lambda col: float(np.count_nonzero(col)),
    "sum": lambda col: float(np.sum(col)),
    "min": _min,
    "max": _max,
}


def get_aggregate(name: str) -> AggFn:
    key = name.lower()
    if key not in _AGGS:
        raise ValueError(f"Unknown PQL aggregate '{name}'")
    return _AGGS[key]


def is_scalar_aggregate(name: str) -> bool:
    return name.lower() in _AGGS
