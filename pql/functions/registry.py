from __future__ import annotations
from typing import Callable, Dict, List
import numpy as np

# A value function: (ctx, player_idx, args) -> column (N,)
# ctx is an EvalContext; args are the value-call's extra Ident names
# (e.g. ['river', 'flush'] for exactHandType).
ValueFn = Callable[[object, int, List[str]], np.ndarray]

_VALUE_FNS: Dict[str, ValueFn] = {}


def register_value(name: str):
    def deco(fn: ValueFn) -> ValueFn:
        _VALUE_FNS[name.lower()] = fn
        return fn
    return deco


def get_value_fn(name: str) -> ValueFn:
    key = name.lower()
    if key not in _VALUE_FNS:
        raise ValueError(f"Unknown PQL value function '{name}'")
    return _VALUE_FNS[key]


def value_fn_names():
    return sorted(_VALUE_FNS)
