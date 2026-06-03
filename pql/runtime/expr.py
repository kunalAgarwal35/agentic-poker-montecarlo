"""Per-trial expression evaluator: walks an expression AST to an (N,) float column,
reusing the engine's scalar≡vector-per-trial invariant. Value-function calls keep
their token args (player + street/category/int tokens); operators are vectorized."""
from __future__ import annotations
import numpy as np

from pql.parser.ast import Call, Ident, IntLit, BinOp, Compare, Logical, Not
from pql.functions.registry import get_value_fn

# Aggregate names are NOT valid inside an expression (no nested aggregates).
_AGG_NAMES = {"avg", "count", "sum", "min", "max", "histogram"}


def _player_index(scenario, ident_name: str) -> int:
    for i, p in enumerate(scenario.players):
        if p.name.lower() == ident_name.lower():
            return i
    raise ValueError(f"Query references unknown player '{ident_name}'")


def eval_value_call(call: Call, ctx, scenario) -> np.ndarray:
    """Evaluate a single value-function call to an (N,) column."""
    if call.name.lower() in _AGG_NAMES:
        raise ValueError(f"Aggregate '{call.name}' cannot be nested inside an expression")
    if not call.args or not isinstance(call.args[0], Ident):
        raise TypeError(f"Value function '{call.name}' needs a player identifier as its first argument")
    pidx = _player_index(scenario, call.args[0].name)
    token_args = [a.name for a in call.args[1:] if isinstance(a, Ident)]
    return get_value_fn(call.name)(ctx, pidx, token_args)


def eval_expr(node, ctx, scenario) -> np.ndarray:
    """Evaluate an expression node to an (N,) float64 column."""
    n = ctx.boards.shape[0]
    if isinstance(node, Call):
        return np.asarray(eval_value_call(node, ctx, scenario), dtype=np.float64)
    if isinstance(node, IntLit):
        return np.full(n, float(node.value), dtype=np.float64)
    if isinstance(node, BinOp):
        l = eval_expr(node.left, ctx, scenario)
        r = eval_expr(node.right, ctx, scenario)
        if node.op == "+": return l + r
        if node.op == "-": return l - r
        if node.op == "*": return l * r
        if node.op == "/": return l / r
        raise ValueError(f"Unknown arithmetic operator '{node.op}'")
    if isinstance(node, Compare):
        l = eval_expr(node.left, ctx, scenario)
        r = eval_expr(node.right, ctx, scenario)
        ops = {">": l > r, "<": l < r, ">=": l >= r, "<=": l <= r,
               "=": l == r, "!=": l != r}
        if node.op not in ops:
            raise ValueError(f"Unknown comparison operator '{node.op}'")
        return ops[node.op].astype(np.float64)
    if isinstance(node, Logical):
        l = eval_expr(node.left, ctx, scenario) != 0
        r = eval_expr(node.right, ctx, scenario) != 0
        combined = np.logical_and(l, r) if node.op == "and" else np.logical_or(l, r)
        return combined.astype(np.float64)
    if isinstance(node, Not):
        return (eval_expr(node.operand, ctx, scenario) == 0).astype(np.float64)
    if isinstance(node, Ident):
        raise TypeError(f"Bare identifier '{node.name}' is not a valid expression "
                        f"(expected a value-function call, number, or operator expression)")
    raise TypeError(f"Cannot evaluate expression node of type {type(node).__name__}")
