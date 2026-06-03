from __future__ import annotations
import numpy as np

from pql.parser.ast import Query, Call, Ident
from pql.parser.parse import parse
from pql.scenario import build_scenario, Scenario
from pql.runtime.sampler import complete_boards, sample_trials
from pql.runtime.enumerate import enumerate_boards
from pql.runtime.evaluator import player_scores
from pql.runtime.context import EvalContext
from pql.functions import values as _values  # noqa: F401  (registers value fns)
from pql.functions.aggregates import get_aggregate, is_scalar_aggregate
from pql.runtime.expr import eval_expr
from pql.result import PQLResult
from pql.parser.ast import Call as _Call
from hand_categories import CATEGORY_NAMES

# value functions whose column is a category index (0-8) -> histogram gets category labels
_CATEGORY_VALUE_FNS = {"handtype", "exacthandtype", "minhandtype", "maxhandtype", "winninghandtype"}


def _histogram(column, expr_node) -> dict:
    """Distribution of an integer-valued column as sorted {value,count} pairs, with
    category labels when the histogram's inner expression is a single category value-fn."""
    if column.size == 0:
        return {"pairs": [], "labels": None}
    vals, counts = np.unique(np.rint(column).astype(int), return_counts=True)
    pairs = [{"value": int(v), "count": int(c)} for v, c in zip(vals, counts)]
    labels = None
    if isinstance(expr_node, _Call) and expr_node.name.lower() in _CATEGORY_VALUE_FNS:
        labels = {str(i): CATEGORY_NAMES[i] for i in range(len(CATEGORY_NAMES))}
    return {"pairs": pairs, "labels": labels}


def _player_index(scenario: Scenario, ident_name: str) -> int:
    for i, p in enumerate(scenario.players):
        if p.name.lower() == ident_name.lower():
            return i
    raise ValueError(f"Query references unknown player '{ident_name}'")


def execute(query: Query, trials: int, seed: int | None) -> PQLResult:
    scenario = build_scenario(query)

    has_range = any(pl.pool is not None for pl in scenario.players)
    boards_only = None if has_range else enumerate_boards(scenario)
    if boards_only is not None:
        mode = "enumeration"
        eff_trials = boards_only.shape[0]
        H = scenario.game.num_hole
        hands = np.stack([pl.cards for pl in scenario.players])           # (P, H)
        player_hands = np.broadcast_to(hands, (eff_trials, len(scenario.players), H)).copy()
        boards = boards_only
    else:
        mode = "monte_carlo"
        eff_trials = trials
        player_hands, boards = sample_trials(scenario, trials, seed)

    scores = player_scores(scenario, player_hands, boards)  # (N, P)
    ctx = EvalContext(scenario=scenario, player_hands=player_hands, boards=boards, scores=scores)

    if query.where is not None:
        mask = eval_expr(query.where, ctx, scenario) != 0
    else:
        mask = np.ones(eff_trials, dtype=bool)
    denom = int(mask.sum())

    values_out: dict[str, float] = {}
    histograms_out: dict[str, dict] = {}
    columns: list[str] = []
    for i, item in enumerate(query.select):
        agg_call: Call = item.expr
        agg_name = agg_call.name.lower()
        expr_node = agg_call.args[0]
        alias = item.alias or f"col{i + 1}"
        columns.append(alias)
        column = eval_expr(expr_node, ctx, scenario)[mask]
        if agg_name == "histogram":
            histograms_out[alias] = _histogram(column, expr_node)
            continue
        if not is_scalar_aggregate(agg_name):
            raise ValueError(f"Unknown PQL aggregate '{agg_call.name}'")
        values_out[alias] = get_aggregate(agg_name)(column)

    return PQLResult(values=values_out, trials=denom, mode=mode,
                     seed=seed if mode == "monte_carlo" else None,
                     columns=columns, histograms=histograms_out)


def run_pql(query_text: str, trials: int = 20000, seed: int | None = None) -> PQLResult:
    """Parse and execute a PQL query. Public entry point."""
    return execute(parse(query_text), trials=trials, seed=seed)
