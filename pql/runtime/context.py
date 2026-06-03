from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from pql.scenario import Scenario


@dataclass
class EvalContext:
    """Per-query evaluation context: best-5 scores for every (trial, player),
    plus lazily-computed best-5 categories (only built when a category function asks)."""
    scenario: Scenario
    player_hands: np.ndarray       # (N, P, H)
    boards: np.ndarray             # (N, 5)
    scores: np.ndarray             # (N, P)
    _categories: np.ndarray | None = field(default=None)
    _nut_scores: np.ndarray | None = field(default=None)

    def categories(self) -> np.ndarray:
        if self._categories is None:
            from pql.runtime.evaluator import player_categories
            self._categories = player_categories(self.scenario, self.player_hands, self.boards)
        return self._categories

    def nut_scores(self) -> np.ndarray:
        if self._nut_scores is None:
            from pql.runtime.evaluator import compute_nut_scores
            self._nut_scores = compute_nut_scores(self.scenario, self.boards)
        return self._nut_scores
