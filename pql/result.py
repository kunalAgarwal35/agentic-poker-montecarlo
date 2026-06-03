from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PQLResult:
    values: Dict[str, float]
    trials: int
    mode: str               # "monte_carlo" | "enumeration"
    seed: int | None = None
    columns: list = field(default_factory=list)        # alias order (scalar + histogram)
    histograms: Dict[str, dict] = field(default_factory=dict)  # alias -> {"pairs":[...], "labels":{}|None}
