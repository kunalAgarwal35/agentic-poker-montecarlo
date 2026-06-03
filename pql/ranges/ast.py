from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Pair: rank: str
@dataclass(frozen=True)
class PairPlus: rank: str           # e.g. QQ+ -> QQ,KK,AA
@dataclass(frozen=True)
class PairRange:
    low: str
    high: str                       # e.g. 22-99
@dataclass(frozen=True)
class Suited:
    r1: str
    r2: str
@dataclass(frozen=True)
class Offsuit:
    r1: str
    r2: str
@dataclass(frozen=True)
class Both:
    r1: str
    r2: str
@dataclass(frozen=True)
class Combo: text: str              # explicit, e.g. 'AsKh' / 'AsAhKsKh'
@dataclass(frozen=True)
class Percentile: pct: float
@dataclass(frozen=True)
class Any: pass
@dataclass
class Pattern:                      # Omaha rank pattern, e.g. AAxx / AAds / AKQJ
    ranks: List[str]               # required ranks (non-wildcard)
    num_wild: int                  # count of 'x' wildcards
    suitedness: Optional[str]      # None | 'ds' | 'ss' | 'ns'

    def __eq__(self, other):
        return (isinstance(other, Pattern) and list(self.ranks) == list(other.ranks)
                and self.num_wild == other.num_wild and self.suitedness == other.suitedness)


@dataclass(frozen=True)
class PercentileBand:
    lo: float
    hi: float                       # hands ranked between lo% and hi% (lo <= hi)


@dataclass(frozen=True)
class Exclude:
    base: tuple                     # tuple of terms (the base range)
    excl: tuple                     # tuple of terms to subtract
