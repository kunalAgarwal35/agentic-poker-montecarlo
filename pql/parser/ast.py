from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class Ident:
    """A bare identifier argument, e.g. PLAYER_1 or 'river'."""
    name: str


@dataclass
class IntLit:
    """An integer literal used inside an expression (e.g. the 5 in `outs > 5`)."""
    value: int


@dataclass
class BinOp:
    """Arithmetic: left <op> right, op in {+, -, *, /}."""
    op: str
    left: object
    right: object


@dataclass
class Compare:
    """Comparison producing a 0/1 column: left <op> right, op in {>, <, >=, <=, =, !=}."""
    op: str
    left: object
    right: object


@dataclass
class Logical:
    """Boolean combination: left <op> right, op in {and, or}."""
    op: str
    left: object
    right: object


@dataclass
class Not:
    """Boolean negation of a 0/1 column."""
    operand: object


@dataclass
class Call:
    """A function call: name(args...). Aggregates and value-functions are both Calls."""
    name: str
    args: List[Union["Call", Ident]] = field(default_factory=list)


@dataclass
class SelectItem:
    """One column in the SELECT list: an aggregate Call with an optional alias."""
    expr: Call
    alias: Optional[str] = None


@dataclass
class Query:
    """A full parsed PQL query."""
    select: List[SelectItem]
    params: dict  # lowercased-key -> raw string value from the FROM clause
    where: Optional[object] = None  # an expression node, or None
