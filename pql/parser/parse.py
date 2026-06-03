from __future__ import annotations
import os
from lark import Lark, Transformer, v_args

from pql.parser.ast import (
    Query, SelectItem, Call, Ident, IntLit, BinOp, Compare, Logical, Not,
)

_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "grammar.lark")

with open(_GRAMMAR_PATH, "r", encoding="utf-8") as _f:
    _PARSER = Lark(_f.read(), parser="lalr")


def _unquote(tok: str) -> str:
    return tok[1:-1]  # strip surrounding single quotes


@v_args(inline=True)
class _ToAst(Transformer):
    def ident(self, name):
        return Ident(str(name))

    def intlit(self, tok):
        # Integer literals (e.g. the threshold in minOutsToHandType(..., 9)) reuse the
        # Ident node carrying the digit string, so the executor's existing arg plumbing
        # (`[a.name for a in args if isinstance(a, Ident)]`) passes "9" straight through.
        return Ident(str(tok))

    def exprint(self, tok):
        # Integer literal used inside an expression (distinct from value-fn token ints).
        return IntLit(int(tok))

    def agg_call(self, name, expr):
        # An aggregate wraps a single expression argument.
        return Call(str(name), [expr])

    def not_op(self, operand):
        return Not(operand)

    def compare(self, *args):
        # 1 child -> no comparison present (collapse); 3 children -> left OP right.
        if len(args) == 1:
            return args[0]
        left, op, right = args
        return Compare(str(op), left, right)

    def arith(self, *args):
        # Fold a left-associative  atom (OP atom)*  chain into nested BinOps.
        node = args[0]
        i = 1
        while i < len(args):
            node = BinOp(str(args[i]), node, args[i + 1])
            i += 2
        return node

    def and_expr(self, *args):
        node = args[0]
        for rhs in args[1:]:
            node = Logical("and", node, rhs)
        return node

    def or_expr(self, *args):
        node = args[0]
        for rhs in args[1:]:
            node = Logical("or", node, rhs)
        return node

    def where_clause(self, expr):
        return expr

    def nested_call(self, call):
        return call

    def call_args(self, *args):
        return list(args)

    def call(self, name, args):
        return Call(str(name), args)

    def select_item(self, *parts):
        # keywords are filtered out, so parts is (call,) or (call, NAME)
        call = parts[0]
        alias = str(parts[1]) if len(parts) == 2 else None
        return SelectItem(call, alias)

    def select_list(self, *items):
        return list(items)

    def param(self, name, string):
        return (str(name).lower(), _unquote(str(string)))

    def from_clause(self, *params):
        return dict(params)

    def start(self, select_list, params, where=None):
        return Query(select=select_list, params=params, where=where)


def parse(text: str) -> Query:
    """Parse PQL query text into a Query AST."""
    tree = _PARSER.parse(text)
    return _ToAst().transform(tree)
