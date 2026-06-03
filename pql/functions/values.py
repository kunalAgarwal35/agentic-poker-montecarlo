from __future__ import annotations
import numpy as np
from pql.functions.registry import register_value
from hand_categories import TOKEN_TO_INDEX
from pql.functions._outs import (
    outs_count, street_index, infer_street, nut_hi_for_own_category,
    category_at_street, MODE_GE, MODE_MEMBER,
)


def _category(token: str) -> int:
    t = token.lower()
    if t not in TOKEN_TO_INDEX:
        raise ValueError(f"Unknown hand-type '{token}'. Expected one of {sorted(TOKEN_TO_INDEX)}")
    return TOKEN_TO_INDEX[t]


_STRAIGHT = TOKEN_TO_INDEX["straight"]
_FLUSH = TOKEN_TO_INDEX["flush"]
_STRAIGHTFLUSH = TOKEN_TO_INDEX["straightflush"]


@register_value("riverEquity")
def river_equity(ctx, player_idx, args) -> np.ndarray:
    scores = ctx.scores
    mx = scores.max(axis=1)
    is_max = scores == mx[:, None]
    winners = is_max.sum(axis=1)
    player_is_winner = is_max[:, player_idx]
    return np.where(player_is_winner, 1.0 / winners, 0.0)


@register_value("winsHi")
def wins_hi(ctx, player_idx, args) -> np.ndarray:
    scores = ctx.scores
    mx = scores.max(axis=1)
    is_max = scores == mx[:, None]
    winners = is_max.sum(axis=1)
    sole = is_max[:, player_idx] & (winners == 1)
    return sole.astype(np.float64)


@register_value("tiesHi")
def ties_hi(ctx, player_idx, args) -> np.ndarray:
    """1.0 iff player_idx shares the max with at least one other player."""
    scores = ctx.scores
    mx = scores.max(axis=1)
    is_max = scores == mx[:, None]
    winners = is_max.sum(axis=1)
    tied = is_max[:, player_idx] & (winners > 1)
    return tied.astype(np.float64)


@register_value("exactHandType")
def exact_hand_type(ctx, player_idx, args) -> np.ndarray:
    """1.0 iff the player's best-5 category at `street` equals the category token.
    args = [street, category]; street is flop/turn/river."""
    street = street_index(args[0], allow_river=True)
    target = _category(args[1])
    return (category_at_street(ctx, player_idx, street) == target).astype(np.float64)


@register_value("minHandType")
def min_hand_type(ctx, player_idx, args) -> np.ndarray:
    street = street_index(args[0], allow_river=True)
    target = _category(args[1])
    return (category_at_street(ctx, player_idx, street) >= target).astype(np.float64)


@register_value("maxHandType")
def max_hand_type(ctx, player_idx, args) -> np.ndarray:
    street = street_index(args[0], allow_river=True)
    target = _category(args[1])
    return (category_at_street(ctx, player_idx, street) <= target).astype(np.float64)


@register_value("handType")
def hand_type(ctx, player_idx, args) -> np.ndarray:
    """The player's best-5 made-hand category index (0-8) at the given street.
    args = [street]; street is flop/turn/river (defaults to river). This is the raw
    category used by histogram(handType(...)) and by where-clause conditions."""
    street = street_index(args[0], allow_river=True) if args else 5
    return category_at_street(ctx, player_idx, street).astype(np.float64)


@register_value("nutHi")
def nut_hi(ctx, player_idx, args) -> np.ndarray:
    """1.0 iff the player's best-5 score equals the board's best achievable (the nuts)."""
    return (ctx.scores[:, player_idx] == ctx.nut_scores()).astype(np.float64)


@register_value("winningHandType")
def winning_hand_type(ctx, player_idx, args) -> np.ndarray:
    """1.0 iff the winning (max-score) player's category equals the token. Cross-player:
    the player arg is ignored. Ties share the max score -> same category."""
    token = args[-1].lower()
    if token not in TOKEN_TO_INDEX:
        raise ValueError(f"Unknown hand-type '{token}'. Expected one of {sorted(TOKEN_TO_INDEX)}")
    scores = ctx.scores
    cats = ctx.categories()
    winner = scores.argmax(axis=1)
    winner_cat = cats[np.arange(cats.shape[0]), winner]
    return (winner_cat == TOKEN_TO_INDEX[token]).astype(np.float64)


@register_value("playersWithBestHi")
def players_with_best_hi(ctx, player_idx, args) -> np.ndarray:
    """Per-trial count of players whose score equals the max (the player arg is ignored)."""
    scores = ctx.scores
    mx = scores.max(axis=1)
    return (scores == mx[:, None]).sum(axis=1).astype(np.float64)


@register_value("outsToHandType")
def outs_to_hand_type(ctx, player_idx, args) -> np.ndarray:
    """Per-trial count of single-card outs at `street` to reach >= `handtype`.
    args = [street, handtype]; street is 'flop' or 'turn'."""
    street = street_index(args[0])
    target = _category(args[1])
    return outs_count(ctx, player_idx, street, MODE_GE, target, target).astype(np.float64)


@register_value("minOutsToHandType")
def min_outs_to_hand_type(ctx, player_idx, args) -> np.ndarray:
    """1.0 iff outsToHandType(...) >= n. args = [street, handtype, n]."""
    street = street_index(args[0])
    target = _category(args[1])
    n = int(args[2])
    return (outs_count(ctx, player_idx, street, MODE_GE, target, target) >= n).astype(np.float64)


@register_value("flushDraw")
def flush_draw(ctx, player_idx, args) -> np.ndarray:
    """1.0 iff the player holds a flush draw (>= 8 single-card outs to a flush)."""
    s = infer_street(ctx, args)
    o = outs_count(ctx, player_idx, s, MODE_MEMBER, _FLUSH, _STRAIGHTFLUSH)
    return (o >= 8).astype(np.float64)


@register_value("straightDraw")
def straight_draw(ctx, player_idx, args) -> np.ndarray:
    """1.0 iff the player holds any straight draw (>= 4 single-card straight outs)."""
    s = infer_street(ctx, args)
    o = outs_count(ctx, player_idx, s, MODE_MEMBER, _STRAIGHT, _STRAIGHTFLUSH)
    return (o >= 4).astype(np.float64)


@register_value("oesd")
def open_ended_straight_draw(ctx, player_idx, args) -> np.ndarray:
    """1.0 iff the player holds an open-ended straight draw (>= 8 single-card straight outs)."""
    s = infer_street(ctx, args)
    o = outs_count(ctx, player_idx, s, MODE_MEMBER, _STRAIGHT, _STRAIGHTFLUSH)
    return (o >= 8).astype(np.float64)


@register_value("gutshot")
def gutshot(ctx, player_idx, args) -> np.ndarray:
    """1.0 iff the player holds a pure inside (gut-shot) straight draw (4-7 straight outs)."""
    s = infer_street(ctx, args)
    o = outs_count(ctx, player_idx, s, MODE_MEMBER, _STRAIGHT, _STRAIGHTFLUSH)
    return ((o >= 4) & (o <= 7)).astype(np.float64)


@register_value("nutHiForHandType")
def nut_hi_for_hand_type(ctx, player_idx, args) -> np.ndarray:
    """1.0 iff the player holds the nut hand of their own made category at `street`.
    args = [street]; street is flop/turn/river. Matches PPT nutHiForHandType(player, street)."""
    street = street_index(args[0], allow_river=True)
    return nut_hi_for_own_category(ctx, player_idx, street)
