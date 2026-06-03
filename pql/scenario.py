from __future__ import annotations
from dataclasses import dataclass
import re
from typing import List
import numpy as np

from card_encoding import hand_str_to_ints
from pql.games import get_game, GameSpec
from pql.parser.ast import Query

_PLAYER_KEY = re.compile(r"^player_(\d+)$")
_CARD_TOKEN = re.compile(r"^([2-9TJQKA][shdc])+$")


@dataclass
class Player:
    name: str                       # canonical "PLAYER_1"
    cards: np.ndarray | None        # fixed hole cards, or None for a range
    pool: np.ndarray | None = None  # (M, num_hole) range combos, or None for fixed


@dataclass
class Scenario:
    game: GameSpec
    players: List[Player]
    board: np.ndarray  # 0..5 card ints
    dead: np.ndarray   # card ints


def _cards(value: str) -> np.ndarray:
    if value == "":
        return np.empty(0, dtype=np.int32)
    return hand_str_to_ints(value)


def build_scenario(query: Query) -> Scenario:
    params = query.params
    if "game" not in params:
        raise ValueError("FROM clause is missing required key 'game'")
    game = get_game(params["game"])

    board = _cards(params.get("board", ""))
    if len(board) > 5:
        raise ValueError(f"Board has {len(board)} cards; a board has at most 5")
    dead = _cards(params.get("dead", ""))

    players: List[Player] = []
    for key, value in params.items():
        m = _PLAYER_KEY.match(key)
        if not m:
            continue
        if not _CARD_TOKEN.match(value):
            from pql.ranges.parse import parse_range
            from pql.ranges.expand import expand
            terms = parse_range(value, game.name)
            pool = expand(terms, game.name, dead=dead)
            if pool.shape[0] == 0:
                raise ValueError(f"{key} range '{value}' matches no hands after dead cards")
            idx = int(m.group(1))
            players.append((idx, Player(name=f"PLAYER_{idx}", cards=None, pool=pool)))
            continue
        cards = _cards(value)
        if len(cards) != game.num_hole:
            raise ValueError(
                f"{key} has {len(cards)} cards but {game.name} expects {game.num_hole}"
            )
        idx = int(m.group(1))
        players.append((idx, Player(name=f"PLAYER_{idx}", cards=cards)))

    # Single-player scenarios are allowed: category / draw / outs / nut functions are
    # per-player. Cross-player functions (riverEquity/winsHi/tiesHi/winningHandType) with
    # one player return a degenerate value (you trivially "beat" an empty field), not an
    # error -- the agent only emits single-player functions for single-player intents.
    if len(players) < 1:
        raise ValueError("At least one player is required")

    # Reject physically impossible setups: any card used more than once across
    # board, dead, and all players' hole cards.
    from card_encoding import int_to_card
    counts: dict[int, int] = {}
    for c in board.tolist():
        counts[c] = counts.get(c, 0) + 1
    for c in dead.tolist():
        counts[c] = counts.get(c, 0) + 1
    for _, pl in players:
        if pl.cards is None:
            continue
        for c in pl.cards.tolist():
            counts[c] = counts.get(c, 0) + 1
    dupes = sorted(c for c, n in counts.items() if n > 1)
    if dupes:
        raise ValueError(
            "Duplicate card(s) used more than once across board/dead/players: "
            + ", ".join(int_to_card(c) for c in dupes)
        )

    players.sort(key=lambda t: t[0])
    return Scenario(game=game, players=[p for _, p in players], board=board, dead=dead)
