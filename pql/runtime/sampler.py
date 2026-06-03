from __future__ import annotations
import numpy as np
from pql.scenario import Scenario
from pql.runtime._deck import live_deck


def complete_boards(scenario: Scenario, num_trials: int, seed: int | None) -> np.ndarray:
    """
    Return an (num_trials, 5) array of full boards. The first len(board) columns
    are the fixed board (repeated); the remaining columns are random distinct
    draws from the live deck (without replacement within each trial).

    Vectorized partial-shuffle: random keys per (trial, deck-card), argsort each
    row, take the first `need` columns. Seeded for reproducibility so the scalar
    and vector evaluation paths see identical trials.
    """
    board = scenario.board
    need = 5 - len(board)
    if need < 0:
        raise ValueError("Board already has more than 5 cards")

    deck = live_deck(scenario)
    rng = np.random.default_rng(seed)

    if need == 0:
        full = np.tile(board, (num_trials, 1)).astype(np.int32)
        return full

    keys = rng.random((num_trials, deck.shape[0]))
    order = np.argsort(keys, axis=1)[:, :need]       # (num_trials, need) deck indices
    drawn = deck[order]                              # (num_trials, need) card ints

    fixed = np.tile(board, (num_trials, 1)).astype(np.int32) if len(board) else \
        np.empty((num_trials, 0), dtype=np.int32)
    return np.concatenate([fixed, drawn], axis=1).astype(np.int32)


def sample_trials(scenario, num_trials: int, seed: int | None):
    """Return (player_hands (N,P,H), boards (N,5)). Fixed players repeat their hand;
    range players sample from their pool per trial (rejection vs used cards). Dead-aware."""
    rng = np.random.default_rng(seed)
    g = scenario.game
    H = g.num_hole
    P = len(scenario.players)
    base_board = scenario.board.tolist()
    need = 5 - len(base_board)
    dead = set(scenario.dead.tolist())

    pools = [pl.pool for pl in scenario.players]
    fixed = [None if pl.pool is not None else pl.cards.tolist() for pl in scenario.players]

    player_hands = np.empty((num_trials, P, H), dtype=np.int32)
    boards = np.empty((num_trials, 5), dtype=np.int32)

    for t in range(num_trials):
        used = set(dead) | set(base_board)
        for j in range(P):
            if pools[j] is None:
                hand = fixed[j]
            else:
                pool = pools[j]
                hand = None
                for _attempt in range(10000):
                    combo = pool[rng.integers(len(pool))]
                    cl = combo.tolist()
                    if not (used & set(cl)):
                        hand = cl
                        break
                if hand is None:
                    raise ValueError(
                        f"could not sample a hand for {scenario.players[j].name}: its range "
                        f"conflicts with the board/dead/other players' cards"
                    )
            player_hands[t, j] = hand
            used.update(hand)
        if need > 0:
            deck = [c for c in range(52) if c not in used]
            pick = rng.choice(len(deck), size=need, replace=False)
            comp = [deck[k] for k in pick]
        else:
            comp = []
        boards[t] = base_board + comp
    return player_hands, boards
