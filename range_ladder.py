"""Postflop range ladder: hero equity vs board-strength percentile slices.

See docs/superpowers/plans/2026-08-13-postflop-range-ladder.md
"""
from itertools import combinations

import numpy as np

from card_encoding import generate_deck_ints, hand_str_to_ints, ints_to_hand_str
from hand_categories import CATEGORY_TOKENS
from hand_indexing import BINOMIAL
from hand_rank_evaluator import (
    _BOARD_COMBOS,
    _HAND_COMBOS,
    _batch_best_score,
    detect_game_type,
    get_score_array,
)
from optimized_evaluator import best5_category_omaha_numba, get_category_array

DEFAULT_BUCKETS = (5, 15, 25, 40, 60, 100)

# hand_categories.CATEGORY_TOKENS is index-aligned (low -> high strength) with
# the category_array.npy lookup used below, so mapping through it keeps this
# module's labels in lockstep with the engine's own naming (PQL's
# exactHandType, category_playground, etc.) instead of re-deriving strength
# bands from score_array by hand. Only the spacing/casing differs from the
# brief's nine literal labels.
_CATEGORY_LABELS = {
    "highcard": "high card",
    "pair": "pair",
    "twopair": "two pair",
    "trips": "trips",
    "straight": "straight",
    "flush": "flush",
    "fullhouse": "full house",
    "quads": "quads",
    "straightflush": "straight flush",
}


def sample_villains(deck, num_cards, n, rng):
    """Distinct legal villain hands drawn from `deck`, as sorted rows.

    Falls back to exhaustive enumeration when the space is smaller than `n`,
    so tiny decks return every hand exactly once instead of looping forever.
    """
    deck = np.asarray(deck, dtype=np.int32)
    total = len(deck)
    if total < num_cards:
        return np.empty((0, num_cards), dtype=np.int32)

    # Exhaustive when the space is small enough to enumerate cheaply.
    space = int(BINOMIAL[total, num_cards]) if total <= 52 and num_cards <= 6 else n + 1
    if space <= max(n, 1):
        rows = [sorted(c) for c in combinations(deck.tolist(), num_cards)]
        return np.array(rows, dtype=np.int32)

    seen = set()
    rows = []
    while len(rows) < n:
        need = n - len(rows)
        draw = rng.random((need * 2, total)).argsort(axis=1)[:, :num_cards]
        cand = np.sort(deck[draw], axis=1)
        for row in cand:
            key = tuple(row.tolist())
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            if len(rows) == n:
                break
    return np.array(rows, dtype=np.int32)


def sample_runouts(deck, board_len, r, rng):
    """`r` random completions of the board. River -> a single empty runout."""
    deck = np.asarray(deck, dtype=np.int32)
    need = 5 - board_len
    if need <= 0:
        return np.empty((1, 0), dtype=np.int32)
    draw = rng.random((r, len(deck))).argsort(axis=1)[:, :need]
    return deck[draw].astype(np.int32)


def eligible_mask(villains, runout):
    """False for villains holding a card that the runout also uses.

    Those pairings are impossible and must never be scored. Skipping them
    evaluates each villain over exactly the runouts compatible with it, which
    is the correct conditional distribution -- unbiased. See spec 4.3.
    """
    if runout.size == 0:
        return np.ones(villains.shape[0], dtype=bool)
    return ~np.isin(villains, runout).any(axis=1)


def score_hands(hands, board5, game, score_array, chunk=2000):
    """Best 5-card score for each hand on one complete board. Higher = better."""
    hand_combos = _HAND_COMBOS[game]
    n = hands.shape[0]
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk):
        block = hands[start:start + chunk]
        boards = np.broadcast_to(board5, (block.shape[0], 5))
        out[start:start + chunk] = _batch_best_score(
            block, np.ascontiguousarray(boards),
            hand_combos, _BOARD_COMBOS, score_array, BINOMIAL,
        )
    return out


def evaluate_population(villains, heroes, board_ints, runouts, game, score_array):
    """The single O((N + H) x R) pass. Returns (strength, hero_equity, counts).

    strength: (N,) float64 -- each villain's mean share of the eligible field
        it beats (win + 1/2 tie), i.e. its equity against the population.
        NaN-free: villains eligible on zero counted runouts get 0.0.
    hero_equity: (H, N) float64 -- hero h's equity against villain i, averaged
        over the runouts counted for i.
    counts: (N,) float64 -- the number of runouts that actually contributed to
        villain i's strength/hero_equity (i.e. i was eligible AND the runout
        had at least 2 eligible villains). This is a strict subset of "i was
        eligible on this runout" -- see the guard below -- so callers that
        need to know whether a villain has any real data must check `counts`,
        not re-derive eligibility themselves.
    """
    n = villains.shape[0]
    h = heroes.shape[0]

    beat_sum = np.zeros(n, dtype=np.float64)     # share of field beaten, summed
    beat_cnt = np.zeros(n, dtype=np.float64)     # runouts this villain was eligible for
    hero_sum = np.zeros((h, n), dtype=np.float64)

    for runout in runouts:
        board5 = np.concatenate([board_ints, runout]).astype(np.int32)
        mask = eligible_mask(villains, runout)
        idx = np.flatnonzero(mask)
        if idx.size < 2:
            # `strength` needs at least 2 eligible villains -- with idx.size == 1
            # the "share of field beaten" denominator (idx.size - 1) is zero,
            # and with idx.size == 0 there's no field at all. Skipping here is
            # correct for strength, but it also throws away hero-vs-villain
            # equity for this runout, which IS well defined for a single
            # eligible villain. At N=10000 the chance of <2 eligible villains
            # on a runout is essentially zero; at N==1 every runout hits this
            # guard and hero_equity comes back silently all-zero instead of
            # the correct heads-up number. Degenerate for tiny populations --
            # not restructured here, flagged for awareness.
            continue

        vs = score_hands(villains[idx], board5, game, score_array)
        hs = score_hands(heroes, board5, game, score_array)

        # Share of the eligible field each villain beats: win + 1/2 tie.
        order = np.sort(vs)
        lower = np.searchsorted(order, vs, side="left")          # strictly worse
        upper = np.searchsorted(order, vs, side="right")
        ties = upper - lower - 1                                  # excluding self
        share = (lower + 0.5 * ties) / (idx.size - 1)
        beat_sum[idx] += share
        beat_cnt[idx] += 1.0

        # Hero vs each eligible villain, heads-up.
        for j in range(h):
            hero_sum[j, idx] += np.where(hs[j] > vs, 1.0, np.where(hs[j] == vs, 0.5, 0.0))

    seen = beat_cnt > 0
    strength = np.zeros(n, dtype=np.float64)
    strength[seen] = beat_sum[seen] / beat_cnt[seen]
    hero_equity = np.zeros((h, n), dtype=np.float64)
    hero_equity[:, seen] = hero_sum[:, seen] / beat_cnt[seen]
    return strength, hero_equity, beat_cnt


def describe_category(hand_ints, board5):
    """Human label for a hand's made category, e.g. "flush", "full house".

    Delegates to optimized_evaluator's numba category lookup (the same
    category_array.npy the rest of the engine uses via
    pql.runtime.evaluator.player_categories) rather than re-deriving hand
    strength here, so labels agree with the rest of the engine by
    construction.
    """
    game = detect_game_type(len(hand_ints))
    idx = best5_category_omaha_numba(
        np.asarray(hand_ints, dtype=np.int32),
        np.asarray(board5, dtype=np.int32),
        _HAND_COMBOS[game],
        _BOARD_COMBOS,
        get_score_array(),
        get_category_array(),
        BINOMIAL,
    )
    return _CATEGORY_LABELS[CATEGORY_TOKENS[int(idx)]]


def build_rungs(strength, hero_equity_row, villains, buckets, board5_for_category):
    """One rung per bucket: hero equity vs that slice + the slice's weakest hand."""
    order = np.argsort(-strength, kind="stable")     # strongest first
    n = order.size
    rungs = []
    for pct in buckets:
        take = max(1, int(round(n * pct / 100.0)))
        sl = order[:take]
        edge_idx = int(sl[-1])                        # weakest hand in the slice
        rungs.append({
            "bucket": pct,
            "equity": float(hero_equity_row[sl].mean()),
            "edge": {
                "cards": ints_to_hand_str(villains[edge_idx]),
                "category": describe_category(villains[edge_idx], board5_for_category),
            },
        })
    return rungs


def compute_range_ladder(board, dead, heroes, buckets=DEFAULT_BUCKETS,
                         hands=10000, runouts=None, seed=None):
    """The whole feature: hero equity vs board-strength percentile slices.

    See spec Section 5 for the response shape.
    """
    board_ints = hand_str_to_ints(board)
    board_len = len(board_ints)
    if board_len not in (3, 4, 5):
        raise ValueError(f"board must be 3, 4 or 5 cards, got {board_len}")

    # Zero (or negative) runouts is only meaningful on the river, where the
    # count is 0 by construction and sample_runouts already returns the
    # single empty completion. Off the river it must raise, not silently
    # produce a (0, need) runout array -- that starves evaluate_population's
    # per-runout loop entirely, leaving every villain/hero at a structural
    # 0.0 strength/equity, and then crashes downstream on runout_rows[0]
    # (IndexError: empty array). `runouts is None` (the "use the default of
    # 800" case) is unaffected by this check.
    if runouts is not None and runouts <= 0 and board_len < 5:
        raise ValueError(
            f"runouts={runouts} is only meaningful on the river (5-card "
            f"board); board here has {board_len} cards. Omit `runouts` "
            "(defaults to 800) or pass a positive count."
        )

    # Built WITHOUT deduping first: a card repeated within one `dead` entry
    # ("AsAs9c2c") or across two different entries (two hands both claiming
    # "As") must show up as a literal duplicate here. Collapsing straight
    # into a set (as the old code did) silently absorbs both -- two hands
    # can never legally share a card.
    dead_cards = []
    for d in dead:
        dead_cards.extend(ints_to_hand_str(hand_str_to_ints(d))[i:i + 2]
                          for i in range(0, len(d), 2))
    if len(dead_cards) != len(set(dead_cards)):
        seen, dupes = set(), []
        for c in dead_cards:
            if c in seen and c not in dupes:
                dupes.append(c)
            seen.add(c)
        raise ValueError(f"duplicate card(s) in `dead`: {sorted(dupes)}")
    dead_set = set(dead_cards)

    # A card cannot be simultaneously dead (in a hand) and live (on the
    # board).
    board_norm = ints_to_hand_str(board_ints)
    board_cards = {board_norm[i:i + 2] for i in range(0, len(board_norm), 2)}
    overlap = dead_set & board_cards
    if overlap:
        raise ValueError(
            f"card(s) {sorted(overlap)} appear in both `dead` and `board`"
        )

    for hero in heroes:
        # Normalise exactly as `dead` was, so case differences cannot cause a
        # spurious rejection (postfloper lower-cases dead cards in places).
        norm = ints_to_hand_str(hand_str_to_ints(hero["cards"]))
        cards = [norm[i:i + 2] for i in range(0, len(norm), 2)]
        missing = [c for c in cards if c not in dead_set]
        if missing:
            raise ValueError(
                f"hero {hero['id']} cards {missing} absent from `dead`; "
                "villains would be allowed to hold them"
            )

    hero_arrays = np.stack([hand_str_to_ints(h["cards"]) for h in heroes])
    num_cards = hero_arrays.shape[1]
    game = detect_game_type(num_cards)
    score_array = get_score_array()

    deck = generate_deck_ints(list(dead_set) + [board[i:i + 2]
                                                for i in range(0, len(board), 2)])
    rng = np.random.default_rng(seed)

    villains = sample_villains(deck, num_cards, hands, rng)
    if villains.shape[0] == 0:
        raise ValueError("no legal villain hands remain")

    r = 1 if board_len == 5 else (runouts if runouts is not None else 800)
    runout_rows = sample_runouts(deck, board_len, r, rng)

    strength, hero_equity, counts = evaluate_population(
        villains, hero_arrays, board_ints, runout_rows, game, score_array
    )

    # A villain that contributed to no runout carries no data; drop it rather
    # than letting a structural 0.0 masquerade as "weakest hand in the
    # population". Filter on `counts` -- the contribution count evaluate_population
    # actually used -- NOT on a recomputed eligible_mask union. The two differ on
    # runouts skipped by the <2-eligible guard, and that gap is exactly how an
    # artefact reaches the user as a boundary hand.
    seen = counts > 0
    villains, strength, hero_equity = villains[seen], strength[seen], hero_equity[:, seen]

    # Category labels are read off the first sampled runout (runout_rows[0])
    # so a flop/turn board still names a complete 5-card hand; on the river
    # runout_rows[0] IS the real board (the only, empty-completion, row).
    # Loop-invariant across heroes, so it's computed once here rather than
    # inside the per-hero loop below.
    board5 = np.concatenate([board_ints, runout_rows[0]]).astype(np.int32)

    ladders = []
    for j, hero in enumerate(heroes):
        ladders.append({
            "id": hero["id"],
            "rungs": build_rungs(strength, hero_equity[j], villains, list(buckets), board5),
        })

    return {
        "population": int(villains.shape[0]),
        "runouts": 0 if board_len == 5 else int(runout_rows.shape[0]),
        "exact": board_len == 5,
        "ladders": ladders,
    }
