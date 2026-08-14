"""
Hand Rank evaluator: returns a 1-100 percentile rank for PLO4/PLO5/PLO6 starting hands.

Rank = 1 means a top-tier hand, 100 means the weakest tier.

Methodology (see documentations and learnings/HAND_RANK_API.md):
  1. Run a 3-way Monte Carlo: hero vs `num_opponents` (default 2) random opponent hands,
     board drawn at random to the river, over `num_trials` iterations.
  2. Returned value: hero's equity share (win=1, tie=1/(#tied+1), loss=0).
  3. Map the equity to a rank using a precomputed baseline CDF
     (`handrank_cdf_plo{4,5,6}.npy`) - one per game variant.

Two execution paths share identical public behavior:

  * Numba `@njit` path (activated when numba imports successfully, i.e. on
    production x86_64 instances just like `optimized_evaluator.py`). Inner
    loop is an explicit per-trial kernel with partial Fisher-Yates sampling
    (O(needed) per trial instead of O(deck_len)) and combinatorial-number-
    system indexing. Expected 5-10x faster than the NumPy path.

  * Pure NumPy vectorized path (fallback for platforms without Numba wheels,
    e.g. Windows ARM64 dev boxes). Argpartition-based sampling across all
    trials at once. This is what produced the current stress-test numbers.

The path is chosen automatically at import time, overridable via env var
`HANDRANK_USE_NUMBA` (`0` to force NumPy, `1` to force Numba if available).
Seeded calls always route to NumPy (NumPy's Generator gives reproducibility
across threads that Numba's thread-local RNG does not).
"""

import os
import numpy as np
from typing import List, Optional, Union, Dict, Any

from card_encoding import CARD_TO_INT, hand_list_to_ints
from hand_indexing import (
    BINOMIAL,
    PLO4_HAND_COMBOS,
    PLO5_HAND_COMBOS,
    PLO6_HAND_COMBOS,
    PLO4_BOARD_COMBOS,
)


# ---------------------------------------------------------------------------
# Numba feature-flag
# ---------------------------------------------------------------------------

try:
    from numba import njit  # noqa: F401
    HAVE_NUMBA = True
except Exception:  # pragma: no cover - platform without numba wheels
    HAVE_NUMBA = False

_NUMBA_ENV = os.environ.get('HANDRANK_USE_NUMBA', '1')
USE_NUMBA = HAVE_NUMBA and _NUMBA_ENV != '0'


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

_SCORE_ARRAY: Optional[np.ndarray] = None


def get_score_array() -> np.ndarray:
    """Load and cache score_array.npy (shape (2_598_960,))."""
    global _SCORE_ARRAY
    if _SCORE_ARRAY is None:
        path = os.path.join(os.path.dirname(__file__), 'score_array.npy')
        _SCORE_ARRAY = np.load(path)
    return _SCORE_ARRAY


_CDF_CACHE: Dict[str, Optional[np.ndarray]] = {}


def get_handrank_cdf(plo_type: str) -> Optional[np.ndarray]:
    """
    Load and cache the baseline CDF (sorted ascending equity array) for a PLO variant.
    Returns None if the CDF file has not yet been generated.
    """
    if plo_type in _CDF_CACHE:
        return _CDF_CACHE[plo_type]
    path = os.path.join(os.path.dirname(__file__), f'handrank_cdf_{plo_type}.npy')
    if not os.path.exists(path):
        _CDF_CACHE[plo_type] = None
    else:
        arr = np.load(path)
        arr.sort()
        _CDF_CACHE[plo_type] = arr
    return _CDF_CACHE[plo_type]


def reload_handrank_cdf() -> None:
    """Drop cached CDFs so the next call re-reads from disk."""
    _CDF_CACHE.clear()


# ---------------------------------------------------------------------------
# Game type + input parsing
# ---------------------------------------------------------------------------

_GAME_TYPES = {4: 'plo4', 5: 'plo5', 6: 'plo6'}
_NUM_CARDS = {'plo4': 4, 'plo5': 5, 'plo6': 6}
_HAND_COMBOS = {
    'plo4': PLO4_HAND_COMBOS,
    'plo5': PLO5_HAND_COMBOS,
    'plo6': PLO6_HAND_COMBOS,
}
_BOARD_COMBOS = PLO4_BOARD_COMBOS  # always 10 (3 from 5 board cards)


def detect_game_type(num_cards: int) -> str:
    if num_cards not in _GAME_TYPES:
        raise ValueError(
            f"Invalid hand size {num_cards}. Expected 4 (PLO4), 5 (PLO5), or 6 (PLO6)."
        )
    return _GAME_TYPES[num_cards]


def _to_card_list(value: Union[str, List[str], None]) -> List[str]:
    """
    Accept either a concatenated string ('AsKsQsJs') or an already-split list
    (['As', 'Ks', 'Qs', 'Js']). None / empty returns [].
    """
    if value is None or value == "" or value == []:
        return []
    if isinstance(value, list):
        return [str(c) for c in value]
    if isinstance(value, str):
        if len(value) % 2 != 0:
            raise ValueError(f"Invalid card string (odd length): {value!r}")
        return [value[i:i + 2] for i in range(0, len(value), 2)]
    raise TypeError(f"Unsupported card input: {type(value)!r}")


def _validate_cards(cards: List[str]) -> None:
    for c in cards:
        if c not in CARD_TO_INT:
            raise ValueError(f"Invalid card: {c!r}")


# ---------------------------------------------------------------------------
# Vectorized Monte Carlo core
# ---------------------------------------------------------------------------

def _sample_without_replacement(rng, num_trials, deck_len, needed):
    """`needed` distinct deck positions per trial, uniform in EVERY output slot.

    Returns (num_trials, needed) int indices into the deck.

    The slot-level guarantee is the point. Callers slice this positionally --
    the first `board_needed` columns become board cards, the rest are dealt out
    as opponent hands -- so it is not enough for the SET to be uniform; each
    column must be too.

    The previous implementation took `np.argpartition(rand_vals, needed - 1)`
    and sliced the first `needed` columns, with a comment arguing that "we only
    need a random unordered slice". That reasoning does not survive the
    positional slicing above. argpartition yields a uniform random subset but
    leaves it in partition order, which correlates with deck position, so the
    board drew systematically lower-ranked cards than the opponents: mean deck
    index 22.64 in board slots against 23.73 in opponent slots, a 2.78 spread
    where an unbiased sampler gives 23.5 everywhere. Hero equity came out ~3-4.5
    points off against ProPokerTools PQL.

    A full argsort of the random values is a genuine per-row permutation, so
    every column is an unbiased draw. It costs O(deck_len log deck_len) instead
    of O(deck_len), which on a 52-card deck is not worth a correctness risk.
    """
    rand_vals = rng.random((num_trials, deck_len))
    return np.argsort(rand_vals, axis=1)[:, :needed]


def _batch_best_score(
    hands: np.ndarray,          # (N, num_cards) int32
    boards: np.ndarray,         # (N, 5) int32
    hand_combos: np.ndarray,    # (C_h, 2)
    board_combos: np.ndarray,   # (C_b, 3)
    score_array: np.ndarray,
    binomial: np.ndarray,
) -> np.ndarray:
    """
    For each (hand, board) row, enumerate every (2-card hand combo) x (3-card board combo),
    look up its 5-card score, and return the per-row maximum.

    Returns: (N,) float64
    """
    N = hands.shape[0]
    C_h = hand_combos.shape[0]
    C_b = board_combos.shape[0]

    # Fancy-index: (N, C_h, 2) and (N, C_b, 3)
    hand_pairs = hands[:, hand_combos]
    board_triples = boards[:, board_combos]

    # Broadcast to (N, C_h, C_b, *) then concatenate into full 5-card rows
    hp = np.broadcast_to(hand_pairs[:, :, None, :], (N, C_h, C_b, 2))
    bt = np.broadcast_to(board_triples[:, None, :, :], (N, C_h, C_b, 3))
    all5 = np.concatenate([hp, bt], axis=-1)      # (N, C_h, C_b, 5)
    all5 = all5.reshape(N, C_h * C_b, 5)
    all5 = np.sort(all5, axis=-1)

    # Combinatorial number system -> unique index into score_array
    idx = (
        binomial[all5[:, :, 0], 1]
        + binomial[all5[:, :, 1], 2]
        + binomial[all5[:, :, 2], 3]
        + binomial[all5[:, :, 3], 4]
        + binomial[all5[:, :, 4], 5]
    )
    scores = score_array[idx]
    return scores.max(axis=1)


def run_handrank_mc(
    hero_ints: np.ndarray,          # (num_cards,) int32
    num_cards: int,
    num_opponents: int,
    base_deck: np.ndarray,          # remaining deck after removing dead cards
    board: np.ndarray,              # (board_len,) int32
    board_len: int,
    num_trials: int,
    hand_combos: np.ndarray,
    board_combos: np.ndarray,
    score_array: np.ndarray,
    binomial: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Vectorized Monte Carlo: hero vs `num_opponents` random opponents on random runout.

    Returns hero's average equity share (1 for outright win, 1/(k+1) if tied with k opps,
    0 if any opp strictly beats hero).
    """
    if rng is None:
        rng = np.random.default_rng()

    deck_len = base_deck.shape[0]
    board_needed = 5 - board_len
    opp_cards_total = num_opponents * num_cards
    needed = board_needed + opp_cards_total
    if needed > deck_len:
        raise ValueError(
            f"Not enough live cards: need {needed}, have {deck_len} (too many dead cards?)"
        )

    perm = _sample_without_replacement(rng, num_trials, deck_len, needed)
    sampled = base_deck[perm]   # (num_trials, needed) int32

    boards = np.empty((num_trials, 5), dtype=np.int32)
    for i in range(board_len):
        boards[:, i] = board[i]
    if board_needed > 0:
        boards[:, board_len:] = sampled[:, :board_needed]

    hero_tile = np.broadcast_to(hero_ints, (num_trials, num_cards)).copy()
    hero_scores = _batch_best_score(hero_tile, boards, hand_combos, board_combos, score_array, binomial)

    hero_wins = np.ones(num_trials, dtype=bool)
    tie_counts = np.zeros(num_trials, dtype=np.int32)

    opp_offset = board_needed
    for o in range(num_opponents):
        start = opp_offset + o * num_cards
        opp_hands = sampled[:, start:start + num_cards]
        opp_scores = _batch_best_score(opp_hands, boards, hand_combos, board_combos, score_array, binomial)
        hero_wins &= ~(opp_scores > hero_scores)
        tie_counts += (opp_scores == hero_scores).astype(np.int32)

    per_trial_equity = np.where(hero_wins, 1.0 / (tie_counts + 1.0), 0.0)
    return float(per_trial_equity.mean())


# ---------------------------------------------------------------------------
# Numba kernel (activated on production x86_64; dev ARM64 falls through to NumPy)
# ---------------------------------------------------------------------------

if USE_NUMBA:

    @njit(cache=True)
    def _best_score_numba(
        hand,               # (num_cards,) int32
        board5,             # (5,) int32
        hand_combos,        # (C_h, 2) int32
        board_combos,       # (C_b, 3) int32
        score_array,        # (2_598_960,) float64 (score_array.npy is float64 today)
        binomial,           # (53, 7) int64
    ):
        """
        Max score over all (2-of-hand) x (3-of-board) combinations for one player.

        Uses the same combinatorial-number-system indexing as the NumPy path so
        both kernels address identical slots in score_array (bit-exact
        agreement required).
        """
        best = -1.0  # score_array values are >= 0, so -1 is a safe sentinel
        C_h = hand_combos.shape[0]
        C_b = board_combos.shape[0]
        cards5 = np.empty(5, dtype=np.int32)

        for h in range(C_h):
            for b in range(C_b):
                # All five slots must be re-seated every iteration. The insertion
                # sort below permutes cards5 in place, so after the first board
                # combo the hole cards no longer sit in slots 0 and 1 -- writing
                # only slots 2-4 here would overwrite whichever hole card the sort
                # moved down, and leave a stale board card behind. That defect
                # scored 9 of every 10 board combos against a corrupted card set:
                # 277/300 hands mis-scored, 35% of head-to-head comparisons
                # flipped, premium hands understated by ~13 equity points.
                cards5[0] = hand[hand_combos[h, 0]]
                cards5[1] = hand[hand_combos[h, 1]]
                cards5[2] = board5[board_combos[b, 0]]
                cards5[3] = board5[board_combos[b, 1]]
                cards5[4] = board5[board_combos[b, 2]]

                # Insertion sort - 5 elements, branch-predictable, beats
                # calling into np.sort from inside @njit.
                for i in range(1, 5):
                    key = cards5[i]
                    j = i - 1
                    while j >= 0 and cards5[j] > key:
                        cards5[j + 1] = cards5[j]
                        j -= 1
                    cards5[j + 1] = key

                idx = (
                    binomial[cards5[0], 1]
                    + binomial[cards5[1], 2]
                    + binomial[cards5[2], 3]
                    + binomial[cards5[3], 4]
                    + binomial[cards5[4], 5]
                )
                s = score_array[idx]
                if s > best:
                    best = s
        return best

    @njit(cache=True)
    def _run_handrank_mc_numba(
        hero_ints,          # (num_cards,) int32
        base_deck,          # (deck_len,) int32
        board_ints,         # (board_len,) int32
        num_trials,
        num_opponents,
        num_cards,
        score_array,
        hand_combos,
        board_combos,
        binomial,
    ):
        """
        Explicit per-trial MC with partial Fisher-Yates sampling.

        Per trial:
          1. Partial Fisher-Yates on a scratch copy of `base_deck` to pick
             the first `needed` cards uniformly without replacement.
             O(needed) swaps, no (T, deck_len) allocation.
          2. Fill the remaining board slots from the shuffle prefix.
          3. Evaluate hero's best 5-card score; evaluate each opponent's
             best; accumulate equity share (1 / (1 + ties) on win, 0 on loss).

        Numba's `np.random` state is thread-local under @njit, so concurrent
        Flask threads do not contend or produce correlated draws.
        """
        deck_len = base_deck.shape[0]
        board_len = board_ints.shape[0]
        board_needed = 5 - board_len
        opp_cards_total = num_opponents * num_cards
        needed = board_needed + opp_cards_total

        # Pre-allocate scratch buffers (reused across trials).
        deck_copy = base_deck.copy()
        board5 = np.empty(5, dtype=np.int32)
        for i in range(board_len):
            board5[i] = board_ints[i]
        opp_cards = np.empty(num_cards, dtype=np.int32)

        total_equity = 0.0

        for trial in range(num_trials):
            # Partial Fisher-Yates: only shuffle the first `needed` positions.
            for i in range(needed):
                j = i + np.random.randint(0, deck_len - i)
                tmp = deck_copy[i]
                deck_copy[i] = deck_copy[j]
                deck_copy[j] = tmp

            for i in range(board_needed):
                board5[board_len + i] = deck_copy[i]

            hero_best = _best_score_numba(
                hero_ints, board5, hand_combos, board_combos, score_array, binomial
            )

            hero_wins = True
            tie_count = 0
            opp_start = board_needed
            for o in range(num_opponents):
                base = opp_start + o * num_cards
                for k in range(num_cards):
                    opp_cards[k] = deck_copy[base + k]
                opp_best = _best_score_numba(
                    opp_cards, board5, hand_combos, board_combos, score_array, binomial
                )
                if opp_best > hero_best:
                    hero_wins = False
                    # Still need to finish loop? No - any single-loss kills equity.
                    # Early-break is safe because ties also require all opponents
                    # to be equal (tie_count only matters if hero_wins stays True).
                    break
                elif opp_best == hero_best:
                    tie_count += 1

            if hero_wins:
                total_equity += 1.0 / (1.0 + tie_count)

        return total_equity / num_trials


# ---------------------------------------------------------------------------
# Dispatcher - chooses Numba vs NumPy path per call
# ---------------------------------------------------------------------------

def _compute_equity(
    hero_ints: np.ndarray,
    num_cards: int,
    num_opponents: int,
    base_deck: np.ndarray,
    board_ints: np.ndarray,
    num_trials: int,
    hand_combos: np.ndarray,
    board_combos: np.ndarray,
    seed: Optional[int],
) -> float:
    """
    Single entry point used by hand_rank_single. Routes to the fastest kernel
    available on the current platform.

    Seeded calls always use the NumPy Generator path (reproducibility across
    threads). Unseeded calls use Numba when available.
    """
    score_array = get_score_array()

    if USE_NUMBA and seed is None:
        return float(
            _run_handrank_mc_numba(
                hero_ints.astype(np.int32, copy=False),
                base_deck.astype(np.int32, copy=False),
                board_ints.astype(np.int32, copy=False),
                int(num_trials),
                int(num_opponents),
                int(num_cards),
                score_array,
                hand_combos,
                board_combos,
                BINOMIAL,
            )
        )

    rng = np.random.default_rng(seed)
    return run_handrank_mc(
        hero_ints, num_cards, num_opponents,
        base_deck,
        board_ints, len(board_ints),
        num_trials,
        hand_combos, _BOARD_COMBOS,
        score_array, BINOMIAL,
        rng=rng,
    )


# ---------------------------------------------------------------------------
# Rank mapping
# ---------------------------------------------------------------------------

def equity_to_rank(equity: float, cdf: Optional[np.ndarray]) -> Optional[int]:
    """
    Map a 3-way equity to an integer rank in [1, 100] using the precomputed CDF.
    Returns None when CDF is unavailable.

    cdf is sorted ascending. Rank = 1 means top tier (highest equity).
    """
    if cdf is None or len(cdf) == 0:
        return None
    below = int(np.searchsorted(cdf, equity, side='right'))
    remaining = len(cdf) - below
    rank = int(np.ceil(remaining / len(cdf) * 100))
    if rank < 1:
        rank = 1
    if rank > 100:
        rank = 100
    return rank


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hand_rank_single(
    hand: Union[str, List[str]],
    dead_cards: Union[str, List[str], None] = None,
    board: Union[str, List[str], None] = None,
    num_opponents: int = 2,
    num_trials: int = 5000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute rank for a single hero hand.

    Returns a dict with: hand, game_type, equity, rank, trials, num_opponents.
    rank is None if the baseline CDF for that variant has not been generated yet.
    """
    hand_list = _to_card_list(hand)
    dead_list = _to_card_list(dead_cards)
    board_list = _to_card_list(board)
    _validate_cards(hand_list + dead_list + board_list)

    plo_type = detect_game_type(len(hand_list))
    num_cards = _NUM_CARDS[plo_type]
    hand_combos = _HAND_COMBOS[plo_type]

    hero_ints = hand_list_to_ints(hand_list)
    dead_ints = np.array([CARD_TO_INT[c] for c in dead_list], dtype=np.int32) if dead_list else np.empty(0, dtype=np.int32)
    board_ints = np.array([CARD_TO_INT[c] for c in board_list], dtype=np.int32) if board_list else np.empty(0, dtype=np.int32)

    all_fixed = np.concatenate([hero_ints, dead_ints, board_ints])
    if len(set(all_fixed.tolist())) != len(all_fixed):
        raise ValueError("Duplicate cards detected across hero / dead / board inputs.")

    full_deck = np.arange(52, dtype=np.int32)
    mask = np.ones(52, dtype=bool)
    mask[all_fixed] = False
    base_deck = full_deck[mask]

    equity = _compute_equity(
        hero_ints, num_cards, num_opponents,
        base_deck, board_ints,
        num_trials,
        hand_combos, _BOARD_COMBOS,
        seed,
    )

    rank = equity_to_rank(equity, get_handrank_cdf(plo_type))
    return {
        'hand': ''.join(hand_list),
        'game_type': plo_type,
        'equity': equity,
        'rank': rank,
        'trials': num_trials,
        'num_opponents': num_opponents,
    }


def hand_rank_batch(
    hands: List[Union[str, List[str]]],
    dead_cards: Union[str, List[str], None] = None,
    board: Union[str, List[str], None] = None,
    num_opponents: int = 2,
    num_trials: int = 5000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Rank multiple hero hands in one call.

    Each hand's MC treats ALL OTHER hero hands' cards as additional dead cards,
    matching the semantics of the existing /plo*python* endpoints.

    Concurrency model (matches the newer optimized_parallel_runner pattern):
      * Each request runs single-process. The vectorized NumPy inner loop
        is fast enough that submitting to the shared ProcessPool would
        only add IPC overhead and fight legacy endpoints for the 4
        workers on each EC2 instance.
      * Concurrent requests parallelize naturally across Flask threads
        because NumPy releases the GIL during vectorized ops. The stress
        test verified ~4x throughput scaling from 1 -> 8 worker threads.

    Raises ValueError if hands have mixed card counts (mixed PLO types).
    """
    if not hands:
        raise ValueError("No hands provided.")

    parsed = [_to_card_list(h) for h in hands]
    for cards in parsed:
        _validate_cards(cards)

    sizes = {len(h) for h in parsed}
    if len(sizes) != 1:
        raise ValueError(
            f"All hands in one request must be the same PLO variant. Got sizes: {sorted(sizes)}"
        )
    plo_type = detect_game_type(next(iter(sizes)))

    dead_list = _to_card_list(dead_cards)
    board_list = _to_card_list(board)

    ranks: Dict[str, Optional[int]] = {}
    equities: Dict[str, float] = {}

    for i, hero in enumerate(parsed):
        others: List[str] = []
        for j, other in enumerate(parsed):
            if j != i:
                others.extend(other)
        combined_dead = dead_list + others
        sub_seed = None if seed is None else seed + i
        result = hand_rank_single(
            hero,
            dead_cards=combined_dead,
            board=board_list,
            num_opponents=num_opponents,
            num_trials=num_trials,
            seed=sub_seed,
        )
        key = ''.join(hero)
        ranks[key] = result['rank']
        equities[key] = result['equity']

    return {
        'game_type': plo_type,
        'ranks': ranks,
        'equities': equities,
        'trials': num_trials,
        'num_opponents': num_opponents,
        'cdf_available': get_handrank_cdf(plo_type) is not None,
    }


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def warmup_handrank() -> None:
    """
    Pre-load score_array and baseline CDFs, trigger Numba compilation (on
    platforms that have it), and run one tiny MC per PLO variant to warm
    numpy + numba caches. Safe to call multiple times.
    """
    print(f"[HandRank] Warming up (numba={'on' if USE_NUMBA else 'off'}, path={'numba' if USE_NUMBA else 'numpy'})...")
    get_score_array()
    for plo in ('plo4', 'plo5', 'plo6'):
        cdf = get_handrank_cdf(plo)
        if cdf is None:
            print(f"[HandRank] WARNING: baseline CDF for {plo} not found -> /handrank will return equity only.")
        else:
            print(f"[HandRank] {plo} CDF loaded: n={len(cdf)}, min={cdf.min():.4f}, median={np.median(cdf):.4f}, max={cdf.max():.4f}")

    sample_hands = {
        'plo4': 'AsKsQsJs',
        'plo5': 'AsKsQsJsTs',
        'plo6': 'AsKsQsJsTs9s',
    }
    # First run on the numpy path (always available).
    for plo, sample in sample_hands.items():
        try:
            hand_rank_single(sample, num_trials=50, seed=0)
        except Exception as e:
            print(f"[HandRank] Warmup {plo} MC (numpy path) failed: {e}")

    # Second run WITHOUT a seed forces the Numba dispatcher so LLVM compiles
    # and caches the njit kernels (first call would otherwise eat the ~1s JIT
    # cost on the first real request).
    if USE_NUMBA:
        import time as _time
        for plo, sample in sample_hands.items():
            t0 = _time.time()
            try:
                hand_rank_single(sample, num_trials=100, seed=None)
                print(f"[HandRank] Numba JIT warmed for {plo} in {_time.time()-t0:.2f}s")
            except Exception as e:
                print(f"[HandRank] Numba warmup {plo} failed: {e}")

    print("[HandRank] Warmup complete.")


if __name__ == "__main__":
    import time
    warmup_handrank()

    for trials in (500, 2000, 5000):
        for sample in ('AsAhKsKh', 'AsAhKsKhQs', 'AsAhKsKhQsJs'):
            t0 = time.perf_counter()
            r = hand_rank_single(sample, num_trials=trials, seed=42)
            dt = (time.perf_counter() - t0) * 1000
            print(f"trials={trials} hand={sample} equity={r['equity']:.4f} rank={r['rank']} t={dt:.1f}ms")
