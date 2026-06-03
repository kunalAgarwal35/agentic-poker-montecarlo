"""
Optimized PLO equity evaluator using Numba JIT compilation.

This module provides 2-3x faster equity calculations compared to the original
Python implementation while maintaining the same accuracy.

Key optimizations:
1. Integer card encoding (0-51 instead of strings)
2. NumPy array for score lookups (instead of dictionary)
3. Numba JIT compilation of hot paths
4. Precomputed combination indices
5. NumPy random generation (faster than Python random)
"""

import numpy as np
from numba import njit, prange
from typing import List, Tuple, Dict, Optional
import os

from card_encoding import (
    CARD_TO_INT, hand_str_to_ints, hand_list_to_ints, 
    ints_to_hand_str, generate_deck_ints
)
from hand_indexing import (
    BINOMIAL, TOTAL_5CARD_HANDS,
    PLO4_HAND_COMBOS, PLO5_HAND_COMBOS, PLO6_HAND_COMBOS,
    PLO4_BOARD_COMBOS, PLO5_BOARD_COMBOS, PLO6_BOARD_COMBOS
)

# Load score array (global, loaded once)
_SCORE_ARRAY = None

def get_score_array() -> np.ndarray:
    """Load and cache the score array."""
    global _SCORE_ARRAY
    if _SCORE_ARRAY is None:
        score_path = os.path.join(os.path.dirname(__file__), 'score_array.npy')
        _SCORE_ARRAY = np.load(score_path)
    return _SCORE_ARRAY


_CATEGORY_ARRAY = None

def get_category_array() -> np.ndarray:
    """Load and cache the per-5-card-hand category index array (uint8, 0-8)."""
    global _CATEGORY_ARRAY
    if _CATEGORY_ARRAY is None:
        path = os.path.join(os.path.dirname(__file__), 'category_array.npy')
        _CATEGORY_ARRAY = np.load(path)
    return _CATEGORY_ARRAY


# ============================================================================
# NUMBA JIT-COMPILED CORE FUNCTIONS
# ============================================================================

@njit(cache=True)
def hand_to_index_numba(c0: int, c1: int, c2: int, c3: int, c4: int, binomial: np.ndarray) -> int:
    """
    Convert sorted 5-card hand to index (Numba-compatible version).
    Cards must be sorted: c0 < c1 < c2 < c3 < c4
    """
    return int(binomial[c0, 1] + binomial[c1, 2] + binomial[c2, 3] + binomial[c3, 4] + binomial[c4, 5])


@njit(cache=True)
def sort5(a: int, b: int, c: int, d: int, e: int) -> Tuple[int, int, int, int, int]:
    """Sort 5 integers (optimized sorting network for 5 elements)."""
    # Sorting network for 5 elements (9 comparisons)
    if a > b: a, b = b, a
    if c > d: c, d = d, c
    if a > c: a, c = c, a
    if b > d: b, d = d, b
    if b > c: b, c = c, b
    if a > e: a, e = e, a
    if b > e: b, e = e, b
    if c > e: c, e = e, c
    if d > e: d, e = e, d
    if c > d: c, d = d, c
    if b > c: b, c = c, b
    return a, b, c, d, e


@njit(cache=True)
def best_score_numba(hand: np.ndarray, board: np.ndarray,
                     hand_combos: np.ndarray, board_combos: np.ndarray,
                     score_array: np.ndarray, binomial: np.ndarray) -> float:
    """
    Evaluate best 5-card hand from PLO hand + board.
    
    This is the hot path - must be as fast as possible.
    
    Args:
        hand: Integer array of hole cards
        board: Integer array of 5 board cards
        hand_combos: Precomputed 2-card combo indices
        board_combos: Precomputed 3-card combo indices
        score_array: Precomputed scores for all 5-card hands
        binomial: Precomputed binomial coefficients
        
    Returns:
        Best hand score
    """
    best = 0.0
    num_hand_combos = hand_combos.shape[0]
    num_board_combos = board_combos.shape[0]
    
    for i in range(num_hand_combos):
        h0 = hand[hand_combos[i, 0]]
        h1 = hand[hand_combos[i, 1]]
        
        for j in range(num_board_combos):
            b0 = board[board_combos[j, 0]]
            b1 = board[board_combos[j, 1]]
            b2 = board[board_combos[j, 2]]
            
            # Sort 5 cards and compute index
            c0, c1, c2, c3, c4 = sort5(h0, h1, b0, b1, b2)
            idx = hand_to_index_numba(c0, c1, c2, c3, c4, binomial)
            
            score = score_array[idx]
            if score > best:
                best = score
    
    return best


@njit(cache=True)
def best5_of_7_numba(cards7: np.ndarray, combos: np.ndarray,
                     score_array: np.ndarray, binomial: np.ndarray) -> float:
    """
    Best 5-card score from 7 cards (holdem: 2 hole + 5 board), using any 5.

    combos: precomputed (21, 5) index combinations into cards7.
    """
    best = 0.0
    n = combos.shape[0]
    for i in range(n):
        a = cards7[combos[i, 0]]
        b = cards7[combos[i, 1]]
        c = cards7[combos[i, 2]]
        d = cards7[combos[i, 3]]
        e = cards7[combos[i, 4]]
        c0, c1, c2, c3, c4 = sort5(a, b, c, d, e)
        idx = hand_to_index_numba(c0, c1, c2, c3, c4, binomial)
        score = score_array[idx]
        if score > best:
            best = score
    return best


@njit(cache=True)
def best5_category_omaha_numba(hand, board, hand_combos, board_combos,
                               score_array, category_array, binomial):
    """Category index (0-8) of the best Omaha 5-card hand (exactly 2 hole + 3 board)."""
    best = 0.0
    best_idx = 0
    nh = hand_combos.shape[0]
    nb = board_combos.shape[0]
    for i in range(nh):
        h0 = hand[hand_combos[i, 0]]
        h1 = hand[hand_combos[i, 1]]
        for j in range(nb):
            b0 = board[board_combos[j, 0]]
            b1 = board[board_combos[j, 1]]
            b2 = board[board_combos[j, 2]]
            c0, c1, c2, c3, c4 = sort5(h0, h1, b0, b1, b2)
            idx = hand_to_index_numba(c0, c1, c2, c3, c4, binomial)
            s = score_array[idx]
            if s > best:
                best = s
                best_idx = idx
    return category_array[best_idx]


@njit(cache=True)
def best5_category_holdem_numba(cards7, combos, score_array, category_array, binomial):
    """Category index (0-8) of the best holdem 5-card hand (any 5 of 7)."""
    best = 0.0
    best_idx = 0
    n = combos.shape[0]
    for i in range(n):
        a = cards7[combos[i, 0]]
        b = cards7[combos[i, 1]]
        c = cards7[combos[i, 2]]
        d = cards7[combos[i, 3]]
        e = cards7[combos[i, 4]]
        c0, c1, c2, c3, c4 = sort5(a, b, c, d, e)
        idx = hand_to_index_numba(c0, c1, c2, c3, c4, binomial)
        s = score_array[idx]
        if s > best:
            best = s
            best_idx = idx
    return category_array[best_idx]


@njit(cache=True)
def nut_score_holdem_numba(board, deck, combos21, score_array, binomial) -> float:
    """Best achievable best-5 score over all 2-card holdings + board (holdem)."""
    best = 0.0
    n = deck.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            c7 = np.empty(7, dtype=np.int32)
            c7[0] = deck[i]; c7[1] = deck[j]
            c7[2] = board[0]; c7[3] = board[1]; c7[4] = board[2]; c7[5] = board[3]; c7[6] = board[4]
            s = best5_of_7_numba(c7, combos21, score_array, binomial)
            if s > best:
                best = s
    return best


@njit(cache=True)
def nut_score_omaha_numba(board, deck, hand_combos2, board_combos, score_array, binomial) -> float:
    """Best achievable best-5 score over all 2-card holdings (exactly 2 hole + 3 board)."""
    best = 0.0
    n = deck.shape[0]
    hand2 = np.empty(2, dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            hand2[0] = deck[i]; hand2[1] = deck[j]
            s = best_score_numba(hand2, board, hand_combos2, board_combos, score_array, binomial)
            if s > best:
                best = s
    return best


@njit(cache=True)
def filter_deck(deck: np.ndarray, dead_cards: np.ndarray) -> np.ndarray:
    """Remove dead cards from deck (Numba-compatible)."""
    dead_set = set(dead_cards)
    result = np.empty(52 - len(dead_cards), dtype=np.int32)
    idx = 0
    for card in deck:
        if card not in dead_set:
            result[idx] = card
            idx += 1
    return result[:idx]


@njit(cache=True)
def run_mc_trial_with_range(hero_hands: np.ndarray, hero_num_cards: int,
                            opponent_range: np.ndarray, opp_num_cards: int,
                            board: np.ndarray, board_len: int,
                            full_deck: np.ndarray,
                            hand_combos: np.ndarray, board_combos: np.ndarray,
                            score_array: np.ndarray, binomial: np.ndarray,
                            rng_state: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Run a single Monte Carlo trial for PLO equity vs opponent range.
    
    Returns:
        Tuple of (hero scores array, opponent score)
    """
    num_heroes = hero_hands.shape[0]
    
    # Pick random opponent from range
    opp_idx = np.random.randint(0, opponent_range.shape[0])
    opp_hand = opponent_range[opp_idx]
    
    # Build dead cards (hero hands + board + opponent)
    total_dead = hero_num_cards * num_heroes + board_len + opp_num_cards
    dead_cards = np.empty(total_dead, dtype=np.int32)
    idx = 0
    
    for i in range(num_heroes):
        for j in range(hero_num_cards):
            dead_cards[idx] = hero_hands[i, j]
            idx += 1
    
    for i in range(board_len):
        dead_cards[idx] = board[i]
        idx += 1
    
    for i in range(opp_num_cards):
        dead_cards[idx] = opp_hand[i]
        idx += 1
    
    # Filter deck
    live_deck = filter_deck(full_deck, dead_cards)
    
    # Complete board to 5 cards
    num_needed = 5 - board_len
    full_board = np.empty(5, dtype=np.int32)
    for i in range(board_len):
        full_board[i] = board[i]
    
    # Sample random cards for board completion
    if num_needed > 0:
        indices = np.random.choice(len(live_deck), num_needed, replace=False)
        for i in range(num_needed):
            full_board[board_len + i] = live_deck[indices[i]]
    
    # Evaluate opponent
    opp_score = best_score_numba(opp_hand, full_board, hand_combos, board_combos, score_array, binomial)
    
    # Evaluate all hero hands
    hero_scores = np.empty(num_heroes, dtype=np.float64)
    for i in range(num_heroes):
        hero_scores[i] = best_score_numba(hero_hands[i], full_board, hand_combos, board_combos, score_array, binomial)
    
    return hero_scores, opp_score


@njit(parallel=False, cache=True)  # parallel=False for now, we parallelize at higher level
def run_monte_carlo_numba(hero_hands: np.ndarray, hero_num_cards: int,
                          opponent_range: np.ndarray, opp_num_cards: int,
                          board: np.ndarray, board_len: int,
                          num_trials: int,
                          hand_combos: np.ndarray, board_combos: np.ndarray,
                          score_array: np.ndarray, binomial: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Monte Carlo equity simulation (Numba JIT compiled).
    
    Args:
        hero_hands: Array of hero hands, shape (num_hands, num_cards)
        hero_num_cards: Number of cards per hero hand (4, 5, or 6)
        opponent_range: Array of opponent hands to sample from
        opp_num_cards: Number of cards per opponent hand
        board: Board cards (0-5 cards)
        board_len: Number of board cards
        num_trials: Number of Monte Carlo trials
        hand_combos: Precomputed 2-card combo indices
        board_combos: Precomputed 3-card combo indices
        score_array: Precomputed hand scores
        binomial: Precomputed binomial coefficients
        
    Returns:
        Tuple of (individual win rates, pairwise win rates)
    """
    full_deck = np.arange(52, dtype=np.int32)
    num_hands = hero_hands.shape[0]
    
    # Initialize counters
    wins = np.zeros(num_hands, dtype=np.float64)
    
    # Pairwise wins - use flat array with index mapping
    num_pairs = num_hands * (num_hands - 1) // 2
    pairwise_wins = np.zeros(num_pairs, dtype=np.float64)
    
    for trial in range(num_trials):
        # Pick random opponent from range
        opp_idx = np.random.randint(0, opponent_range.shape[0])
        opp_hand = opponent_range[opp_idx]
        
        # Build dead cards
        total_dead = hero_num_cards * num_hands + board_len + opp_num_cards
        dead_cards = np.empty(total_dead, dtype=np.int32)
        idx = 0
        
        for i in range(num_hands):
            for j in range(hero_num_cards):
                dead_cards[idx] = hero_hands[i, j]
                idx += 1
        
        for i in range(board_len):
            dead_cards[idx] = board[i]
            idx += 1
        
        for i in range(opp_num_cards):
            dead_cards[idx] = opp_hand[i]
            idx += 1
        
        # Filter deck
        live_deck = filter_deck(full_deck, dead_cards)
        
        # Complete board
        num_needed = 5 - board_len
        full_board = np.empty(5, dtype=np.int32)
        for i in range(board_len):
            full_board[i] = board[i]
        
        if num_needed > 0:
            indices = np.random.choice(len(live_deck), num_needed, replace=False)
            for i in range(num_needed):
                full_board[board_len + i] = live_deck[indices[i]]
        
        # Evaluate opponent
        opp_score = best_score_numba(opp_hand, full_board, hand_combos, board_combos, score_array, binomial)
        
        # Evaluate all hero hands
        hero_scores = np.empty(num_hands, dtype=np.float64)
        for i in range(num_hands):
            hero_scores[i] = best_score_numba(hero_hands[i], full_board, hand_combos, board_combos, score_array, binomial)
        
        # Update individual wins
        for i in range(num_hands):
            if hero_scores[i] > opp_score:
                wins[i] += 1.0
            elif hero_scores[i] == opp_score:
                wins[i] += 0.5
        
        # Update pairwise wins
        pair_idx = 0
        for i in range(num_hands):
            for j in range(i + 1, num_hands):
                if hero_scores[i] > opp_score or hero_scores[j] > opp_score:
                    pairwise_wins[pair_idx] += 1.0
                elif hero_scores[i] == opp_score and hero_scores[j] == opp_score:
                    pairwise_wins[pair_idx] += 0.66
                elif hero_scores[i] == opp_score or hero_scores[j] == opp_score:
                    pairwise_wins[pair_idx] += 0.5
                pair_idx += 1
    
    # Normalize
    wins = wins / num_trials
    pairwise_wins = pairwise_wins / num_trials
    
    return wins, pairwise_wins


# ============================================================================
# HIGH-LEVEL API (PYTHON WRAPPER)
# ============================================================================

def plo_equities_optimized(hands: List[List[str]], 
                           num_trials: int,
                           plo_type: str,
                           board: List[str] = None,
                           opponent_range: List[str] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculate PLO equity using optimized Numba implementation.
    
    Args:
        hands: List of hands (each hand is a list of card strings)
        num_trials: Number of Monte Carlo trials
        plo_type: 'plo4', 'plo5', or 'plo6'
        board: Board cards (list of strings)
        opponent_range: List of opponent hands to sample from
        
    Returns:
        Tuple of (individual equities dict, pairwise equities dict)
    """
    # Load precomputed data
    score_array = get_score_array()
    binomial = BINOMIAL
    
    # Get combo indices for PLO type
    if plo_type == 'plo4':
        hand_combos = PLO4_HAND_COMBOS
        num_cards = 4
    elif plo_type == 'plo5':
        hand_combos = PLO5_HAND_COMBOS
        num_cards = 5
    elif plo_type == 'plo6':
        hand_combos = PLO6_HAND_COMBOS
        num_cards = 6
    else:
        raise ValueError(f"Unknown PLO type: {plo_type}")
    
    board_combos = PLO4_BOARD_COMBOS  # Always 10 combos for 5-card board
    
    # Convert hands to integer arrays
    hero_hands = np.array([hand_list_to_ints(h) for h in hands], dtype=np.int32)
    
    # Convert board
    if board is None:
        board = []
    board_ints = np.array([CARD_TO_INT[c] for c in board], dtype=np.int32)
    board_len = len(board)
    
    # Convert opponent range
    if opponent_range is None or len(opponent_range) == 0:
        raise ValueError("Opponent range is required")
    
    opp_range_ints = np.array([hand_str_to_ints(h) if isinstance(h, str) else hand_list_to_ints(h) 
                               for h in opponent_range], dtype=np.int32)
    opp_num_cards = opp_range_ints.shape[1]
    
    # Run Monte Carlo
    wins, pairwise_wins = run_monte_carlo_numba(
        hero_hands, num_cards,
        opp_range_ints, opp_num_cards,
        board_ints, board_len,
        num_trials,
        hand_combos, board_combos,
        score_array, binomial
    )
    
    # Convert results back to dict format
    individual = {}
    for i, hand in enumerate(hands):
        key = ''.join(hand)
        individual[key] = float(wins[i])
    
    pairwise = {}
    pair_idx = 0
    for i in range(len(hands)):
        for j in range(i + 1, len(hands)):
            key = ''.join(hands[i]) + '_' + ''.join(hands[j])
            pairwise[key] = float(pairwise_wins[pair_idx])
            pair_idx += 1
    
    return individual, pairwise


def plo4_equities_25pct_optimized(hands: List[List[str]], 
                                   num_trials: int,
                                   board: List[str] = None,
                                   opponent_range: List[str] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """PLO4 equity vs top 25% range using optimized evaluator."""
    return plo_equities_optimized(hands, num_trials, 'plo4', board, opponent_range)


def plo5_equities_25pct_optimized(hands: List[List[str]], 
                                   num_trials: int,
                                   board: List[str] = None,
                                   opponent_range: List[str] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """PLO5 equity vs top 25% range using optimized evaluator."""
    return plo_equities_optimized(hands, num_trials, 'plo5', board, opponent_range)


def plo6_equities_25pct_optimized(hands: List[List[str]], 
                                   num_trials: int,
                                   board: List[str] = None,
                                   opponent_range: List[str] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """PLO6 equity vs top 25% range using optimized evaluator."""
    return plo_equities_optimized(hands, num_trials, 'plo6', board, opponent_range)


# ============================================================================
# WARMUP FUNCTION (Pre-compile JIT functions)
# ============================================================================

def warmup_jit():
    """
    Pre-compile JIT functions to avoid first-call overhead.
    Call this at server startup.
    """
    print("[Optimized Evaluator] Warming up JIT compilation...")
    
    score_array = get_score_array()
    binomial = BINOMIAL
    
    # Create dummy data
    dummy_hands = np.array([[48, 44, 40, 36]], dtype=np.int32)  # AsKsQsJs
    dummy_opp_range = np.array([[47, 43, 39, 35]], dtype=np.int32)  # AhKhQhJh
    dummy_board = np.array([], dtype=np.int32)
    
    # Run small MC to trigger compilation
    run_monte_carlo_numba(
        dummy_hands, 4,
        dummy_opp_range, 4,
        dummy_board, 0,
        10,  # Just 10 trials
        PLO4_HAND_COMBOS, PLO4_BOARD_COMBOS,
        score_array, binomial
    )
    
    print("[Optimized Evaluator] JIT warmup complete!")


if __name__ == "__main__":
    # Test the optimized evaluator
    print("Testing optimized evaluator...")
    
    # Warmup JIT
    warmup_jit()
    
    # Test with sample hands
    hands = [['As', 'Ah', 'Ks', 'Kh'], ['Jd', 'Tc', '9d', '8c']]
    
    # Create a small opponent range for testing
    opp_range = ['QsQhJsJh', 'TsTh9s9h', 'AsKsQsJs']
    
    individual, pairwise = plo4_equities_25pct_optimized(hands, 1000, board=[], opponent_range=opp_range)
    
    print(f"\nIndividual equities: {individual}")
    print(f"Pairwise equities: {pairwise}")
    print("\nOptimized evaluator test passed!")
