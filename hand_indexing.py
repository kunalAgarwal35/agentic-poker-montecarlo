"""
Fast hand-to-index conversion using combinatorial number system.

A 5-card hand from a 52-card deck can be uniquely mapped to an index 0 to C(52,5)-1 = 2,598,959.
This allows using a simple numpy array instead of a dictionary for score lookups.

The combinatorial number system gives us:
index = C(c0,1) + C(c1,2) + C(c2,3) + C(c3,4) + C(c4,5)
where c0 < c1 < c2 < c3 < c4 are the sorted card values.
"""

import numpy as np
from typing import Tuple
from functools import lru_cache
from itertools import combinations

# Total number of 5-card combinations from 52 cards
TOTAL_5CARD_HANDS = 2598960

# Precompute binomial coefficients C(n, k) for n=0..51, k=0..5
# This avoids repeated computation during hand evaluation
MAX_N = 52
MAX_K = 6
BINOMIAL = np.zeros((MAX_N + 1, MAX_K + 1), dtype=np.int64)

for n in range(MAX_N + 1):
    for k in range(min(n + 1, MAX_K + 1)):
        if k == 0:
            BINOMIAL[n, k] = 1
        elif k == n:
            BINOMIAL[n, k] = 1
        else:
            BINOMIAL[n, k] = BINOMIAL[n-1, k-1] + BINOMIAL[n-1, k]


def hand_to_index(cards: np.ndarray) -> int:
    """
    Convert sorted 5-card hand to unique index using combinatorial number system.
    
    Args:
        cards: Sorted numpy array of 5 card integers (0-51)
        
    Returns:
        Unique index (0 to 2,598,959)
    """
    sorted_cards = np.sort(cards)
    return int(
        BINOMIAL[sorted_cards[0], 1] +
        BINOMIAL[sorted_cards[1], 2] +
        BINOMIAL[sorted_cards[2], 3] +
        BINOMIAL[sorted_cards[3], 4] +
        BINOMIAL[sorted_cards[4], 5]
    )


def hand_to_index_unsorted(cards: np.ndarray) -> int:
    """
    Convert unsorted 5-card hand to unique index.
    
    Args:
        cards: Numpy array of 5 card integers (0-51), any order
        
    Returns:
        Unique index (0 to 2,598,959)
    """
    return hand_to_index(np.sort(cards))


def index_to_hand(index: int) -> np.ndarray:
    """
    Convert index back to sorted 5-card hand (inverse of hand_to_index).
    
    Args:
        index: Unique index (0 to 2,598,959)
        
    Returns:
        Sorted numpy array of 5 card integers
    """
    cards = np.zeros(5, dtype=np.int32)
    remaining = index
    
    for k in range(5, 0, -1):
        # Find the largest n such that C(n, k) <= remaining
        for n in range(51, -1, -1):
            if BINOMIAL[n, k] <= remaining:
                cards[k-1] = n
                remaining -= BINOMIAL[n, k]
                break
    
    return cards


# Precompute combination indices for PLO hand evaluation
# PLO requires: 2 cards from hole cards + 3 cards from board

def get_plo_combo_indices(num_hole_cards: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get precomputed combination indices for PLO hand evaluation.
    
    Args:
        num_hole_cards: 4 for PLO4, 5 for PLO5, 6 for PLO6
        
    Returns:
        Tuple of (hand_combos, board_combos) as numpy arrays
    """
    from itertools import combinations
    
    # 2-card combinations from hole cards
    hand_combos = np.array(list(combinations(range(num_hole_cards), 2)), dtype=np.int32)
    
    # 3-card combinations from 5-card board
    board_combos = np.array(list(combinations(range(5), 3)), dtype=np.int32)
    
    return hand_combos, board_combos


# Precompute for each PLO variant
PLO4_HAND_COMBOS, PLO4_BOARD_COMBOS = get_plo_combo_indices(4)  # 6 x 10 = 60 combos
PLO5_HAND_COMBOS, PLO5_BOARD_COMBOS = get_plo_combo_indices(5)  # 10 x 10 = 100 combos
PLO6_HAND_COMBOS, PLO6_BOARD_COMBOS = get_plo_combo_indices(6)  # 15 x 10 = 150 combos

# All C(7,5)=21 five-card subsets of a holdem 7-card set (2 hole + 5 board).
HOLDEM_7CARD_COMBOS = np.array(list(combinations(range(7), 5)), dtype=np.int32)  # (21, 5)


def get_combos_for_plo(plo_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """Get precomputed combos for PLO type ('plo4', 'plo5', 'plo6')."""
    if plo_type == 'plo4':
        return PLO4_HAND_COMBOS, PLO4_BOARD_COMBOS
    elif plo_type == 'plo5':
        return PLO5_HAND_COMBOS, PLO5_BOARD_COMBOS
    elif plo_type == 'plo6':
        return PLO6_HAND_COMBOS, PLO6_BOARD_COMBOS
    else:
        raise ValueError(f"Unknown PLO type: {plo_type}")


if __name__ == "__main__":
    # Test hand indexing
    print("Testing hand indexing...")
    
    # Test specific hands
    hand1 = np.array([0, 1, 2, 3, 4], dtype=np.int32)  # Lowest possible hand
    idx1 = hand_to_index(hand1)
    print(f"Lowest hand {hand1} -> index {idx1}")
    assert idx1 == 0, f"Expected 0, got {idx1}"
    
    hand2 = np.array([47, 48, 49, 50, 51], dtype=np.int32)  # Highest possible hand
    idx2 = hand_to_index(hand2)
    print(f"Highest hand {hand2} -> index {idx2}")
    assert idx2 == TOTAL_5CARD_HANDS - 1, f"Expected {TOTAL_5CARD_HANDS - 1}, got {idx2}"
    
    # Test roundtrip
    test_hand = np.array([5, 12, 23, 34, 45], dtype=np.int32)
    idx = hand_to_index(test_hand)
    recovered = index_to_hand(idx)
    assert np.array_equal(test_hand, recovered), f"Roundtrip failed: {test_hand} -> {idx} -> {recovered}"
    print(f"Roundtrip test passed: {test_hand} -> {idx} -> {recovered}")
    
    # Test combo indices
    print(f"\nPLO4 hand combos shape: {PLO4_HAND_COMBOS.shape}")  # (6, 2)
    print(f"PLO5 hand combos shape: {PLO5_HAND_COMBOS.shape}")  # (10, 2)
    print(f"PLO6 hand combos shape: {PLO6_HAND_COMBOS.shape}")  # (15, 2)
    print(f"Board combos shape: {PLO4_BOARD_COMBOS.shape}")  # (10, 3)
    
    print("\nAll hand indexing tests passed!")
