"""
Integer-based card encoding for fast poker hand evaluation.

Card encoding: 0-51 where card = rank * 4 + suit
- rank: 0-12 (2, 3, 4, 5, 6, 7, 8, 9, T, J, Q, K, A)
- suit: 0-3 (s, h, d, c)

This encoding allows:
1. Fast arithmetic operations instead of string manipulation
2. NumPy array operations
3. Numba JIT compilation compatibility
"""

import numpy as np
from typing import List, Tuple, Union

# Constants
RANKS = '23456789TJQKA'
SUITS = 'shdc'
NUM_CARDS = 52

# Precompute lookup tables
CARD_TO_INT = {}
INT_TO_CARD = {}

for rank_idx, rank in enumerate(RANKS):
    for suit_idx, suit in enumerate(SUITS):
        card_str = rank + suit
        card_int = rank_idx * 4 + suit_idx
        CARD_TO_INT[card_str] = card_int
        INT_TO_CARD[card_int] = card_str

# NumPy array versions for fast lookup
CARD_STR_TO_INT_ARRAY = np.array([CARD_TO_INT.get(r+s, -1) for r in RANKS for s in SUITS], dtype=np.int32)


def card_to_int(card_str: str) -> int:
    """Convert card string like 'As' to integer (0-51)."""
    return CARD_TO_INT[card_str]


def int_to_card(card_int: int) -> str:
    """Convert integer (0-51) to card string like 'As'."""
    return INT_TO_CARD[card_int]


def hand_str_to_ints(hand_str: str) -> np.ndarray:
    """
    Convert hand string like 'AsKsQsJs' to numpy array of integers.
    
    Args:
        hand_str: Concatenated card strings (2 chars per card)
        
    Returns:
        NumPy array of card integers
    """
    cards = [hand_str[i:i+2] for i in range(0, len(hand_str), 2)]
    return np.array([CARD_TO_INT[c] for c in cards], dtype=np.int32)


def hand_list_to_ints(hand_list: List[str]) -> np.ndarray:
    """
    Convert list of card strings like ['As', 'Ks', 'Qs', 'Js'] to numpy array.
    
    Args:
        hand_list: List of card strings
        
    Returns:
        NumPy array of card integers
    """
    return np.array([CARD_TO_INT[c] for c in hand_list], dtype=np.int32)


def ints_to_hand_str(card_ints: np.ndarray) -> str:
    """
    Convert numpy array of integers back to hand string.
    
    Args:
        card_ints: NumPy array of card integers
        
    Returns:
        Concatenated card strings
    """
    return ''.join(INT_TO_CARD[int(c)] for c in card_ints)


def ints_to_hand_list(card_ints: np.ndarray) -> List[str]:
    """
    Convert numpy array of integers back to list of card strings.
    
    Args:
        card_ints: NumPy array of card integers
        
    Returns:
        List of card strings
    """
    return [INT_TO_CARD[int(c)] for c in card_ints]


def generate_deck_ints(exclude_cards: Union[List[str], np.ndarray] = None) -> np.ndarray:
    """
    Generate deck as numpy array of integers, excluding specified cards.
    
    Args:
        exclude_cards: Cards to exclude (list of strings or numpy array of ints)
        
    Returns:
        NumPy array of remaining card integers
    """
    if exclude_cards is None:
        return np.arange(52, dtype=np.int32)
    
    if isinstance(exclude_cards, np.ndarray):
        exclude_set = set(exclude_cards.tolist())
    else:
        # Convert strings to ints
        exclude_set = set(CARD_TO_INT[c] if isinstance(c, str) else c for c in exclude_cards)
    
    deck = np.array([i for i in range(52) if i not in exclude_set], dtype=np.int32)
    return deck


def get_rank(card_int: int) -> int:
    """Get rank (0-12) from card integer."""
    return card_int // 4


def get_suit(card_int: int) -> int:
    """Get suit (0-3) from card integer."""
    return card_int % 4


# Vectorized versions for NumPy arrays
def get_ranks(card_ints: np.ndarray) -> np.ndarray:
    """Get ranks from array of card integers."""
    return card_ints // 4


def get_suits(card_ints: np.ndarray) -> np.ndarray:
    """Get suits from array of card integers."""
    return card_ints % 4


if __name__ == "__main__":
    # Test the encoding
    print("Testing card encoding...")
    
    # Test individual cards
    assert card_to_int('2s') == 0
    assert card_to_int('As') == 48
    assert card_to_int('Ac') == 51
    
    assert int_to_card(0) == '2s'
    assert int_to_card(48) == 'As'
    assert int_to_card(51) == 'Ac'
    
    # Test hand conversion
    hand = 'AsKsQsJs'
    ints = hand_str_to_ints(hand)
    print(f"Hand '{hand}' -> {ints}")
    back = ints_to_hand_str(ints)
    assert back == hand, f"Roundtrip failed: {hand} -> {ints} -> {back}"
    
    # Test deck generation
    deck = generate_deck_ints(['As', 'Ks'])
    assert len(deck) == 50
    assert 48 not in deck  # As
    assert 44 not in deck  # Ks
    
    print("All card encoding tests passed!")
