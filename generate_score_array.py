"""
Convert score_dict.pkl to score_array.npy for fast array-indexed lookups.

This script:
1. Loads the existing score_dict.pkl (maps hand strings to scores)
2. Converts each hand string to an integer index using combinatorial indexing
3. Creates a numpy array where array[index] = score
4. Saves as score_array.npy for efficient memory-mapped access
"""

import numpy as np
import pickle
import os
from itertools import combinations

from card_encoding import CARD_TO_INT, hand_str_to_ints, RANKS, SUITS
from hand_indexing import hand_to_index, TOTAL_5CARD_HANDS, BINOMIAL


def load_score_dict(filepath: str = 'score_dict.pkl') -> dict:
    """Load the existing score dictionary from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def convert_to_score_array(score_dict: dict) -> np.ndarray:
    """
    Convert score dictionary to numpy array indexed by combinatorial hand index.
    
    Args:
        score_dict: Dictionary mapping sorted hand strings to scores
        
    Returns:
        NumPy array of shape (2598960,) with scores
    """
    print(f"Converting {len(score_dict)} hands to score array...")
    
    # Create array initialized with zeros
    score_array = np.zeros(TOTAL_5CARD_HANDS, dtype=np.float64)
    
    converted = 0
    errors = 0
    
    for hand_str, score in score_dict.items():
        try:
            # Convert hand string to card integers
            card_ints = hand_str_to_ints(hand_str)
            
            # Get combinatorial index
            idx = hand_to_index(card_ints)
            
            # Store score
            score_array[idx] = score
            converted += 1
            
            if converted % 500000 == 0:
                print(f"  Converted {converted}/{len(score_dict)} hands...")
                
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error converting '{hand_str}': {e}")
    
    print(f"Conversion complete: {converted} hands, {errors} errors")
    return score_array


def verify_conversion(score_dict: dict, score_array: np.ndarray, num_samples: int = 1000) -> bool:
    """
    Verify the conversion by checking random samples.
    
    Args:
        score_dict: Original dictionary
        score_array: Converted array
        num_samples: Number of samples to check
        
    Returns:
        True if all samples match
    """
    import random
    
    print(f"\nVerifying conversion with {num_samples} random samples...")
    
    hands = list(score_dict.keys())
    samples = random.sample(hands, min(num_samples, len(hands)))
    
    mismatches = 0
    for hand_str in samples:
        expected = score_dict[hand_str]
        card_ints = hand_str_to_ints(hand_str)
        idx = hand_to_index(card_ints)
        actual = score_array[idx]
        
        if expected != actual:
            mismatches += 1
            if mismatches <= 5:
                print(f"  Mismatch: '{hand_str}' expected {expected}, got {actual}")
    
    if mismatches == 0:
        print("Verification passed: All samples match!")
        return True
    else:
        print(f"Verification failed: {mismatches}/{num_samples} mismatches")
        return False


def save_score_array(score_array: np.ndarray, filepath: str = 'score_array.npy'):
    """Save score array to numpy file."""
    np.save(filepath, score_array)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Saved score array to '{filepath}' ({size_mb:.2f} MB)")


def load_score_array(filepath: str = 'score_array.npy') -> np.ndarray:
    """Load score array from numpy file."""
    return np.load(filepath)


def main():
    # Check if score_array already exists
    if os.path.exists('score_array.npy'):
        print("score_array.npy already exists. Loading and verifying...")
        score_array = load_score_array()
        score_dict = load_score_dict()
        verify_conversion(score_dict, score_array)
        return score_array
    
    # Load existing score dictionary
    print("Loading score_dict.pkl...")
    score_dict = load_score_dict()
    print(f"Loaded {len(score_dict)} hand scores")
    
    # Show a few examples
    print("\nSample entries:")
    for i, (hand, score) in enumerate(list(score_dict.items())[:5]):
        print(f"  '{hand}' -> {score}")
    
    # Convert to array
    score_array = convert_to_score_array(score_dict)
    
    # Verify
    if verify_conversion(score_dict, score_array):
        # Save
        save_score_array(score_array)
    else:
        print("Conversion verification failed! Not saving.")
        return None
    
    return score_array


if __name__ == "__main__":
    main()
