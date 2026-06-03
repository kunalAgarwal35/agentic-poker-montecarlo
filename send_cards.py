"""
Send PLO6 hands to the equity calculation API.

First hand is from a specific scenario, the rest can be random.
"""

import itertools
import random
import time
import calling_api


def generate_deck():
    """Generate a standard 52-card deck."""
    ranks = '23456789TJQKA'
    suits = 'shdc'
    return [r + s for r in ranks for s in suits]


def generate_random_plo6_hands(num_hands, exclude_cards=[]):
    """Generate random PLO6 hands (6 cards each), excluding specified cards."""
    deck = generate_deck()
    # Remove excluded cards
    deck = [c for c in deck if c not in exclude_cards]
    random.shuffle(deck)
    
    hands = []
    for i in range(num_hands):
        hand = deck[i*6:(i+1)*6]
        hands.append(''.join(hand))
    
    return hands


def send_plo6_hands(hands, board=[], trials=10000):
    """
    Send PLO6 hands to the API and get individual + pairwise equities.
    
    Args:
        hands: List of hand strings (6 cards each, e.g., 'AsKsQsJsTs9s')
        board: Optional board cards
        trials: Number of Monte Carlo trials
    
    Returns:
        API result with individual and pairwise equities
    """
    print(f"\nSending {len(hands)} PLO6 hands to API...")
    print(f"Hands: {hands}")
    if board:
        print(f"Board: {board}")
    print(f"Trials: {trials}")
    
    t1 = time.time()
    result = calling_api.preflop_multithread_plo6(
        hands, 
        dead_cards=[], 
        total_number_of_trials=trials,
        board=board
    )
    t2 = time.time()
    
    print(f"\nAPI call took: {t2-t1:.2f}s")
    
    if result:
        print("\n" + "="*60)
        print("INDIVIDUAL EQUITIES")
        print("="*60)
        for hand, equity in result.get('individual', {}).items():
            print(f"  {hand}: {equity*100:.2f}%")
        
        print("\n" + "="*60)
        print("PAIRWISE EQUITIES")
        print("="*60)
        for pair, equity in result.get('pairwise', {}).items():
            print(f"  {pair}: {equity*100:.2f}%")
    
    return result


def main():
    # Check health first
    print("Checking API health...")
    try:
        health = calling_api.health_check()
        print(f"API Status: {health}")
    except Exception as e:
        print(f"API health check failed: {e}")
        return
    
    # ==========================================
    # FIRST SCENARIO - From the provided image
    # ==========================================
    print("\n" + "="*60)
    print("SCENARIO 1: Hands from provided image")
    print("="*60)
    
    # Hands from the image:
    # V2356_KAEN (Min Raiser): 2h 7h Td Tc 7c 5h
    # 24108PCE2I_KAEN (Aggressive): Ad Jd 8h 6d 4h 3c
    # CPH2667_KAEN (Passive): 5s Ts Th 8s Qs 9c
    
    hands_from_image = [
        "2h7hTdTc7c5h",   # V2356_KAEN - Min Raiser
        "AdJd8h6d4h3c",   # 24108PCE2I_KAEN - Aggressive
        "5sTsTh8sQs9c",   # CPH2667_KAEN - Passive
    ]
    
    result1 = send_plo6_hands(hands_from_image, trials=10000)
    
    # ==========================================
    # SECOND SCENARIO - Random hands
    # ==========================================
    print("\n\n" + "="*60)
    print("SCENARIO 2: Random PLO6 hands")
    print("="*60)
    
    random_hands = generate_random_plo6_hands(3)
    result2 = send_plo6_hands(random_hands, trials=10000)
    
    # ==========================================
    # THIRD SCENARIO - First hand from image + 2 random
    # ==========================================
    print("\n\n" + "="*60)
    print("SCENARIO 3: First hand from image + 2 random")
    print("="*60)
    
    first_hand = "2h7hTdTc7c5h"  # V2356_KAEN
    excluded = list(first_hand[i:i+2] for i in range(0, len(first_hand), 2))
    
    random_hands_2 = generate_random_plo6_hands(2, exclude_cards=excluded)
    mixed_hands = [first_hand] + random_hands_2
    
    result3 = send_plo6_hands(mixed_hands, trials=10000)
    
    return result1, result2, result3


if __name__ == '__main__':
    main()
