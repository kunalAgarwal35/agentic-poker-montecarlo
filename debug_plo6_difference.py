"""
Debug script to find the exact cause of PLO6 equity differences.

Tests:
1. Preflop only (no board) - eliminates board card filtering issue
2. Same random seed for fair comparison
3. Direct function comparison (not through parallel wrapper)
"""

import random
import time
import numpy as np
from itertools import combinations
import multithread_ploequities3 as mtp
from optimized_evaluator import (
    plo_equities_optimized,
    get_score_array,
    warmup_jit,
    best_score_numba,
    run_monte_carlo_numba
)
from hand_indexing import PLO6_HAND_COMBOS, PLO6_BOARD_COMBOS, PLO4_BOARD_COMBOS, BINOMIAL
from card_encoding import CARD_TO_INT, hand_list_to_ints, hand_str_to_ints
import generating_list as scores


def generate_deck():
    suits = 'shdc'
    ranks = '23456789TJQKA'
    return [r + s for r in ranks for s in suits]


def test_score_lookup_consistency():
    """Test that score lookups match between legacy dict and optimized array."""
    print("="*60)
    print("TEST 1: Score lookup consistency")
    print("="*60)
    
    score_array = get_score_array()
    
    # Test 100 random 5-card hands
    deck = generate_deck()
    mismatches = 0
    
    for _ in range(100):
        random.shuffle(deck)
        hand = deck[:5]
        
        # Legacy lookup
        sorted_hand = ''.join(sorted(hand))
        legacy_score = scores.score_dict.get(sorted_hand, -1)
        
        # Optimized lookup
        card_ints = np.array([CARD_TO_INT[c] for c in hand], dtype=np.int32)
        sorted_ints = np.sort(card_ints)
        from hand_indexing import hand_to_index
        idx = hand_to_index(sorted_ints)
        opt_score = score_array[idx]
        
        if legacy_score != opt_score:
            mismatches += 1
            print(f"  MISMATCH: {hand}")
            print(f"    sorted_hand: {sorted_hand}")
            print(f"    legacy: {legacy_score}, optimized: {opt_score}")
    
    if mismatches == 0:
        print("  All 100 random hands match!")
    else:
        print(f"  {mismatches} mismatches found!")
    
    return mismatches == 0


def test_best_hand_evaluation():
    """Test that best hand on board matches between implementations."""
    print("\n" + "="*60)
    print("TEST 2: Best hand on board evaluation")
    print("="*60)
    
    score_array = get_score_array()
    
    # Test 100 random PLO6 hand + board combinations
    deck = generate_deck()
    mismatches = 0
    
    for i in range(100):
        random.shuffle(deck)
        hand = deck[:6]  # PLO6 hand
        board = deck[6:11]  # 5-card board
        
        # Legacy
        legacy_score = mtp.best_score_of_hand_on_board(hand, board, scores.score_dict)
        
        # Optimized
        hand_ints = np.array([CARD_TO_INT[c] for c in hand], dtype=np.int32)
        board_ints = np.array([CARD_TO_INT[c] for c in board], dtype=np.int32)
        opt_score = best_score_numba(hand_ints, board_ints, PLO6_HAND_COMBOS, PLO4_BOARD_COMBOS, 
                                      score_array, BINOMIAL)
        
        if legacy_score != opt_score:
            mismatches += 1
            if mismatches <= 5:
                print(f"  MISMATCH #{i}:")
                print(f"    Hand: {hand}")
                print(f"    Board: {board}")
                print(f"    Legacy: {legacy_score}, Optimized: {opt_score}")
    
    if mismatches == 0:
        print("  All 100 hands match!")
    else:
        print(f"  {mismatches} mismatches found!")
    
    return mismatches == 0


def test_mc_equity_preflop():
    """Test Monte Carlo equity calculation PREFLOP (no board cards)."""
    print("\n" + "="*60)
    print("TEST 3: Monte Carlo equity (PREFLOP, same opponent range)")
    print("="*60)
    
    random.seed(42)
    np.random.seed(42)
    
    # Generate hands
    deck = generate_deck()
    random.shuffle(deck)
    hands = [deck[i*6:(i+1)*6] for i in range(3)]
    
    # Generate opponent range (excluding hero cards only)
    hero_cards = set(c for h in hands for c in h)
    opp_deck = [c for c in generate_deck() if c not in hero_cards]
    
    opponent_range = []
    for _ in range(500):
        random.shuffle(opp_deck)
        opponent_range.append(''.join(opp_deck[:6]))
    
    board = []
    trials = 10000
    
    print(f"  Hands: {[''.join(h) for h in hands]}")
    print(f"  Opponent range size: {len(opponent_range)}")
    
    # Reset random seeds for fair comparison
    random.seed(123)
    np.random.seed(123)
    
    # Legacy
    start = time.time()
    legacy_ind, legacy_pair = mtp.plo6equities(hands, trials, board, opponent_range)
    legacy_time = time.time() - start
    
    # Reset seeds again
    random.seed(123)
    np.random.seed(123)
    
    # Optimized (via plo_equities_optimized, NOT run_parallel which filters)
    start = time.time()
    hands_list = [list(h) for h in hands]
    opt_ind, opt_pair = plo_equities_optimized(hands_list, trials, 'plo6', board, opponent_range)
    opt_time = time.time() - start
    
    print(f"\n  Legacy time: {legacy_time:.2f}s")
    print(f"  Optimized time: {opt_time:.2f}s")
    
    print("\n  Individual Equities:")
    for key in legacy_ind:
        legacy = legacy_ind[key]
        opt = opt_ind.get(key, 0)
        diff = abs(legacy - opt)
        marker = " ***" if diff > 0.02 else ""
        print(f"    {key}: Legacy={legacy:.4f}, Opt={opt:.4f}, Diff={diff:.4f}{marker}")
    
    print("\n  Pairwise Equities:")
    for key in legacy_pair:
        legacy = legacy_pair[key]
        opt = opt_pair.get(key, 0)
        diff = abs(legacy - opt)
        marker = " ***" if diff > 0.02 else ""
        print(f"    {key}: Legacy={legacy:.4f}, Opt={opt:.4f}, Diff={diff:.4f}{marker}")


def test_single_trial_comparison():
    """Compare a single trial step by step."""
    print("\n" + "="*60)
    print("TEST 4: Single trial step-by-step comparison")
    print("="*60)
    
    score_array = get_score_array()
    
    random.seed(42)
    np.random.seed(42)
    
    # Setup
    deck = generate_deck()
    random.shuffle(deck)
    hands = [deck[i*6:(i+1)*6] for i in range(3)]
    
    hero_cards = set(c for h in hands for c in h)
    opp_deck = [c for c in generate_deck() if c not in hero_cards]
    random.shuffle(opp_deck)
    opponent_hand = opp_deck[:6]
    
    # Remaining deck for board
    all_dead = hero_cards | set(opponent_hand)
    remaining_deck = [c for c in generate_deck() if c not in all_dead]
    random.shuffle(remaining_deck)
    board = remaining_deck[:5]
    
    print(f"  Hero hands: {[''.join(h) for h in hands]}")
    print(f"  Opponent: {opponent_hand}")
    print(f"  Board: {board}")
    
    # Evaluate with legacy
    print("\n  Legacy evaluation:")
    opp_score_legacy = mtp.best_score_of_hand_on_board(opponent_hand, board, scores.score_dict)
    print(f"    Opponent score: {opp_score_legacy}")
    for h in hands:
        score = mtp.best_score_of_hand_on_board(h, board, scores.score_dict)
        result = "WIN" if score > opp_score_legacy else ("TIE" if score == opp_score_legacy else "LOSE")
        print(f"    {''.join(h)}: {score} ({result})")
    
    # Evaluate with optimized
    print("\n  Optimized evaluation:")
    board_ints = np.array([CARD_TO_INT[c] for c in board], dtype=np.int32)
    opp_ints = np.array([CARD_TO_INT[c] for c in opponent_hand], dtype=np.int32)
    opp_score_opt = best_score_numba(opp_ints, board_ints, PLO6_HAND_COMBOS, PLO4_BOARD_COMBOS, 
                                      score_array, BINOMIAL)
    print(f"    Opponent score: {opp_score_opt}")
    for h in hands:
        hand_ints = np.array([CARD_TO_INT[c] for c in h], dtype=np.int32)
        score = best_score_numba(hand_ints, board_ints, PLO6_HAND_COMBOS, PLO4_BOARD_COMBOS, 
                                  score_array, BINOMIAL)
        result = "WIN" if score > opp_score_opt else ("TIE" if score == opp_score_opt else "LOSE")
        print(f"    {''.join(h)}: {score} ({result})")
    
    if opp_score_legacy != opp_score_opt:
        print("\n  *** OPPONENT SCORE MISMATCH! ***")
    else:
        print("\n  Opponent scores match!")


def test_opponent_range_filtering_difference():
    """Test specifically the effect of opponent range filtering."""
    print("\n" + "="*60)
    print("TEST 5: Opponent range filtering difference (with board)")
    print("="*60)
    
    random.seed(42)
    
    # Generate hands and board
    deck = generate_deck()
    random.shuffle(deck)
    hands = [deck[i*6:(i+1)*6] for i in range(3)]
    board = deck[18:21]  # Flop
    
    # Generate opponent range BEFORE board (contains board cards)
    hero_cards = set(c for h in hands for c in h)
    opp_deck = [c for c in generate_deck() if c not in hero_cards]
    
    opponent_range = []
    for _ in range(500):
        random.shuffle(opp_deck)
        opponent_range.append(''.join(opp_deck[:6]))
    
    # Count how many opponent hands contain board cards
    board_set = set(board)
    hands_with_board_cards = 0
    for opp in opponent_range:
        opp_cards = [opp[i:i+2] for i in range(0, len(opp), 2)]
        if any(c in board_set for c in opp_cards):
            hands_with_board_cards += 1
    
    print(f"  Hands: {[''.join(h) for h in hands]}")
    print(f"  Board: {board}")
    print(f"  Opponent range size: {len(opponent_range)}")
    print(f"  Hands containing board cards: {hands_with_board_cards}")
    
    # Filter opponent range (like optimized does)
    dead_cards = [c for h in hands for c in h] + board
    dead_set = set(dead_cards)
    filtered_range = [h for h in opponent_range 
                      if not any(h[i:i+2] in dead_set for i in range(0, len(h), 2))]
    
    print(f"  Filtered range size: {len(filtered_range)}")
    print(f"  Hands removed by filtering: {len(opponent_range) - len(filtered_range)}")
    
    if hands_with_board_cards > 0:
        print("\n  This explains the difference!")
        print("  Legacy uses invalid hands (containing board cards)")
        print("  Optimized filters them out")


if __name__ == "__main__":
    print("PLO6 Implementation Debugging")
    print("="*60)
    
    # Warmup
    print("\nWarming up...")
    warmup_jit()
    get_score_array()
    print()
    
    # Run tests
    test_score_lookup_consistency()
    test_best_hand_evaluation()
    test_single_trial_comparison()
    test_mc_equity_preflop()
    test_opponent_range_filtering_difference()
