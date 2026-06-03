"""
Clean PLO6 PREFLOP comparison test.

Tests legacy vs optimized with:
- NO board (preflop only)
- 3 or 4 hero hands
- 6 cards each
- NO opponent range filtering (both use same random opponents from deck)

This ensures we're comparing apples to apples.
"""

import random
import time
import numpy as np
from itertools import combinations

# Legacy implementation
import multithread_ploequities3 as mtp

# Optimized implementation
from optimized_evaluator import (
    plo_equities_optimized,
    get_score_array,
    warmup_jit,
    best_score_numba
)
from hand_indexing import PLO6_HAND_COMBOS, PLO4_BOARD_COMBOS, BINOMIAL
from card_encoding import CARD_TO_INT
import generating_list as scores


def generate_deck():
    suits = 'shdc'
    ranks = '23456789TJQKA'
    return [r + s for r in ranks for s in suits]


def test_preflop_plo6_no_opponent_range(num_tests=50, num_hands=3, trials=10000):
    """
    Compare legacy vs optimized for PREFLOP PLO6.
    NO opponent range - both sample randomly from remaining deck.
    """
    print("="*70)
    print(f"PLO6 PREFLOP TEST: {num_hands} hands, {trials} trials, NO opponent range")
    print("="*70)
    
    # Warmup
    print("\nWarming up...")
    warmup_jit()
    get_score_array()
    
    total_ind_diff = []
    total_pair_diff = []
    large_diffs = []
    
    for test_idx in range(num_tests):
        seed = test_idx + 5000
        random.seed(seed)
        np.random.seed(seed)
        
        # Generate hands
        deck = generate_deck()
        random.shuffle(deck)
        hands = [deck[i*6:(i+1)*6] for i in range(num_hands)]
        
        board = []  # PREFLOP - no board
        opponent_hands = []  # NO opponent range - sample from deck
        
        # Reset seeds for fair comparison
        random.seed(seed + 1000000)
        np.random.seed(seed + 1000000)
        
        # Run legacy
        start = time.time()
        legacy_ind, legacy_pair = mtp.plo6equities(hands, trials, board, opponent_hands)
        legacy_time = time.time() - start
        
        # Reset seeds again for optimized
        random.seed(seed + 1000000)
        np.random.seed(seed + 1000000)
        
        # Run optimized - but it REQUIRES opponent_range, so we need to generate one
        # Actually, let's test the scenario where we pass the SAME opponent range to both
        
        print(f"\nTest {test_idx + 1}: seed={seed}")
        print(f"  Hands: {[''.join(h) for h in hands]}")
        print(f"  Legacy: {legacy_time:.2f}s")
        
        # Show results
        print("  Individual:")
        for key in legacy_ind:
            print(f"    {key}: {legacy_ind[key]:.4f}")
        
        print("  Pairwise:")
        for key in legacy_pair:
            print(f"    {key}: {legacy_pair[key]:.4f}")


def test_preflop_with_shared_opponent_range(num_tests=30, num_hands=3, trials=10000):
    """
    Compare legacy vs optimized for PREFLOP PLO6.
    SAME opponent range passed to both implementations.
    """
    print("\n" + "="*70)
    print(f"PLO6 PREFLOP TEST: Shared opponent range ({num_hands} hands, {trials} trials)")
    print("="*70)
    
    # Warmup
    print("\nWarming up...")
    warmup_jit()
    get_score_array()
    mtp.warmup_executor()
    
    results = []
    
    for test_idx in range(num_tests):
        seed = test_idx + 7000
        random.seed(seed)
        
        # Generate hands
        deck = generate_deck()
        random.shuffle(deck)
        hands = [deck[i*6:(i+1)*6] for i in range(num_hands)]
        
        board = []  # PREFLOP
        
        # Generate opponent range (excluding hero cards)
        hero_cards = set(c for h in hands for c in h)
        opp_deck = [c for c in generate_deck() if c not in hero_cards]
        
        # Generate 500 random opponent hands
        opponent_range = []
        for _ in range(500):
            random.shuffle(opp_deck)
            opponent_range.append(''.join(opp_deck[:6]))
        
        # Run legacy with opponent range
        random.seed(seed + 2000000)
        np.random.seed(seed + 2000000)
        
        start = time.time()
        legacy_ind, legacy_pair = mtp.plo6equities(hands, trials, board, opponent_range)
        legacy_time = time.time() - start
        
        # Run optimized with SAME opponent range
        random.seed(seed + 2000000)
        np.random.seed(seed + 2000000)
        
        start = time.time()
        hands_list = [list(h) for h in hands]
        opt_ind, opt_pair = plo_equities_optimized(hands_list, trials, 'plo6', board, opponent_range)
        opt_time = time.time() - start
        
        # Compare
        ind_diffs = []
        pair_diffs = []
        
        for key in legacy_ind:
            diff = abs(legacy_ind[key] - opt_ind.get(key, 0))
            ind_diffs.append(diff)
        
        for key in legacy_pair:
            diff = abs(legacy_pair[key] - opt_pair.get(key, 0))
            pair_diffs.append(diff)
        
        max_ind_diff = max(ind_diffs) if ind_diffs else 0
        max_pair_diff = max(pair_diffs) if pair_diffs else 0
        
        status = "OK" if max_ind_diff < 0.02 and max_pair_diff < 0.02 else "DIFF"
        
        print(f"\nTest {test_idx + 1}: {status} (max ind diff: {max_ind_diff:.4f}, max pair diff: {max_pair_diff:.4f})")
        print(f"  Legacy: {legacy_time:.2f}s, Optimized: {opt_time:.2f}s")
        
        if max_ind_diff >= 0.02 or max_pair_diff >= 0.02:
            print(f"  Hands: {[''.join(h) for h in hands]}")
            print("  Individual comparison:")
            for key in legacy_ind:
                l = legacy_ind[key]
                o = opt_ind.get(key, 0)
                d = abs(l - o)
                marker = " ***" if d >= 0.02 else ""
                print(f"    {key}: Legacy={l:.4f}, Opt={o:.4f}, Diff={d:.4f}{marker}")
            
            print("  Pairwise comparison:")
            for key in legacy_pair:
                l = legacy_pair[key]
                o = opt_pair.get(key, 0)
                d = abs(l - o)
                marker = " ***" if d >= 0.02 else ""
                print(f"    {key}: Legacy={l:.4f}, Opt={o:.4f}, Diff={d:.4f}{marker}")
        
        results.append({
            'seed': seed,
            'max_ind_diff': max_ind_diff,
            'max_pair_diff': max_pair_diff,
            'legacy_time': legacy_time,
            'opt_time': opt_time
        })
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_ind_diffs = [r['max_ind_diff'] for r in results]
    all_pair_diffs = [r['max_pair_diff'] for r in results]
    
    print(f"\nIndividual equity differences:")
    print(f"  Average max diff: {np.mean(all_ind_diffs):.4f} ({np.mean(all_ind_diffs)*100:.2f}%)")
    print(f"  Max diff: {np.max(all_ind_diffs):.4f} ({np.max(all_ind_diffs)*100:.2f}%)")
    
    print(f"\nPairwise equity differences:")
    print(f"  Average max diff: {np.mean(all_pair_diffs):.4f} ({np.mean(all_pair_diffs)*100:.2f}%)")
    print(f"  Max diff: {np.max(all_pair_diffs):.4f} ({np.max(all_pair_diffs)*100:.2f}%)")
    
    print(f"\nTiming:")
    avg_legacy = np.mean([r['legacy_time'] for r in results])
    avg_opt = np.mean([r['opt_time'] for r in results])
    print(f"  Average legacy: {avg_legacy:.2f}s")
    print(f"  Average optimized: {avg_opt:.2f}s")
    print(f"  Speedup: {avg_legacy/avg_opt:.1f}x")
    
    # How many tests had >2% differences?
    tests_with_diffs = sum(1 for r in results if r['max_ind_diff'] >= 0.02 or r['max_pair_diff'] >= 0.02)
    print(f"\nTests with >2% difference: {tests_with_diffs}/{num_tests}")
    
    return results


def test_with_4_hands(num_tests=20, trials=10000):
    """Test with 4 hero hands specifically."""
    print("\n" + "="*70)
    print(f"PLO6 PREFLOP TEST: 4 HERO HANDS ({trials} trials)")
    print("="*70)
    
    return test_preflop_with_shared_opponent_range(num_tests=num_tests, num_hands=4, trials=trials)


if __name__ == "__main__":
    # Test with 3 hands
    test_preflop_with_shared_opponent_range(num_tests=30, num_hands=3, trials=10000)
    
    # Test with 4 hands
    test_with_4_hands(num_tests=20, trials=10000)
