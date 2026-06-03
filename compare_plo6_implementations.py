"""
Compare PLO6 equity calculations between:
1. Legacy Python implementation (multithread_ploequities3.run_parallel_plo6equities)
2. Optimized Numba implementation (optimized_parallel_runner.run_parallel_plo6equities_optimized)

Run 100 hands and identify disagreements.
"""

import random
import time
import numpy as np
from itertools import combinations
import multithread_ploequities3 as mtp
from optimized_parallel_runner import (
    run_parallel_plo6equities_optimized,
    warmup_optimized_evaluator
)


def generate_deck():
    """Generate a standard 52-card deck."""
    suits = 'shdc'
    ranks = '23456789TJQKA'
    return [r + s for r in ranks for s in suits]


def generate_plo6_hands(num_hands=3, seed=None):
    """Generate random PLO6 hands (6 cards each)."""
    if seed is not None:
        random.seed(seed)
    deck = generate_deck()
    random.shuffle(deck)
    hands = []
    for i in range(num_hands):
        hand = deck[i * 6:(i + 1) * 6]
        hands.append(hand)
    return hands


def generate_opponent_range(hands, num_opponents=500):
    """Generate random opponent hands that don't overlap with hero hands."""
    all_hero_cards = set(card for hand in hands for card in hand)
    deck = [c for c in generate_deck() if c not in all_hero_cards]
    
    opponent_hands = []
    for _ in range(num_opponents):
        random.shuffle(deck)
        opp_hand = ''.join(deck[:6])
        opponent_hands.append(opp_hand)
    return opponent_hands


def compare_single_test(test_num, hands, board, opponent_range, trials=5000):
    """Run single comparison test between legacy and optimized."""
    hands_list = [list(h) for h in hands]
    
    # Run legacy
    start = time.time()
    legacy_ind, legacy_pair = mtp.run_parallel_plo6equities(
        hands_list, trials, 4, board, opponent_range
    )
    legacy_time = time.time() - start
    
    # Run optimized
    start = time.time()
    try:
        opt_ind, opt_pair = run_parallel_plo6equities_optimized(
            hands_list, trials, 4, board, opponent_range
        )
        opt_time = time.time() - start
    except Exception as e:
        print(f"  Optimized FAILED: {e}")
        return None
    
    # Compare
    disagreements = {
        'individual': [],
        'pairwise': []
    }
    
    for key in legacy_ind:
        legacy_val = legacy_ind.get(key, 0)
        opt_val = opt_ind.get(key, 0)
        diff = abs(legacy_val - opt_val)
        if diff > 0.02:  # More than 2% difference
            disagreements['individual'].append({
                'hand': key,
                'legacy': legacy_val,
                'optimized': opt_val,
                'diff': diff
            })
    
    for key in legacy_pair:
        legacy_val = legacy_pair.get(key, 0)
        opt_val = opt_pair.get(key, 0)
        diff = abs(legacy_val - opt_val)
        if diff > 0.02:  # More than 2% difference
            disagreements['pairwise'].append({
                'pair': key,
                'legacy': legacy_val,
                'optimized': opt_val,
                'diff': diff
            })
    
    return {
        'test_num': test_num,
        'hands': [''.join(h) for h in hands],
        'board': board,
        'legacy_ind': legacy_ind,
        'opt_ind': opt_ind,
        'legacy_pair': legacy_pair,
        'opt_pair': opt_pair,
        'legacy_time': legacy_time,
        'opt_time': opt_time,
        'disagreements': disagreements
    }


def run_comparison(num_tests=100, trials_per_test=5000):
    """Run comprehensive comparison between legacy and optimized PLO6."""
    print("="*80)
    print("PLO6 EQUITY COMPARISON: Legacy vs Optimized")
    print("="*80)
    
    # Warmup
    print("\n[Warmup] Initializing...")
    mtp.warmup_executor()
    warmup_optimized_evaluator()
    print("[Warmup] Complete!\n")
    
    all_results = []
    total_disagreements_ind = 0
    total_disagreements_pair = 0
    large_disagreements = []
    
    total_legacy_time = 0
    total_opt_time = 0
    
    print(f"Running {num_tests} tests with {trials_per_test} trials each...\n")
    
    for i in range(num_tests):
        seed = i + 1000  # Reproducible seeds
        hands = generate_plo6_hands(num_hands=3, seed=seed)
        
        # Generate opponent range
        opponent_range = generate_opponent_range(hands, num_opponents=500)
        
        # Random board: empty, flop, or turn
        board_options = [[], [], [], None, None]  # 60% preflop, 20% flop, 20% turn
        random.seed(seed)
        board_choice = random.choice([0, 1, 2, 3, 4])
        
        if board_choice == 0 or board_choice == 1 or board_choice == 2:
            board = []
        elif board_choice == 3:
            all_hero_cards = set(card for hand in hands for card in hand)
            deck = [c for c in generate_deck() if c not in all_hero_cards]
            random.shuffle(deck)
            board = deck[:3]  # Flop
        else:
            all_hero_cards = set(card for hand in hands for card in hand)
            deck = [c for c in generate_deck() if c not in all_hero_cards]
            random.shuffle(deck)
            board = deck[:4]  # Turn
        
        result = compare_single_test(i + 1, hands, board, opponent_range, trials_per_test)
        
        if result is None:
            continue
        
        all_results.append(result)
        
        total_legacy_time += result['legacy_time']
        total_opt_time += result['opt_time']
        
        num_ind_disagree = len(result['disagreements']['individual'])
        num_pair_disagree = len(result['disagreements']['pairwise'])
        total_disagreements_ind += num_ind_disagree
        total_disagreements_pair += num_pair_disagree
        
        # Track large disagreements (>5%)
        for d in result['disagreements']['individual']:
            if d['diff'] > 0.05:
                large_disagreements.append({
                    'test': i + 1,
                    'type': 'individual',
                    'hand': d['hand'],
                    'legacy': d['legacy'],
                    'optimized': d['optimized'],
                    'diff': d['diff']
                })
        
        for d in result['disagreements']['pairwise']:
            if d['diff'] > 0.05:
                large_disagreements.append({
                    'test': i + 1,
                    'type': 'pairwise',
                    'pair': d['pair'],
                    'legacy': d['legacy'],
                    'optimized': d['optimized'],
                    'diff': d['diff']
                })
        
        # Print progress
        status = "OK" if num_ind_disagree == 0 and num_pair_disagree == 0 else f"DISAGREE({num_ind_disagree}i,{num_pair_disagree}p)"
        print(f"Test {i+1:3d}: {status:20s} Legacy: {result['legacy_time']:.2f}s, Opt: {result['opt_time']:.2f}s")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nTotal tests: {len(all_results)}")
    print(f"Total individual disagreements (>2%): {total_disagreements_ind}")
    print(f"Total pairwise disagreements (>2%): {total_disagreements_pair}")
    print(f"Large disagreements (>5%): {len(large_disagreements)}")
    
    print(f"\nTiming:")
    print(f"  Total Legacy time: {total_legacy_time:.2f}s")
    print(f"  Total Optimized time: {total_opt_time:.2f}s")
    print(f"  Speedup: {total_legacy_time/max(total_opt_time, 0.001):.2f}x")
    
    if large_disagreements:
        print("\n" + "="*80)
        print("LARGE DISAGREEMENTS (>5%)")
        print("="*80)
        for d in large_disagreements:
            if d['type'] == 'individual':
                print(f"\nTest {d['test']}: Individual {d['hand']}")
            else:
                print(f"\nTest {d['test']}: Pairwise {d['pair']}")
            print(f"  Legacy:    {d['legacy']:.4f} ({d['legacy']*100:.2f}%)")
            print(f"  Optimized: {d['optimized']:.4f} ({d['optimized']*100:.2f}%)")
            print(f"  Diff:      {d['diff']:.4f} ({d['diff']*100:.2f}%)")
    
    # Show sample results for first few tests with disagreements
    tests_with_disagreements = [r for r in all_results if r['disagreements']['individual'] or r['disagreements']['pairwise']]
    
    if tests_with_disagreements:
        print("\n" + "="*80)
        print("DETAILED DISAGREEMENTS (first 10)")
        print("="*80)
        
        for r in tests_with_disagreements[:10]:
            print(f"\n--- Test {r['test_num']} ---")
            print(f"Hands: {r['hands']}")
            print(f"Board: {r['board']}")
            
            print("\nIndividual Equities:")
            for key in r['legacy_ind']:
                legacy = r['legacy_ind'].get(key, 0)
                opt = r['opt_ind'].get(key, 0)
                diff = abs(legacy - opt)
                marker = " ***" if diff > 0.02 else ""
                print(f"  {key}: Legacy={legacy:.4f}, Opt={opt:.4f}, Diff={diff:.4f}{marker}")
            
            print("\nPairwise Equities:")
            for key in r['legacy_pair']:
                legacy = r['legacy_pair'].get(key, 0)
                opt = r['opt_pair'].get(key, 0)
                diff = abs(legacy - opt)
                marker = " ***" if diff > 0.02 else ""
                print(f"  {key}: Legacy={legacy:.4f}, Opt={opt:.4f}, Diff={diff:.4f}{marker}")
    
    # Calculate average differences
    all_ind_diffs = []
    all_pair_diffs = []
    for r in all_results:
        for key in r['legacy_ind']:
            diff = abs(r['legacy_ind'].get(key, 0) - r['opt_ind'].get(key, 0))
            all_ind_diffs.append(diff)
        for key in r['legacy_pair']:
            diff = abs(r['legacy_pair'].get(key, 0) - r['opt_pair'].get(key, 0))
            all_pair_diffs.append(diff)
    
    if all_ind_diffs:
        print(f"\n\nAverage individual diff: {np.mean(all_ind_diffs):.4f} ({np.mean(all_ind_diffs)*100:.2f}%)")
        print(f"Max individual diff: {np.max(all_ind_diffs):.4f} ({np.max(all_ind_diffs)*100:.2f}%)")
    
    if all_pair_diffs:
        print(f"Average pairwise diff: {np.mean(all_pair_diffs):.4f} ({np.mean(all_pair_diffs)*100:.2f}%)")
        print(f"Max pairwise diff: {np.max(all_pair_diffs):.4f} ({np.max(all_pair_diffs)*100:.2f}%)")
    
    return all_results


if __name__ == "__main__":
    results = run_comparison(num_tests=100, trials_per_test=5000)
