"""
Benchmark script to compare original vs optimized PLO equity evaluators.

This script:
1. Loads the top 25% opponent ranges
2. Runs multiple test cases with different configurations
3. Compares speed and accuracy between implementations
4. Reports detailed results
"""

import time
import numpy as np
from typing import List, Tuple, Dict
import sys

# Import original implementation
import multithread_ploequities3 as original
import generating_list as scores

# Import optimized implementation
from optimized_evaluator import (
    plo4_equities_25pct_optimized,
    plo5_equities_25pct_optimized,
    plo6_equities_25pct_optimized,
    warmup_jit,
    get_score_array
)
from card_encoding import hand_str_to_ints


def load_opponent_ranges():
    """Load the top 25% opponent ranges."""
    print("Loading opponent ranges...")
    
    # Force load of LOH25 ranges
    original.warmup_loh25()
    
    loh25_plo4 = original.load_loh25_plo4()
    loh25_plo5 = original.load_loh25_plo5()
    
    print(f"  PLO4 range: {len(loh25_plo4)} hands")
    print(f"  PLO5 range: {len(loh25_plo5)} hands")
    
    return loh25_plo4, loh25_plo5


def filter_opponent_range(opponent_range: List[str], dead_cards: List[str]) -> List[str]:
    """Filter opponent range to exclude hands with dead cards."""
    dead_set = set(dead_cards)
    filtered = []
    for hand in opponent_range:
        cards = [hand[i:i+2] for i in range(0, len(hand), 2)]
        if not any(c in dead_set for c in cards):
            filtered.append(hand)
    return filtered


def run_original_plo4(hands: List[List[str]], num_trials: int, board: List[str], 
                       opponent_range: List[str]) -> Tuple[Dict, Dict, float]:
    """Run original PLO4 implementation and return results with timing."""
    # Filter opponent range
    dead_cards = [c for h in hands for c in h] + board
    filtered_range = filter_opponent_range(opponent_range, dead_cards)
    
    start = time.perf_counter()
    individual, pairwise = original.plo4equities(hands, num_trials, board, filtered_range)
    elapsed = time.perf_counter() - start
    
    return individual, pairwise, elapsed


def run_original_plo5(hands: List[List[str]], num_trials: int, board: List[str], 
                       opponent_range: List[str]) -> Tuple[Dict, Dict, float]:
    """Run original PLO5 implementation and return results with timing."""
    # Filter opponent range
    dead_cards = [c for h in hands for c in h] + board
    filtered_range = filter_opponent_range(opponent_range, dead_cards)
    
    start = time.perf_counter()
    individual, pairwise = original.plo5equities(hands, num_trials, board, filtered_range)
    elapsed = time.perf_counter() - start
    
    return individual, pairwise, elapsed


def run_optimized_plo4(hands: List[List[str]], num_trials: int, board: List[str],
                        opponent_range: List[str]) -> Tuple[Dict, Dict, float]:
    """Run optimized PLO4 implementation and return results with timing."""
    # Filter opponent range
    dead_cards = [c for h in hands for c in h] + board
    filtered_range = filter_opponent_range(opponent_range, dead_cards)
    
    start = time.perf_counter()
    individual, pairwise = plo4_equities_25pct_optimized(hands, num_trials, board, filtered_range)
    elapsed = time.perf_counter() - start
    
    return individual, pairwise, elapsed


def run_optimized_plo5(hands: List[List[str]], num_trials: int, board: List[str],
                        opponent_range: List[str]) -> Tuple[Dict, Dict, float]:
    """Run optimized PLO5 implementation and return results with timing."""
    # Filter opponent range
    dead_cards = [c for h in hands for c in h] + board
    filtered_range = filter_opponent_range(opponent_range, dead_cards)
    
    start = time.perf_counter()
    individual, pairwise = plo5_equities_25pct_optimized(hands, num_trials, board, filtered_range)
    elapsed = time.perf_counter() - start
    
    return individual, pairwise, elapsed


def compare_results(original_ind: Dict, original_pair: Dict,
                    optimized_ind: Dict, optimized_pair: Dict,
                    tolerance: float = 0.03) -> Tuple[bool, float, float]:
    """
    Compare results from original and optimized implementations.
    
    Returns:
        Tuple of (all_within_tolerance, max_ind_diff, max_pair_diff)
    """
    max_ind_diff = 0.0
    max_pair_diff = 0.0
    
    for key in original_ind:
        diff = abs(original_ind[key] - optimized_ind[key])
        max_ind_diff = max(max_ind_diff, diff)
    
    for key in original_pair:
        diff = abs(original_pair[key] - optimized_pair[key])
        max_pair_diff = max(max_pair_diff, diff)
    
    all_within = max_ind_diff <= tolerance and max_pair_diff <= tolerance
    return all_within, max_ind_diff, max_pair_diff


def run_benchmark_suite():
    """Run comprehensive benchmark suite."""
    print("=" * 70)
    print("PLO EQUITY EVALUATOR BENCHMARK")
    print("=" * 70)
    
    # Load opponent ranges
    loh25_plo4, loh25_plo5 = load_opponent_ranges()
    
    # Warmup JIT
    print("\nWarming up JIT compilation...")
    warmup_jit()
    
    # Test cases
    test_cases = [
        # PLO4 test cases
        {
            'name': 'PLO4 - 2 hands, preflop',
            'type': 'plo4',
            'hands': [['As', 'Ah', 'Ks', 'Kh'], ['Jd', 'Tc', '9d', '8c']],
            'board': [],
            'trials': [1000, 5000, 10000],
            'range': loh25_plo4
        },
        {
            'name': 'PLO4 - 2 hands, flop',
            'type': 'plo4',
            'hands': [['As', 'Ah', 'Ks', 'Kh'], ['Jd', 'Tc', '9d', '8c']],
            'board': ['2c', '3c', '4c'],
            'trials': [1000, 5000, 10000],
            'range': loh25_plo4
        },
        {
            'name': 'PLO4 - 3 hands, preflop',
            'type': 'plo4',
            'hands': [['As', 'Ah', 'Ks', 'Kh'], ['Jd', 'Tc', '9d', '8c'], ['Qs', 'Qh', 'Jh', 'Ts']],
            'board': [],
            'trials': [1000, 5000, 10000],
            'range': loh25_plo4
        },
        # PLO5 test cases
        {
            'name': 'PLO5 - 2 hands, preflop',
            'type': 'plo5',
            'hands': [['As', 'Ah', 'Ks', 'Kh', 'Qs'], ['Jd', 'Tc', '9d', '8c', '7d']],
            'board': [],
            'trials': [1000, 5000, 10000],
            'range': loh25_plo5
        },
        {
            'name': 'PLO5 - 2 hands, flop',
            'type': 'plo5',
            'hands': [['As', 'Ah', 'Ks', 'Kh', 'Qs'], ['Jd', 'Tc', '9d', '8c', '7d']],
            'board': ['2c', '3c', '4c'],
            'trials': [1000, 5000, 10000],
            'range': loh25_plo5
        },
    ]
    
    results = []
    
    for tc in test_cases:
        print(f"\n{'='*70}")
        print(f"Test: {tc['name']}")
        print(f"{'='*70}")
        
        for num_trials in tc['trials']:
            print(f"\n  Trials: {num_trials}")
            
            # Run original
            if tc['type'] == 'plo4':
                orig_ind, orig_pair, orig_time = run_original_plo4(
                    tc['hands'], num_trials, tc['board'], tc['range']
                )
            else:
                orig_ind, orig_pair, orig_time = run_original_plo5(
                    tc['hands'], num_trials, tc['board'], tc['range']
                )
            
            # Run optimized
            if tc['type'] == 'plo4':
                opt_ind, opt_pair, opt_time = run_optimized_plo4(
                    tc['hands'], num_trials, tc['board'], tc['range']
                )
            else:
                opt_ind, opt_pair, opt_time = run_optimized_plo5(
                    tc['hands'], num_trials, tc['board'], tc['range']
                )
            
            # Compare
            within_tol, max_ind_diff, max_pair_diff = compare_results(
                orig_ind, orig_pair, opt_ind, opt_pair
            )
            
            speedup = orig_time / opt_time if opt_time > 0 else 0
            
            print(f"    Original:  {orig_time:.3f}s")
            print(f"    Optimized: {opt_time:.3f}s")
            print(f"    Speedup:   {speedup:.2f}x")
            print(f"    Max individual diff: {max_ind_diff:.4f}")
            print(f"    Max pairwise diff:   {max_pair_diff:.4f}")
            print(f"    Accuracy OK: {'YES' if within_tol else 'NO'}")
            
            # Show sample results
            if num_trials == tc['trials'][-1]:  # Only for largest trial count
                print(f"\n    Sample results (individual):")
                for key in orig_ind:
                    print(f"      {key}: orig={orig_ind[key]:.4f}, opt={opt_ind[key]:.4f}")
            
            results.append({
                'name': tc['name'],
                'trials': num_trials,
                'orig_time': orig_time,
                'opt_time': opt_time,
                'speedup': speedup,
                'max_ind_diff': max_ind_diff,
                'max_pair_diff': max_pair_diff,
                'accuracy_ok': within_tol
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    avg_speedup = np.mean([r['speedup'] for r in results])
    min_speedup = min([r['speedup'] for r in results])
    max_speedup = max([r['speedup'] for r in results])
    all_accurate = all([r['accuracy_ok'] for r in results])
    
    print(f"\nAverage speedup: {avg_speedup:.2f}x")
    print(f"Min speedup:     {min_speedup:.2f}x")
    print(f"Max speedup:     {max_speedup:.2f}x")
    print(f"All results within tolerance: {'YES' if all_accurate else 'NO'}")
    
    # Detailed table
    print("\n" + "-" * 70)
    print(f"{'Test':<35} {'Trials':<8} {'Orig':<8} {'Opt':<8} {'Speedup':<8} {'Acc':<5}")
    print("-" * 70)
    for r in results:
        name = r['name'][:35]
        print(f"{name:<35} {r['trials']:<8} {r['orig_time']:<8.3f} {r['opt_time']:<8.3f} {r['speedup']:<8.2f} {'OK' if r['accuracy_ok'] else 'FAIL':<5}")
    
    return results


def run_high_trial_benchmark():
    """Run benchmark with higher trial counts to measure sustained performance."""
    print("\n" + "=" * 70)
    print("HIGH TRIAL COUNT BENCHMARK")
    print("=" * 70)
    
    loh25_plo4, loh25_plo5 = load_opponent_ranges()
    
    # Warmup
    warmup_jit()
    
    # Test with 30,000 trials (production-like)
    hands = [['As', 'Ah', 'Ks', 'Kh'], ['Jd', 'Tc', '9d', '8c']]
    board = []
    num_trials = 30000
    
    print(f"\nPLO4 with {num_trials} trials:")
    
    # Original
    orig_ind, orig_pair, orig_time = run_original_plo4(hands, num_trials, board, loh25_plo4)
    print(f"  Original:  {orig_time:.3f}s")
    
    # Optimized
    opt_ind, opt_pair, opt_time = run_optimized_plo4(hands, num_trials, board, loh25_plo4)
    print(f"  Optimized: {opt_time:.3f}s")
    
    speedup = orig_time / opt_time
    print(f"  Speedup:   {speedup:.2f}x")
    
    # Verify accuracy
    within_tol, max_ind, max_pair = compare_results(orig_ind, orig_pair, opt_ind, opt_pair)
    print(f"  Accuracy:  {'OK' if within_tol else 'FAIL'} (max diff: {max(max_ind, max_pair):.4f})")
    
    print(f"\n  Results:")
    for key in orig_ind:
        print(f"    {key}: orig={orig_ind[key]:.4f}, opt={opt_ind[key]:.4f}")


if __name__ == "__main__":
    # Run full benchmark suite
    results = run_benchmark_suite()
    
    # Run high trial benchmark
    run_high_trial_benchmark()
