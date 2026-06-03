"""
Comprehensive accuracy validation for the optimized PLO equity evaluator.

This script runs extensive tests to validate that the optimized implementation
produces results that match the original within acceptable Monte Carlo variance.
"""

import numpy as np
from typing import List, Tuple, Dict
import time
import statistics

# Import both implementations
import multithread_ploequities3 as original
from optimized_parallel_runner import (
    run_parallel_plo4equities_25pct_optimized,
    run_parallel_plo5equities_25pct_optimized,
    warmup_optimized_evaluator
)


def expected_variance(n_trials: int, true_equity: float = 0.5) -> float:
    """
    Calculate expected standard deviation for Monte Carlo equity estimate.
    
    For a binomial process, variance = p*(1-p)/n
    """
    return np.sqrt(true_equity * (1 - true_equity) / n_trials)


def run_validation_test(name: str, hands: List[List[str]], board: List[str],
                        plo_type: str, num_trials: int, num_runs: int = 5) -> Dict:
    """
    Run validation test comparing original vs optimized across multiple runs.
    
    Args:
        name: Test name
        hands: List of hands
        board: Board cards
        plo_type: 'plo4' or 'plo5'
        num_trials: Trials per run
        num_runs: Number of runs to average
        
    Returns:
        Dict with test results
    """
    print(f"\n  Testing: {name}")
    
    # Get the appropriate functions
    if plo_type == 'plo4':
        orig_func = original.run_parallel_plo4equities_25pct
        opt_func = run_parallel_plo4equities_25pct_optimized
    else:
        orig_func = original.run_parallel_plo5equities_25pct
        opt_func = run_parallel_plo5equities_25pct_optimized
    
    orig_results = []
    opt_results = []
    orig_times = []
    opt_times = []
    
    for run in range(num_runs):
        # Run original
        start = time.perf_counter()
        orig_ind, orig_pair = orig_func(hands, num_trials, 4, board)
        orig_times.append(time.perf_counter() - start)
        orig_results.append(orig_ind)
        
        # Run optimized
        start = time.perf_counter()
        opt_ind, opt_pair = opt_func(hands, num_trials, 4, board)
        opt_times.append(time.perf_counter() - start)
        opt_results.append(opt_ind)
    
    # Calculate statistics
    result = {
        'name': name,
        'plo_type': plo_type,
        'trials': num_trials,
        'runs': num_runs,
        'orig_time_avg': statistics.mean(orig_times),
        'opt_time_avg': statistics.mean(opt_times),
        'speedup': statistics.mean(orig_times) / statistics.mean(opt_times),
        'hands': {}
    }
    
    # Per-hand analysis
    for key in orig_results[0].keys():
        orig_values = [r[key] for r in orig_results]
        opt_values = [r[key] for r in opt_results]
        
        orig_mean = statistics.mean(orig_values)
        opt_mean = statistics.mean(opt_values)
        orig_std = statistics.stdev(orig_values) if len(orig_values) > 1 else 0
        opt_std = statistics.stdev(opt_values) if len(opt_values) > 1 else 0
        
        diff = abs(orig_mean - opt_mean)
        expected_std = expected_variance(num_trials, (orig_mean + opt_mean) / 2)
        
        result['hands'][key] = {
            'orig_mean': orig_mean,
            'opt_mean': opt_mean,
            'diff': diff,
            'orig_std': orig_std,
            'opt_std': opt_std,
            'expected_std': expected_std,
            'within_2sigma': diff <= 2 * expected_std
        }
        
        status = "OK" if diff <= 0.03 else "WARN"
        print(f"    {key}: orig={orig_mean:.4f}±{orig_std:.4f}, opt={opt_mean:.4f}±{opt_std:.4f}, diff={diff:.4f} [{status}]")
    
    print(f"    Avg speedup: {result['speedup']:.2f}x ({result['orig_time_avg']:.3f}s -> {result['opt_time_avg']:.3f}s)")
    
    return result


def run_comprehensive_validation():
    """Run comprehensive validation suite."""
    print("=" * 70)
    print("COMPREHENSIVE ACCURACY VALIDATION")
    print("=" * 70)
    
    # Initialize
    print("\nInitializing...")
    original.warmup_loh25()
    warmup_optimized_evaluator()
    
    # Test configurations
    test_cases = [
        # PLO4 tests
        {
            'name': 'PLO4 Premium vs Drawing hand (preflop)',
            'hands': [['As', 'Ah', 'Ks', 'Kh'], ['Jd', 'Tc', '9d', '8c']],
            'board': [],
            'plo_type': 'plo4'
        },
        {
            'name': 'PLO4 Premium vs Premium (preflop)',
            'hands': [['As', 'Ah', 'Ks', 'Kh'], ['Qs', 'Qh', 'Jh', 'Ts']],
            'board': [],
            'plo_type': 'plo4'
        },
        {
            'name': 'PLO4 Three-way (preflop)',
            'hands': [['As', 'Ah', 'Ks', 'Kh'], ['Jd', 'Tc', '9d', '8c'], ['Qs', 'Qh', 'Jh', 'Ts']],
            'board': [],
            'plo_type': 'plo4'
        },
        {
            'name': 'PLO4 Premium vs Drawing (flop)',
            'hands': [['As', 'Ah', 'Ks', 'Kh'], ['Jd', 'Tc', '9d', '8c']],
            'board': ['2c', '3c', '4c'],
            'plo_type': 'plo4'
        },
        {
            'name': 'PLO4 Premium vs Drawing (turn)',
            'hands': [['As', 'Ah', 'Ks', 'Kh'], ['Jd', 'Tc', '9d', '8c']],
            'board': ['2c', '3c', '4c', '7h'],
            'plo_type': 'plo4'
        },
        {
            'name': 'PLO4 Rundown vs Pairs (preflop)',
            'hands': [['Td', '9d', '8c', '7c'], ['Kh', 'Ks', 'Qh', 'Qs']],
            'board': [],
            'plo_type': 'plo4'
        },
        # PLO5 tests
        {
            'name': 'PLO5 Premium vs Drawing (preflop)',
            'hands': [['As', 'Ah', 'Ks', 'Kh', 'Qs'], ['Jd', 'Tc', '9d', '8c', '7d']],
            'board': [],
            'plo_type': 'plo5'
        },
        {
            'name': 'PLO5 Three-way (preflop)',
            'hands': [['As', 'Ah', 'Ks', 'Kh', 'Qs'], ['Jd', 'Tc', '9d', '8c', '7d'], ['Qc', 'Qd', 'Jc', 'Td', '9s']],
            'board': [],
            'plo_type': 'plo5'
        },
        {
            'name': 'PLO5 Premium vs Drawing (flop)',
            'hands': [['As', 'Ah', 'Ks', 'Kh', 'Qs'], ['Jd', 'Tc', '9d', '8c', '7d']],
            'board': ['2c', '3c', '4c'],
            'plo_type': 'plo5'
        },
    ]
    
    results = []
    
    # Run tests with increasing trial counts
    trial_counts = [5000, 10000, 20000]
    
    for trials in trial_counts:
        print(f"\n{'='*70}")
        print(f"RUNNING TESTS WITH {trials} TRIALS")
        print(f"{'='*70}")
        
        for tc in test_cases:
            result = run_validation_test(
                tc['name'], tc['hands'], tc['board'], tc['plo_type'],
                num_trials=trials, num_runs=3
            )
            results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    # Check all results
    all_passed = True
    total_speedup = []
    max_diff = 0
    
    for r in results:
        for hand_key, hand_data in r['hands'].items():
            if hand_data['diff'] > 0.03:  # 3% tolerance
                print(f"WARNING: {r['name']} - {hand_key}: diff={hand_data['diff']:.4f}")
                all_passed = False
            max_diff = max(max_diff, hand_data['diff'])
        total_speedup.append(r['speedup'])
    
    print(f"\nTotal tests run: {len(results)}")
    print(f"Average speedup: {statistics.mean(total_speedup):.2f}x")
    print(f"Min speedup: {min(total_speedup):.2f}x")
    print(f"Max speedup: {max(total_speedup):.2f}x")
    print(f"Max equity difference: {max_diff:.4f}")
    print(f"All results within tolerance: {'YES' if all_passed else 'NO'}")
    
    return results, all_passed


def run_consistency_test():
    """Test that repeated runs produce consistent results (no memory leaks, etc.)."""
    print("\n" + "=" * 70)
    print("CONSISTENCY TEST (10 consecutive runs)")
    print("=" * 70)
    
    hands = [['As', 'Ah', 'Ks', 'Kh'], ['Jd', 'Tc', '9d', '8c']]
    board = []
    trials = 10000
    
    times = []
    equities = []
    
    for i in range(10):
        start = time.perf_counter()
        ind, pair = run_parallel_plo4equities_25pct_optimized(hands, trials, 4, board)
        elapsed = time.perf_counter() - start
        
        times.append(elapsed)
        equities.append(ind['AsAhKsKh'])
        
        print(f"  Run {i+1}: {elapsed:.3f}s, equity={ind['AsAhKsKh']:.4f}")
    
    print(f"\n  Time: mean={statistics.mean(times):.3f}s, std={statistics.stdev(times):.4f}s")
    print(f"  Equity: mean={statistics.mean(equities):.4f}, std={statistics.stdev(equities):.4f}")
    
    # Check for memory leaks or performance degradation
    first_half_avg = statistics.mean(times[:5])
    second_half_avg = statistics.mean(times[5:])
    
    if second_half_avg > first_half_avg * 1.2:
        print("  WARNING: Performance degradation detected!")
        return False
    else:
        print("  No performance degradation detected - PASS")
        return True


if __name__ == "__main__":
    # Run comprehensive validation
    results, all_passed = run_comprehensive_validation()
    
    # Run consistency test
    consistent = run_consistency_test()
    
    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    if all_passed and consistent:
        print("ALL TESTS PASSED - Optimized evaluator is accurate and reliable!")
        print("\nRecommendation: Safe to deploy to production")
    else:
        print("SOME TESTS FAILED - Review warnings above")
        print("\nRecommendation: Investigate failures before deployment")
