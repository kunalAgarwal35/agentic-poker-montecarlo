"""
Comparison script: Legacy Java API vs New Python PLO5 Equity Calculator

This script compares:
1. Legacy /strength endpoint (Java ProPokerTools)
2. New /plo5python25pct endpoint (Python Monte Carlo)

For 10 different sets of 4 PLO5 hands
"""

import time
import random
import logging
from itertools import combinations
import multi_queries
import multithread_ploequities3 as mtp
from optimized_parallel_runner import (
    run_parallel_plo5equities_25pct_optimized,
    warmup_optimized_evaluator
)

# Initialize
logger = logging.getLogger("comparison_test")
pwe = multi_queries.PlayerWithEquity(logger)


def generate_deck():
    """Generate a standard 52-card deck."""
    suits = 'shdc'
    ranks = '23456789TJQKA'
    return [r + s for r in ranks for s in suits]


def generate_test_hands(num_hands=4, cards_per_hand=5, seed=None):
    """Generate random PLO5 hands."""
    if seed is not None:
        random.seed(seed)
    deck = generate_deck()
    random.shuffle(deck)
    hands = []
    for i in range(num_hands):
        hand = deck[i * cards_per_hand:(i + 1) * cards_per_hand]
        hands.append(hand)
    return hands


def run_legacy_api(hands, board=""):
    """
    Run the legacy Java-based /strength API.
    
    Returns:
        dict: {hand_string: equity_percentage}
    """
    results = {
        'individual': {},
        'timing': 0,
        'pairwise': {}
    }
    
    start = time.time()
    
    # Individual equities - one at a time via legacy API
    for i, hand in enumerate(hands):
        hand_str = ''.join(hand)
        other_hands = hands[:i] + hands[i+1:]
        dead_cards = ''.join([''.join(h) for h in other_hands])
        
        try:
            result = pwe.equity_with_dc(
                hero_cards=hand_str,
                dead_cards=dead_cards,
                p2_range="25%",
                board=board
            )
            results['individual'][hand_str] = result.get(hand_str, 'ERROR')
        except Exception as e:
            results['individual'][hand_str] = f"ERROR: {e}"
    
    # Pairwise equities
    for pair in combinations(hands, 2):
        hand1_str = ''.join(pair[0])
        hand2_str = ''.join(pair[1])
        remaining_hands = [h for h in hands if h not in pair]
        dead_cards = ''.join([''.join(h) for h in remaining_hands])
        
        try:
            pairwise_result = pwe.pairwise_with_dc(
                player_range="25%",
                p2_cards=hand1_str,
                p3_cards=hand2_str,
                deadcards=dead_cards,
                board=board
            )
            key = f"{hand1_str} : {hand2_str}"
            if key in pairwise_result:
                # Combined equity = 100 - range equity (range is the third player)
                range_eq = pairwise_result[key].get("25%", 0)
                if isinstance(range_eq, (int, float)):
                    combined_eq = 100 - range_eq
                    results['pairwise'][f"{hand1_str}_{hand2_str}"] = combined_eq
        except Exception as e:
            results['pairwise'][f"{hand1_str}_{hand2_str}"] = f"ERROR: {e}"
    
    results['timing'] = time.time() - start
    return results


def run_new_python_api(hands, board=[], trials=10000, processes=4, use_optimized=True):
    """
    Run the new Python-based plo5python25pct API.
    
    Returns:
        dict: {'individual': {hand: equity}, 'pairwise': {pair: equity}}
    """
    start = time.time()
    
    # Convert hands to format expected by the function
    hands_list = [list(h) for h in hands]
    
    if use_optimized:
        individual, pairwise = run_parallel_plo5equities_25pct_optimized(
            hands_list, trials, processes, board
        )
    else:
        individual, pairwise = mtp.run_parallel_plo5equities_25pct(
            hands_list, trials, processes, board
        )
    
    timing = time.time() - start
    
    # Convert to percentages for comparison
    individual_pct = {k: v * 100 for k, v in individual.items()}
    pairwise_pct = {k: v * 100 for k, v in pairwise.items()}
    
    return {
        'individual': individual_pct,
        'pairwise': pairwise_pct,
        'timing': timing
    }


def compare_results(legacy, new_python):
    """Compare results from both APIs."""
    comparison = {
        'individual': {},
        'pairwise': {},
        'timing_comparison': {
            'legacy': legacy['timing'],
            'new_python': new_python['timing'],
            'speedup': legacy['timing'] / max(new_python['timing'], 0.001)
        }
    }
    
    # Compare individual equities
    for hand in legacy['individual']:
        legacy_eq = legacy['individual'].get(hand)
        python_eq = new_python['individual'].get(hand)
        
        if isinstance(legacy_eq, (int, float)) and isinstance(python_eq, (int, float)):
            diff = abs(legacy_eq - python_eq)
            comparison['individual'][hand] = {
                'legacy': round(legacy_eq, 2),
                'python': round(python_eq, 2),
                'diff': round(diff, 2)
            }
        else:
            comparison['individual'][hand] = {
                'legacy': legacy_eq,
                'python': python_eq,
                'diff': 'N/A'
            }
    
    # Compare pairwise equities
    for key in new_python['pairwise']:
        legacy_eq = legacy['pairwise'].get(key)
        python_eq = new_python['pairwise'].get(key)
        
        if isinstance(legacy_eq, (int, float)) and isinstance(python_eq, (int, float)):
            diff = abs(legacy_eq - python_eq)
            comparison['pairwise'][key] = {
                'legacy': round(legacy_eq, 2),
                'python': round(python_eq, 2),
                'diff': round(diff, 2)
            }
        else:
            comparison['pairwise'][key] = {
                'legacy': legacy_eq if legacy_eq else 'N/A',
                'python': round(python_eq, 2) if isinstance(python_eq, (int, float)) else python_eq,
                'diff': 'N/A'
            }
    
    return comparison


def print_test_results(test_num, hands, comparison):
    """Print formatted test results."""
    print(f"\n{'='*80}")
    print(f"TEST SET {test_num}")
    print(f"{'='*80}")
    
    print("\nHands:")
    for i, hand in enumerate(hands):
        print(f"  Hand {i+1}: {''.join(hand)}")
    
    print(f"\n{'Individual Equities':^80}")
    print(f"{'Hand':<20} {'Legacy (Java)':<15} {'Python':<15} {'Difference':<15}")
    print("-" * 65)
    
    for hand, data in comparison['individual'].items():
        print(f"{hand:<20} {str(data['legacy']):<15} {str(data['python']):<15} {str(data['diff']):<15}")
    
    print(f"\n{'Pairwise Equities (Combined Equity vs 25% Range)':^80}")
    print(f"{'Pair':<25} {'Legacy (Java)':<15} {'Python':<15} {'Difference':<15}")
    print("-" * 70)
    
    for pair, data in comparison['pairwise'].items():
        short_pair = f"H{hands_index(comparison, pair)}"
        print(f"{pair[:24]:<25} {str(data['legacy']):<15} {str(data['python']):<15} {str(data['diff']):<15}")
    
    print(f"\nTiming:")
    print(f"  Legacy (Java):  {comparison['timing_comparison']['legacy']:.2f}s")
    print(f"  Python:         {comparison['timing_comparison']['new_python']:.2f}s")
    print(f"  Speedup:        {comparison['timing_comparison']['speedup']:.1f}x")


def hands_index(comparison, pair):
    """Helper to get hand indices for a pair."""
    return pair.split('_')


def run_comparison_tests(num_tests=10, include_legacy=True):
    """Run the full comparison test suite."""
    
    print("="*80)
    print("PLO5 EQUITY API COMPARISON: Legacy Java vs New Python")
    print("="*80)
    
    # Warmup
    print("\n[Warmup] Initializing Python evaluator...")
    warmup_optimized_evaluator()
    mtp.warmup_loh25()
    
    if include_legacy:
        print("[Warmup] Initializing Java JVM...")
        try:
            multi_queries.warmup_jvm()
        except:
            print("[Warmup] JVM warmup skipped (may not be available)")
    
    # Predefined test hands for reproducibility (seeds 1-10)
    test_seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    
    all_results = []
    summary_stats = {
        'avg_diff_individual': [],
        'avg_diff_pairwise': [],
        'speedups': []
    }
    
    print("\n" + "="*80)
    print("RUNNING TESTS")
    print("="*80)
    
    for i, seed in enumerate(test_seeds[:num_tests], 1):
        print(f"\n--- Test {i}/{num_tests} (seed={seed}) ---")
        
        # Generate hands
        hands = generate_test_hands(num_hands=4, cards_per_hand=5, seed=seed)
        print(f"Generated hands: {[''.join(h) for h in hands]}")
        
        # Run Python API
        print("Running Python API...")
        python_results = run_new_python_api(hands, board=[], trials=10000, use_optimized=True)
        print(f"  Python completed in {python_results['timing']:.2f}s")
        
        # Run Legacy API (if enabled)
        if include_legacy:
            print("Running Legacy Java API...")
            try:
                legacy_results = run_legacy_api(hands, board="")
                print(f"  Legacy completed in {legacy_results['timing']:.2f}s")
            except Exception as e:
                print(f"  Legacy API failed: {e}")
                legacy_results = {
                    'individual': {k: 'N/A' for k in python_results['individual']},
                    'pairwise': {k: 'N/A' for k in python_results['pairwise']},
                    'timing': 0
                }
        else:
            legacy_results = {
                'individual': {k: 'N/A (skipped)' for k in python_results['individual']},
                'pairwise': {k: 'N/A (skipped)' for k in python_results['pairwise']},
                'timing': 0
            }
        
        # Compare
        comparison = compare_results(legacy_results, python_results)
        all_results.append({
            'test_num': i,
            'hands': hands,
            'comparison': comparison
        })
        
        # Collect stats
        if include_legacy and legacy_results['timing'] > 0:
            diffs = [d['diff'] for d in comparison['individual'].values() if isinstance(d['diff'], (int, float))]
            if diffs:
                summary_stats['avg_diff_individual'].append(sum(diffs) / len(diffs))
            
            pair_diffs = [d['diff'] for d in comparison['pairwise'].values() if isinstance(d['diff'], (int, float))]
            if pair_diffs:
                summary_stats['avg_diff_pairwise'].append(sum(pair_diffs) / len(pair_diffs))
            
            if comparison['timing_comparison']['speedup'] > 0:
                summary_stats['speedups'].append(comparison['timing_comparison']['speedup'])
    
    # Print all results
    for result in all_results:
        print_test_results(result['test_num'], result['hands'], result['comparison'])
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\n### KEY DIFFERENCES ###\n")
    print("1. EQUITY FORMAT:")
    print("   - Legacy: Returns percentage (e.g., 68.97)")
    print("   - Python: Returns decimal (0.6897), converted to % for comparison")
    
    print("\n2. API STRUCTURE:")
    print("   - Legacy: Separate calls for individual (/strength) and pairwise (/pair)")
    print("   - Python: Single call returns both individual AND pairwise")
    
    print("\n3. RANGE HANDLING:")
    print("   - Legacy: String '25%' interpreted by ProPokerTools")
    print("   - Python: Precomputed loh25_plo5.txt with ~16k top 25% hands")
    
    print("\n4. BOARD FORMAT:")
    print("   - Legacy: String (e.g., '2c3c4c')")
    print("   - Python: Array (e.g., ['2c', '3c', '4c'])")
    
    print("\n5. PAIRWISE CALCULATION:")
    print("   - Legacy: Equity that the pair has vs the range (needs calculation)")
    print("   - Python: Direct combined equity when either hand wins")
    
    if summary_stats['avg_diff_individual']:
        avg_ind = sum(summary_stats['avg_diff_individual']) / len(summary_stats['avg_diff_individual'])
        print(f"\nAverage Individual Equity Difference: {avg_ind:.2f}%")
    
    if summary_stats['avg_diff_pairwise']:
        avg_pair = sum(summary_stats['avg_diff_pairwise']) / len(summary_stats['avg_diff_pairwise'])
        print(f"Average Pairwise Equity Difference: {avg_pair:.2f}%")
    
    if summary_stats['speedups']:
        avg_speedup = sum(summary_stats['speedups']) / len(summary_stats['speedups'])
        print(f"Average Speedup: {avg_speedup:.1f}x faster")
    
    return all_results


def run_python_only_demo():
    """Run Python-only demo without legacy Java API."""
    print("="*80)
    print("PLO5 NEW PYTHON API - Demo with 10 Test Sets")
    print("="*80)
    
    # Warmup
    print("\n[Warmup] Initializing Python evaluator...")
    warmup_optimized_evaluator()
    mtp.warmup_loh25()
    
    test_seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    
    print("\n" + "="*80)
    print("NEW PYTHON API OUTPUT FORMAT")
    print("="*80)
    print("""
Request:
POST /plo5python25pct
{
    "cards": ["Hand1", "Hand2", "Hand3", "Hand4"],
    "board": [],
    "trials": 10000,
    "processes": 4
}

Response:
{
    "individual": {
        "Hand1": 0.XXXX,  // equity as decimal (0.0 to 1.0)
        "Hand2": 0.XXXX,
        ...
    },
    "pairwise": {
        "Hand1_Hand2": 0.XXXX,  // combined equity when either wins
        "Hand1_Hand3": 0.XXXX,
        ...
    }
}
""")
    
    for i, seed in enumerate(test_seeds, 1):
        hands = generate_test_hands(num_hands=4, cards_per_hand=5, seed=seed)
        hands_list = [list(h) for h in hands]
        
        start = time.time()
        individual, pairwise = run_parallel_plo5equities_25pct_optimized(
            hands_list, 10000, 4, []
        )
        elapsed = time.time() - start
        
        print(f"\n{'='*80}")
        print(f"TEST SET {i} (seed={seed})")
        print(f"{'='*80}")
        
        print(f"\nRequest cards: {[''.join(h) for h in hands]}")
        print(f"Time: {elapsed:.2f}s")
        
        print("\nResponse:")
        print("{")
        print('  "individual": {')
        for j, (hand, eq) in enumerate(individual.items()):
            comma = "," if j < len(individual) - 1 else ""
            print(f'    "{hand}": {eq:.4f}{comma}')
        print("  },")
        print('  "pairwise": {')
        for j, (pair, eq) in enumerate(pairwise.items()):
            comma = "," if j < len(pairwise) - 1 else ""
            print(f'    "{pair}": {eq:.4f}{comma}')
        print("  }")
        print("}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--python-only":
        # Run without legacy Java API
        run_python_only_demo()
    else:
        # Try full comparison, fall back to Python-only if Java fails
        print("Attempting full comparison (Legacy Java + New Python)...")
        print("Note: If Java API fails, will show Python-only results.\n")
        
        try:
            run_comparison_tests(num_tests=10, include_legacy=True)
        except Exception as e:
            print(f"\nLegacy Java API not available: {e}")
            print("\nRunning Python-only demo instead...\n")
            run_python_only_demo()
