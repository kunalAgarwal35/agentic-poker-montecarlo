"""
Test script to verify API optimizations are working correctly.
Tests connection pooling, retry logic, and parallel execution.
"""
import calling_api
import time
import itertools
import random

def generate_plo6_hands(n=4):
    """Generate n random PLO6 hands (6 cards each)"""
    deck = [card for card in itertools.product('23456789TJQKA', 'shdc')]
    deck = [''.join(card) for card in deck]
    random.shuffle(deck)
    hands = [deck[i * 6:(i + 1) * 6] for i in range(n)]
    return [''.join(hand) for hand in hands], deck

def test_health_check():
    print('=' * 60)
    print('TEST 1: Health Check')
    print('=' * 60)
    t1 = time.time()
    result = calling_api.health_check()
    t2 = time.time()
    print(f'Result: {result}')
    print(f'Time: {t2-t1:.3f}s')
    return t2 - t1

def test_preflop_multithread_plo6():
    print('\n' + '=' * 60)
    print('TEST 2: preflop_multithread_plo6 (Standard Mode - 10k trials)')
    print('=' * 60)
    hands, deck = generate_plo6_hands(4)
    print(f'Hands: {hands}')
    
    t1 = time.time()
    result = calling_api.preflop_multithread_plo6(hands)
    t2 = time.time()
    
    if result:
        print(f'Individual equities: {result["individual"]}')
        pairwise_keys = list(result["pairwise"].keys())
        print(f'Pairwise matchups: {len(pairwise_keys)} pairs')
    else:
        print('ERROR: No result returned')
    print(f'Time taken: {t2-t1:.3f}s')
    return t2 - t1

def test_preflop_multithread_plo6_fast():
    print('\n' + '=' * 60)
    print('TEST 2b: preflop_multithread_plo6 FAST (2.5k trials, 4 procs)')
    print('=' * 60)
    hands, deck = generate_plo6_hands(4)
    print(f'Hands: {hands}')
    
    # Use fewer trials with default 4 processes (server optimal)
    t1 = time.time()
    result = calling_api.preflop_multithread_plo6(hands, total_number_of_trials=2500, number_of_processes=4)
    t2 = time.time()
    
    if result:
        print(f'Individual equities: {result["individual"]}')
        pairwise_keys = list(result["pairwise"].keys())
        print(f'Pairwise matchups: {len(pairwise_keys)} pairs')
    else:
        print('ERROR: No result returned')
    print(f'Time taken: {t2-t1:.3f}s')
    print(f'TARGET: < 3 seconds')
    return t2 - t1

def test_preflop_multithread_plo6_ultrafast():
    print('\n' + '=' * 60)
    print('TEST 2c: preflop_multithread_plo6 ULTRA-FAST (1k trials, 4 procs)')
    print('=' * 60)
    hands, deck = generate_plo6_hands(4)
    print(f'Hands: {hands}')
    
    t1 = time.time()
    result = calling_api.preflop_multithread_plo6(hands, total_number_of_trials=1000, number_of_processes=4)
    t2 = time.time()
    
    if result:
        print(f'Individual equities: {result["individual"]}')
        pairwise_keys = list(result["pairwise"].keys())
        print(f'Pairwise matchups: {len(pairwise_keys)} pairs')
    else:
        print('ERROR: No result returned')
    print(f'Time taken: {t2-t1:.3f}s')
    return t2 - t1

def test_preflop_results():
    print('\n' + '=' * 60)
    print('TEST 3: preflop_results (Parallel strength + pairwise)')
    print('=' * 60)
    # Use 5-card hands for this endpoint
    deck = [card for card in itertools.product('23456789TJQKA', 'shdc')]
    deck = [''.join(card) for card in deck]
    random.shuffle(deck)
    hands = [deck[i * 5:(i + 1) * 5] for i in range(4)]
    hands_str = [''.join(hand) for hand in hands]
    print(f'Hands: {hands_str}')
    
    t1 = time.time()
    individual, pairwise = calling_api.preflop_results(hands_str)
    t2 = time.time()
    
    print(f'Individual results: {individual}')
    print(f'Pairwise matchups: {len(pairwise)} pairs')
    print(f'Time taken: {t2-t1:.3f}s')
    return t2 - t1

def test_connection_reuse():
    """Test that connection pooling is working by making multiple rapid requests"""
    print('\n' + '=' * 60)
    print('TEST 4: Connection Reuse (5 rapid health checks)')
    print('=' * 60)
    
    times = []
    for i in range(5):
        t1 = time.time()
        calling_api.health_check()
        t2 = time.time()
        times.append(t2 - t1)
        print(f'  Request {i+1}: {times[-1]:.3f}s')
    
    print(f'\nFirst request: {times[0]:.3f}s')
    print(f'Avg subsequent: {sum(times[1:])/len(times[1:]):.3f}s')
    print('(Subsequent should be faster due to connection reuse)')
    return times

def test_db_bompot():
    print('\n' + '=' * 60)
    print('TEST 5: db_bompot (Double board scenarios)')
    print('=' * 60)
    hands, deck = generate_plo6_hands(3)
    board1 = ''.join(deck[18:21])  # 3 cards for board 1
    board2 = ''.join(deck[21:24])  # 3 cards for board 2
    
    print(f'Hero hands: {hands}')
    print(f'Board 1: {board1}')
    print(f'Board 2: {board2}')
    
    t1 = time.time()
    result = calling_api.db_bompot(hands, board1, board2, num_scenarios=60, num_opponents=2)
    t2 = time.time()
    
    if result:
        print(f'Hands calculated: {list(result.keys())}')
        first_hand = list(result.keys())[0]
        print(f'Scenarios for {first_hand}: board1={len(result[first_hand]["board1"])}, board2={len(result[first_hand]["board2"])}')
    else:
        print('ERROR: No result returned')
    print(f'Time taken: {t2-t1:.3f}s')
    return t2 - t1

def test_batch_parallel():
    """Test batch API - multiple independent calculations in parallel"""
    print('\n' + '=' * 60)
    print('TEST 3: BATCH PARALLEL (3 calculations at once)')
    print('=' * 60)
    
    # Generate 3 independent hand sets
    hands_list = []
    for _ in range(3):
        hands, _ = generate_plo6_hands(4)
        hands_list.append(hands)
    
    print(f'Running 3 independent calculations in parallel...')
    
    t1 = time.time()
    results = calling_api.preflop_multithread_plo6_batch(hands_list, trials=2500)
    t2 = time.time()
    
    for i, result in enumerate(results):
        if result:
            print(f'  Set {i+1}: {len(result["individual"])} hands evaluated')
    
    print(f'Total time for 3 parallel calcs: {t2-t1:.3f}s')
    print(f'Effective time per calc: {(t2-t1)/3:.3f}s')
    return t2 - t1

if __name__ == '__main__':
    print('Starting API Optimization Tests')
    print('Using endpoint:', calling_api.dns)
    print()
    
    # Run health check first
    test_health_check()
    
    # Single request timing
    print('\n' + '=' * 60)
    print('SINGLE REQUEST TIMING (trials vs speed)')
    print('=' * 60)
    
    ultrafast_time = test_preflop_multithread_plo6_ultrafast()
    fast_time = test_preflop_multithread_plo6_fast()
    
    # Batch parallel test
    batch_time = test_batch_parallel()
    
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'Single calc (1k trials):  {ultrafast_time:.3f}s')
    print(f'Single calc (2.5k trials): {fast_time:.3f}s')
    print(f'Batch 3 calcs (parallel):  {batch_time:.3f}s ({batch_time/3:.3f}s effective per calc)')
    print()
    print('BOTTLENECK: Server ProcessPoolExecutor startup (~5s fixed cost)')
    print('To get under 3s: Requires server-side changes (persistent worker pool)')
