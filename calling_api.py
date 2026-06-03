import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import itertools
import random
import json
import traceback

# data = ["AcAd5h6h7h", "KhKdQsJsTs", "7c7d7s8s8c", "2c3c4c5c6c"]
# board = "JcQcKc"

# dns = "http://127.0.0.1:5050"
dns = "http://pptloadbalancer-1427274483.ap-south-1.elb.amazonaws.com"
# dns = "https://socially-flexible-bison.ngrok-free.app"

default_range = "25%"

# ============= CONNECTION POOLING SETUP =============
# Create a session with connection pooling - this reuses TCP connections
# instead of creating new ones for each request (major speed improvement)
def create_session():
    """Create a session with connection pooling and retry logic."""
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,                    # Retry up to 3 times
        backoff_factor=0.5,         # Wait 0.5s, 1s, 2s between retries
        status_forcelist=[500, 502, 503, 504],  # Retry on these status codes
    )
    
    # Configure connection pool
    adapter = HTTPAdapter(
        pool_connections=20,        # Number of connection pools
        pool_maxsize=50,            # Max connections per pool
        max_retries=retry_strategy,
        pool_block=False            # Don't block when pool is full
    )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

# Global session for connection reuse
_session = None

def get_session():
    """Get or create the global session."""
    global _session
    if _session is None:
        _session = create_session()
    return _session


def request(payload, url, session=None):
    """Make a request with connection pooling and timeout optimization."""
    if session is None:
        session = get_session()
    try:
        # Use tuple timeout: (connect_timeout, read_timeout)
        # Fail fast on connection issues, but allow time for computation
        response = session.post(url, json=payload, timeout=(5, 120))
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f'Error occurred: {e}')
        return None

def health_check():
    url = dns + "/health"
    response = get_session().get(url, timeout=(5, 10))
    return response.text

def individual_strengths(locards, board=""):
    url = dns + "/strength"
    payloads = [{"cards": [locards[i]] + [card for j, card in enumerate(locards) if i != j]} for i in
                range(len(locards))]
    for payload in payloads:
        payload['board'] = board
    
    results = {}
    session = get_session()
    # Increase max_workers - let the connection pool handle limits
    with ThreadPoolExecutor(max_workers=min(len(payloads), 20)) as executor:
        futures = [executor.submit(request, payload, url, session) for payload in payloads]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                results.update(result)
    return results

def pairwise_results(locards, board=""):
    url = dns + "/pair"
    # Generate all pairs of distinct hands
    payloads = [{"cards0": locards[i], "cards1": locards[j],
                 "dcs": "".join([card for k, card in enumerate(locards) if k != i and k != j])}
                for i in range(len(locards)) for j in range(i + 1, len(locards))]
    for payload in payloads:
        payload['board'] = board

    results = {}
    session = get_session()
    # Increase max_workers based on payload count
    with ThreadPoolExecutor(max_workers=min(len(payloads), 20)) as executor:
        futures = [executor.submit(request, payload, url, session) for payload in payloads]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                results.update(result)
    return results


def reformat_cards(cards):
    if type(cards[0]) == list:
        cards = ["".join(card) for card in cards]
    ret = []
    for hand in cards:
        # make alternate case, starting from upper, then lower, then upper, then lower
        _ = ''
        for i in range(0, len(hand)):
            if i % 2 == 0:
                _ += hand[i].upper()
            else:
                _ += hand[i].lower()
        ret.append(_)

    return ret

def preflop_multithread_plo6(locards, dead_cards=[], total_number_of_trials=None, number_of_processes=None, board=[]):
    """
    Calculate PLO6 equities for multiple hands.
    
    Args:
        locards: List of hand strings
        dead_cards: List of dead cards
        total_number_of_trials: Number of Monte Carlo trials (default: 10000 on server)
                               Use 2500 for faster results (~7s), 1000 for fastest (~6s)
        number_of_processes: Number of parallel processes (default: 4 on server)
        board: Community cards
    
    Performance notes:
        - 10k trials: ~9-10s, ~0.5% variance (most accurate)
        - 2.5k trials: ~7s, ~1% variance
        - 1k trials: ~6s, ~2% variance (fastest possible with current server)
        - Server has ~5s fixed overhead from process spawning
    """
    url = dns + '/plo6pythonindividualandpairwise'
    locards = reformat_cards(locards)

    payload = {
        "cards": locards,
        "dead_cards": dead_cards,
    }
    if board:
        payload['board'] = board
    if total_number_of_trials:
        payload['trials'] = total_number_of_trials
    if number_of_processes:
        payload['processes'] = number_of_processes
    
    result = request(payload, url, get_session())
    if result:
        # in the pairwise results replace the separator "_" in the keys by " : "
        pairwise_results = result['pairwise']
        result['pairwise'] = {key.replace("_", " : "): value for key, value in pairwise_results.items()}

    return result


def preflop_multithread_plo6_batch(hands_list, dead_cards_list=None, trials=2500, board_list=None):
    """
    Calculate PLO6 equities for MULTIPLE sets of hands in parallel.
    Use this when you have several independent calculations to run.
    
    Args:
        hands_list: List of hand sets, e.g. [[hand1, hand2, hand3], [hand4, hand5, hand6]]
        dead_cards_list: Optional list of dead cards for each set
        trials: Number of trials per calculation
        board_list: Optional list of boards for each set
    
    Returns:
        List of results in same order as input
    
    Example:
        results = preflop_multithread_plo6_batch([
            ['AcAdKhKd5h6h', 'QsQhJcJdTs9s'],
            ['AsAhKcKd7h8h', 'KsKhQcQdJs9s']
        ])
    """
    url = dns + '/plo6pythonindividualandpairwise'
    session = get_session()
    
    # Prepare all payloads
    payloads = []
    for i, hands in enumerate(hands_list):
        hands = reformat_cards(hands)
        payload = {
            "cards": hands,
            "trials": trials,
            "dead_cards": dead_cards_list[i] if dead_cards_list else [],
        }
        if board_list and board_list[i]:
            payload['board'] = board_list[i]
        payloads.append(payload)
    
    # Execute all requests in parallel
    results = [None] * len(payloads)
    with ThreadPoolExecutor(max_workers=len(payloads)) as executor:
        future_to_idx = {executor.submit(request, payload, url, session): i 
                        for i, payload in enumerate(payloads)}
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()
            if result:
                pairwise = result['pairwise']
                result['pairwise'] = {k.replace("_", " : "): v for k, v in pairwise.items()}
            results[idx] = result
    
    return results

def preflop_results(locards):
    url_strength = dns + "/strength"
    url_pair = dns + "/pair"
    # Define payloads
    strength_payloads = [{"cards": [locards[i]] + [card for j, card in enumerate(locards) if i != j]} for i in
                         range(len(locards))]
    pair_payloads = [{"cards0": locards[i], "cards1": locards[j],
                      "dcs": "".join([card for k, card in enumerate(locards) if k != i and k != j])}
                     for i in range(len(locards)) for j in range(i + 1, len(locards))]

    # Total number of requests to be sent
    total_requests = len(strength_payloads) + len(pair_payloads)
    print(f"Total requests: {total_requests}")

    # Define results dictionary
    individual_results = {}
    pairwise_results_dict = {}
    
    session = get_session()

    # KEY FIX: Submit ALL requests at once, not sequentially!
    # Previously, pair requests waited for strength requests to complete
    with ThreadPoolExecutor(max_workers=min(total_requests, 30)) as executor:
        # Submit ALL requests at once
        futures_strength = {executor.submit(request, payload, url_strength, session): 'strength' 
                          for payload in strength_payloads}
        futures_pair = {executor.submit(request, payload, url_pair, session): 'pair' 
                       for payload in pair_payloads}
        
        # Combine all futures
        all_futures = {**futures_strength, **futures_pair}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(all_futures):
            result = future.result()
            if result:
                if all_futures[future] == 'strength':
                    individual_results.update(result)
                else:
                    pairwise_results_dict.update(result)

    return individual_results, pairwise_results_dict


def preflop_multithread_plo5(locards, board=[], total_number_of_trials=None, number_of_processes=None, fast_mode=False):
    """
    Calculate PLO5 equities for multiple hands.
    
    Args:
        fast_mode: If True, uses fewer trials (2500) but more processes (8) for ~3x speedup
    """
    url = dns + '/plo5pythonindividualandpairwise'
    payload = {
        "cards": locards,
        "board": board,
    }
    
    if fast_mode:
        payload['trials'] = total_number_of_trials or 2500
        payload['processes'] = number_of_processes or 8
    else:
        if total_number_of_trials:
            payload['trials'] = total_number_of_trials
        if number_of_processes:
            payload['processes'] = number_of_processes
    
    result = request(payload, url, get_session())
    if result:
        # in the pairwise results replace the separator "_" in the keys by " : "
        pairwise_results = result['pairwise']
        result['pairwise'] = {key.replace("_", " : "): value for key, value in pairwise_results.items()}

    return result



def draw_table():
    url = f"{dns}/draw_table"
    session = get_session()
    # Session already has retry logic built in, no need for manual retries
    response = session.get(url, timeout=(5, 30))
    return response.json()



def preflop_multithread_plo6_3h(locards, board=[], opponent_hands=[], total_number_of_trials=None, number_of_processes=None):
    # sample data request format:
    # data = {
    #     "cards": ["AcAd5h6h7h", "KhKdQsJsTs", "7c7d7s8s8c", "2c3c4c5c6c"],
    #     "dead_cards": ["7s", "8s", "8c", "9c", "4s", "4c", "5d", "5c"],
    #     "trials": 10000,
    #     "processes": 4
    # }
    url = dns + '/plo6pythonindividualandpairwise_3h'
    payload = {
        "cards": locards,
    }
    if total_number_of_trials:
        payload['trials'] = total_number_of_trials
    if number_of_processes:
        payload['processes'] = number_of_processes
    if board:
        payload['board'] = board
    if opponent_hands:
        payload['opponent_hands'] = opponent_hands
    
    result = request(payload, url, get_session())
    if result:
        # in the pairwise results replace the separator "_" in the keys by " : "
        pairwise_results = result['pairwise']
        result['pairwise'] = {key.replace("_", " : "): value for key, value in pairwise_results.items()}

    return result

def db_bompot(locards, board1, board2, deadcards=[], num_scenarios=150, num_opponents=2):
    url = dns + '/dbbompot'
    scenarios_per_request = 15
    payload = {
        "herocards": locards,
        "board1": board1,
        "board2": board2,
        "numscenarios": scenarios_per_request,
        "numopponents": num_opponents
    }
    if len(deadcards):
        payload['deadcards'] = deadcards

    bpresults = []
    session = get_session()
    num_requests = num_scenarios // scenarios_per_request
    
    with ThreadPoolExecutor(max_workers=min(num_requests, 20)) as executor:
        results = [executor.submit(request, payload.copy(), url, session) for _ in range(num_requests)]
        for future in concurrent.futures.as_completed(results):
            result = future.result()
            if result:
                bpresults.append(result)

    combined_results = {}
    for result in bpresults:
        for hand, boards in result.items():
            if hand not in combined_results:
                combined_results[hand] = {'board1': [], 'board2': []}
            combined_results[hand]['board1'].extend(boards['board1'])
            combined_results[hand]['board2'].extend(boards['board2'])
    return combined_results

def add_hand_history(data):
    url = dns + '/add_hand_history'
    return request(data, url, get_session())

def plo_equity(hands, board=[], dead_cards=[]):
    """
    Calculate PLO equity between known hands
    
    Args:
        hands (list): List of hands, can be list of lists or list of strings
        board (list or str, optional): Community cards. Defaults to empty list.
        dead_cards (list or str, optional): Dead cards. Defaults to empty list.
    
    Returns:
        dict: Equity results from the API
    """
    url = dns + "/ploequity"
    
    # Prepare payload
    payload = {
        "hands": hands,
    }
    
    # Add optional parameters if provided
    if board:
        payload["board"] = board
    if dead_cards:
        payload["dead_cards"] = dead_cards
    
    try:
        session = get_session()
        response = session.post(url, json=payload, timeout=(5, 120))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f'Error occurred: {e}')
        return None





if __name__ == '__main__':
    # from prettify import prettify_dict_of_dfs

        # while True:
    print(health_check())
    # deck = [card for card in itertools.product('23456789TJQKA', 'shdc')]
    # deck = ["".join(card) for card in deck]
    # random.shuffle(deck)
    # n = 4
    # num_cards_per_hand = 5
    # hands = [deck[i * num_cards_per_hand:(i + 1) * num_cards_per_hand] for i in range(n)]
    # # get last 3 cards from the deck as board
    # board = deck[-3:]
    # print([''.join(hand) for hand in hands])
    # print(board)
    # dead_cards = []
    # total_number_of_trials = 15000
    # t1 = time.time()
    #
    # equities = preflop_results([''.join(hand) for hand in hands])
    #
    #
    # t2 = time.time()
    # # print(individual_results_m)
    # # print(pairwise_results_m)
    # print(f"Time taken: {t2 - t1}")

    # compare the results
    # for key in individual_results.keys():
    #     print(f"Individual Results: {individual_results[key]} vs {individual_results_m[key]}")
    # for key in pairwise_results.keys():
    #     print(f"Pairwise Results: {100 - pairwise_results[key]['25%']} vs {pairwise_results_m[key]}")

    # Example usage of plo_equity function
    
    # Example 1: Hands as lists of lists
    hands1 = [
        ['As', 'Ah', 'Kc', 'Qd'],  # Hand 1
        ['Qs', 'Qh', 'Jc', 'Jd']   # Hand 2
    ]
    board1 = ""
    dead_cards1 = ""
    
    print("Example 1 - Hands as lists of lists:")
    result1 = plo_equity(hands1, board=board1, dead_cards=dead_cards1)
    print(result1)
    
    # Example 2: Hands as strings
    hands2 = ['AsAhKcKd', 'QsQhJcJd']
    board2 = 'TcJd'
    dead_cards2 = '2s3h'
    
    print("\nExample 2 - Hands as strings:")
    result2 = plo_equity(hands2, board=board2, dead_cards=dead_cards2)
    print(result2)
    
    # Example 3: Minimal input (just hands)
    hands3 = [
        ['As', 'Ah', 'Kc', 'Kd'],
        ['Qs', 'Qh', 'Jc', 'Jd'],
        ['Tc', 'Th', '9c', '9d']
    ]
    
    print("\nExample 3 - Minimal input:")
    result3 = plo_equity(hands3)
    print(result3)



