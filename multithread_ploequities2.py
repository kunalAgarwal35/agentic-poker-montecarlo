# In this file we will have multiple functions to calculate frequencies of hand strengths by the river
from concurrent.futures import ProcessPoolExecutor
import random
import generating_list as scores
from itertools import combinations, product
import time
import pandas as pd
import traceback
import atexit
import os

try:
    from line_profiler_pycharm import profile
except ImportError:
    # Dummy decorator if line_profiler not available
    def profile(func):
        return func

# ============= GLOBAL PROCESS POOL (PERSISTENT WORKERS) =============
# This eliminates the ~5 second overhead of creating processes per request
_GLOBAL_EXECUTOR = None
_DEFAULT_WORKERS = 4  # Match server core count

def get_global_executor(num_workers=None):
    """Get or create a persistent ProcessPoolExecutor."""
    global _GLOBAL_EXECUTOR
    if _GLOBAL_EXECUTOR is None:
        workers = num_workers or _DEFAULT_WORKERS
        print(f"[ProcessPool] Initializing global executor with {workers} workers...")
        _GLOBAL_EXECUTOR = ProcessPoolExecutor(max_workers=workers)
        atexit.register(shutdown_executor)
        print(f"[ProcessPool] Global executor ready (PID: {os.getpid()})")
    return _GLOBAL_EXECUTOR

def shutdown_executor():
    """Cleanup function called at exit."""
    global _GLOBAL_EXECUTOR
    if _GLOBAL_EXECUTOR is not None:
        print("[ProcessPool] Shutting down global executor...")
        _GLOBAL_EXECUTOR.shutdown(wait=True)
        _GLOBAL_EXECUTOR = None

def _dummy_warmup_task():
    """Simple task used to warm up process pool workers."""
    return 1

def warmup_executor():
    """Pre-warm the executor by running a dummy task on each worker."""
    executor = get_global_executor()
    futures = [executor.submit(_dummy_warmup_task) for _ in range(_DEFAULT_WORKERS)]
    for f in futures:
        f.result()
    print("[ProcessPool] Workers warmed up and ready")
# ====================================================================

board_categories_to_hand_strengths = {
    "Trips Board": ["Quads", "AA,KK", "TT-QQ", "22-99", "Others"],
    "Paired Board": ["Quads", "Full House", "Trips", "Others"],
    "Unpaired Board - Monotone": ["Nut Flush", "Flush", "Set", "Others"],
    "Unpaired Board - Two Tone (Straight)": ["Nut Straight", "Straight", "Set", "Flush Draw", "Wrap", "Others"],
    "Unpaired Board - Two Tone (Non-Straight)": ["Set", "Top Two", "Nut Flush Draw", "Flush Draw", "Wrap", "Others"],
    "Unpaired Board - Rainbow (Straight)": ["Nut Straight", "Straight", "Set", "Wrap", "Others"],
    "Unpaired Board - Rainbow (Non-Straight)": ["Set", "Top Two", "Wrap", "Others"]
}


def is_quads(hand, board):
    # Ensure that the evaluation correctly handles PLO rules and the specific scenario described
    combined = hand + board
    combined_ranks = [card[0] for card in combined]
    board_ranks = [card[0] for card in board]
    hand_ranks = [card[0] for card in hand]

    if len(set(combined_ranks)) > len(combined) - 3:
        return False

    # Look for ranks that appear four times in the combined hand and board
    for rank in set(combined_ranks):
        if combined_ranks.count(rank) == 4:
            # Check for the scenario where the board has three and the hand has the fourth
            if board_ranks.count(rank) == 3 and hand_ranks.count(rank) == 1:
                return True
            if board_ranks.count(rank) == 2 and hand_ranks.count(rank) == 2:
                return True
            # Check for the scenario where the board has one and the hand has three,
            # which should NOT qualify as quads in PLO since you can only use two cards from your hand
            if board_ranks.count(rank) == 1 and hand_ranks.count(rank) == 3:
                continue  # This does not qualify as quads in PLO, move to next rank
    return False


def is_full_house(hand, board):
    # Check if a full house can be formed with two cards from the hand and three from the board.
    if best_category_of_hand_on_board(hand, board) == 'Full House':
        return True
    return False


def is_nut_flush_or_flush(hand, board):
    '''
    Check if the hand is a nut flush or a flush
    :param hand:
    :param board:
    :return:
        False, False if the hand is not a flush
        True, True if the hand is a nut flush
        False, True if the hand is a flush but not a nut flush
    '''
    bc = best_category_of_hand_on_board(hand, board)
    if not bc in ['Flush', 'Straight Flush', 'Royal Flush']:
        return False, False
    else:
        suits = [card[1] for card in board]
        ranks = [card[0] for card in board]
        flush_suit = max(set(suits), key=suits.count)
        flush_ranks = [rank for rank, suit in zip(ranks, suits) if suit == flush_suit]
        ranks_order = '23456789TJQKA'
        flush_cards = [rank + flush_suit for rank in ranks_order]

        # remove the cards that are in the board
        for card in board:
            if card in flush_cards:
                flush_cards.remove(card)
        # get the last item in the flush_cards list
        nut_flush_card = flush_cards[-1]
        if nut_flush_card in hand:
            return True, True
        else:
            return False, True


def is_nutflushdraw_or_flushdraw(hand, board):
    # This function is only suitable for flop evaluations
    suits = [card[1] for card in board]
    ranks = [card[0] for card in board]
    ranks_order = 'A23456789TJQKA'

    # Identify suits that occur precisely twice on the board
    suit_counts = {suit: suits.count(suit) for suit in set(suits)}
    flush_draw_suits = [suit for suit, count in suit_counts.items() if count == 2]

    # Check if the hand contributes to at least a flush draw for those suits
    hand_suits = [card[1] for card in hand]
    is_flush_draw = False
    is_nut_flush_draw = False
    for flush_suit in flush_draw_suits:
        if hand_suits.count(flush_suit) >= 2:
            # Identify the highest card of the flush suit outside the board
            flush_ranks = [rank for rank, suit in zip(ranks, suits) if suit == flush_suit]
            flush_cards = [rank + flush_suit for rank in ranks_order]
            for card in board:
                if card in flush_cards:
                    flush_cards.remove(card)
            nut_flush_card = flush_cards[-1]
            if nut_flush_card in hand:
                is_nut_flush_draw = True
                is_flush_draw = True
            else:
                is_flush_draw = True
    return is_nut_flush_draw, is_flush_draw


def is_AA_KK(hand, board):
    ranks = [card[0] for card in hand]
    rank_counts = get_rank_counts(hand)
    if 'A' in rank_counts.keys() and rank_counts['A'] >= 2:
        return True
    if 'K' in rank_counts.keys() and rank_counts['K'] >= 2:
        return True
    return False


def is_TT_to_QQ(hand, board):
    ranks = [card[0] for card in hand]
    rank_counts = get_rank_counts(hand)
    if 'T' in rank_counts.keys() and rank_counts['T'] >= 2:
        return True
    if 'J' in rank_counts.keys() and rank_counts['J'] >= 2:
        return True
    if 'Q' in rank_counts.keys() and rank_counts['Q'] >= 2:
        return True
    return False


def is_22_to_99(hand, board):
    ranks = [card[0] for card in hand]
    rank_counts = get_rank_counts(hand)
    if '2' in rank_counts.keys() and rank_counts['2'] >= 2:
        return True
    if '3' in rank_counts.keys() and rank_counts['3'] >= 2:
        return True
    if '4' in rank_counts.keys() and rank_counts['4'] >= 2:
        return True
    if '5' in rank_counts.keys() and rank_counts['5'] >= 2:
        return True
    if '6' in rank_counts.keys() and rank_counts['6'] >= 2:
        return True
    if '7' in rank_counts.keys() and rank_counts['7'] >= 2:
        return True
    if '8' in rank_counts.keys() and rank_counts['8'] >= 2:
        return True
    if '9' in rank_counts.keys() and rank_counts['9'] >= 2:
        return True
    return False


def is_nut_straight_or_straight(hand, board):
    if best_category_of_hand_on_board(hand, board) != 'Straight':
        return False, False
    rank_order = 'A23456789TJQKA'  # Duplicate 'A' to account for the wheel straight (A-2-3-4-5)
    board_ranks = [card[0] for card in board]
    hand_ranks = [card[0] for card in hand]

    # Loop through the highest possible straights
    for i in range(10):  # From 'A' high to '5' high
        potential_straight = rank_order[9 - i:14 - i]  # Grab sequences of 5 ranks in descending order
        # Count how many cards from the potential straight are present on the board
        board_ranks_in_straight = [rank for rank in potential_straight if rank in board_ranks]
        board_match_count = len(board_ranks_in_straight)
        required_ranks_in_hand = [rank for rank in potential_straight if rank not in board_ranks_in_straight]
        if board_match_count == 3:
            if required_ranks_in_hand[0] in hand_ranks and required_ranks_in_hand[1] in hand_ranks:
                return True, True
            else:
                return False, True


def is_flush(hand, board):
    if best_category_of_hand_on_board(hand, board) == 'Flush':
        return True
    return False


def is_set(hand, board):
    combined = hand + board
    rank_counts = get_rank_counts(combined)
    if 3 in rank_counts.values() and not is_full_house(hand, board):
        return True
    return False


def is_wrap(hand, board):
    # Implement checking for a wrap draw
    hand_ = list()
    board_ = list()
    for card in hand:
        if card[0] + 's' not in hand_:
            hand_.append(card[0] + 's')
        elif card[0] + 'h' not in hand_:
            hand_.append(card[0] + 'h')
    for card in board:
        if card[0] + 'c' not in board_:
            board_.append(card[0] + 'c')
        elif card[0] + 'd' not in board_:
            board_.append(card[0] + 'd')
    ranks = '23456789TJQKA'
    ranks_not_in_board = [rank for rank in ranks if rank not in [card[0] for card in board]]
    num_straights = sum([1 for rank in ranks_not_in_board if
                         best_category_of_hand_on_board(hand_, board_ + [rank + 'c']) == 'Straight'])
    if num_straights >= 3:
        return True
    return False


def is_top_two(hand, board):
    # Extract ranks from the board and hand
    board_ranks = [card[0] for card in board]
    hand_ranks = [card[0] for card in hand]

    # Determine the unique ranks on the board and sort them by their poker value
    unique_board_ranks = sorted(set(board_ranks), key=lambda rank: '23456789TJQKA'.index(rank), reverse=True)

    # Extract the top two ranks from the sorted unique ranks on the board
    if len(unique_board_ranks) < 2:
        return False  # Not enough unique ranks for top two
    top_two_ranks = unique_board_ranks[:2]

    # Check if each of the top two ranks from the board occurs exactly once in the hand
    return all(hand_ranks.count(rank) == 1 for rank in top_two_ranks)


def get_rank_counts(cards):
    """Returns a count of each rank in a list of cards."""
    ranks = [card[0] for card in cards]
    rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}
    return rank_counts


def is_straight_board(rank_indices):
    """
    Determines if the board forms a straight
    Args:
        rank_indices (list): List of rank indices for the board, with Ace as 13.

    Returns:
        bool: True if the board forms a straight, False otherwise.
    """
    # Straight check without Ace duality
    if max(rank_indices) - min(rank_indices) <= 4 and len(set(rank_indices)) == 3:
        return True
    if 0 in rank_indices:
        # replace with 13
        rank_indices = [13 if x == 0 else x for x in rank_indices]
        if max(rank_indices) - min(rank_indices) <= 4 and len(set(rank_indices)) == 3:
            return True
    return False


def categorize_board(board):
    """
    Categorizes the board into various types for further analysis
    Args:
        board (str or list): The board as a string or list of cards.
    Returns:
        str: The category of the board.
    Scope of categorization:
    [Trips Board, Paired Board, Unpaired Board - Monotone, Unpaired Board - Two Tone (Straight),
    Unpaired Board - Two Tone (Non-Straight), Unpaired Board - Rainbow (Straight), Unpaired Board - Rainbow (Non-Straight)]
    """
    if type(board) == str:
        board = [board[i:i + 2] for i in range(0, len(board), 2)]
    ranks = "A23456789TJQK"
    suits = [card[1] for card in board]
    rank_indices = sorted([ranks.index(card[0]) for card in board])

    unique_ranks = set(rank_indices)
    unique_suits = set(suits)

    # Use the dedicated function for straight detection
    is_straight = is_straight_board(rank_indices)

    # Classify the board based on suits and straight logic
    if len(unique_ranks) == 1:
        return "Trips Board"
    elif len(unique_ranks) == 2:
        return "Paired Board"
    elif len(unique_suits) == 1:
        return "Unpaired Board - Monotone"
    elif len(unique_suits) == 2:
        if is_straight:
            return "Unpaired Board - Two Tone (Straight)"
        else:
            return "Unpaired Board - Two Tone (Non-Straight)"
    elif len(unique_suits) == 3:
        if is_straight:
            return "Unpaired Board - Rainbow (Straight)"
        else:
            return "Unpaired Board - Rainbow (Non-Straight)"


def hand_strength_category(hand, board, board_category):
    # Example: Sequentially check each hand strength for the given board category
    if board_category == "Trips Board":
        if is_quads(hand, board): return "Quads"
        if is_AA_KK(hand, board): return "AA,KK"
        if is_TT_to_QQ(hand, board): return "TT-QQ"
        if is_22_to_99(hand, board): return "22-99"
        return "Others"
    if board_category == "Paired Board":
        if is_quads(hand, board): return "Quads"
        if is_full_house(hand, board): return "Full House"
        if is_set(hand, board): return "Trips"
        return "Others"
    if board_category == "Unpaired Board - Monotone":
        nut_flush, flush = is_nut_flush_or_flush(hand, board)
        if nut_flush: return "Nut Flush"
        if flush: return "Flush"
        if is_set(hand, board): return "Set"
        return "Others"
    if board_category == "Unpaired Board - Two Tone (Straight)":
        nut_straight, straight = is_nut_straight_or_straight(hand, board)
        if nut_straight: return "Nut Straight"
        if straight: return "Straight"
        if is_set(hand, board): return "Set"
        nut_flush_draw, flush_draw = is_nutflushdraw_or_flushdraw(hand, board)
        if nut_flush_draw: return "Nut Flush Draw"
        if flush_draw: return "Flush Draw"
        if is_wrap(hand, board): return "Wrap"
        return "Others"
    if board_category == "Unpaired Board - Two Tone (Non-Straight)":
        if is_set(hand, board): return "Set"
        if is_top_two(hand, board): return "Top Two"
        nut_flush_draw, flush_draw = is_nutflushdraw_or_flushdraw(hand, board)
        if nut_flush_draw: return "Nut Flush Draw"
        if flush_draw: return "Flush Draw"
        if is_wrap(hand, board): return "Wrap"
        return "Others"
    if board_category == "Unpaired Board - Rainbow (Straight)":
        nut_straight, straight = is_nut_straight_or_straight(hand, board)
        if nut_straight: return "Nut Straight"
        if straight: return "Straight"
        if is_set(hand, board): return "Set"
        if is_wrap(hand, board): return "Wrap"
        return "Others"
    if board_category == "Unpaired Board - Rainbow (Non-Straight)":
        if is_set(hand, board): return "Set"
        if is_top_two(hand, board): return "Top Two"
        if is_wrap(hand, board): return "Wrap"
        return "Others"
    return "Others"


def categorize_hands_chunk(hands_chunk, board):
    """
    Helper function to categorize a chunk of hands.
    """
    board_category = categorize_board(board)
    hand_categories = [(hand, hand_strength_category(hand, board, board_category)) for hand in hands_chunk]
    return hand_categories


def break_hands_into_categories(hands, board, num_processes=4):
    # Number of processes to use
    # Splitting the hands list into chunks for each process
    chunk_size = len(hands) // num_processes
    hands_chunks = [hands[i * chunk_size:(i + 1) * chunk_size] for i in range(num_processes)]
    if len(hands) % num_processes:
        hands_chunks[-1].extend(hands[num_processes * chunk_size:])

    return_dict = {}
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submitting tasks to the executor
        futures = [executor.submit(categorize_hands_chunk, chunk, board) for chunk in hands_chunks]

        # Gathering and aggregating results
        for future in futures:
            hand_categories_chunk = future.result()
            for hand, category in hand_categories_chunk:
                if category not in return_dict:
                    return_dict[category] = [hand]
                else:
                    return_dict[category].append(hand)

    return return_dict


# def best_score_of_hand_on_board(hand, board, score_dict):
#     best_score = 0
#     # Generate all possible 2-card combinations from the hand and 3-card combinations from the board
#     for hand_combo in combinations(hand, 2):
#         for board_combo in combinations(board, 3):
#             # Combine hand and board cards to form a 5-card poker hand
#             full_hand = hand_combo + board_combo
#             # Convert to a string representation or any format that matches the score_dict keys
#             hand_str = scores.sort_hand(full_hand)
#             # Update best score if this combo's score is higher
#             best_score = max(best_score, score_dict.get(hand_str, 0))
#     return best_score

@profile
def best_score_of_hand_on_board(hand, board, score_dict):
    board_combos = list(combinations(board, 3))
    return max(score_dict.get(''.join(sorted(hand_combo + board_combo)), 0)
               for hand_combo in combinations(hand, 2)
               for board_combo in board_combos)


def determine_game_type(hand):
    """Determines the game type based on the number of cards in a hand.

    Args:
        hand (list): A single hand.

    Returns:
        tuple: The game type and number of cards per hand.
    """
    hand_length = len(hand)
    if hand_length == 5:
        return 'plo5', 5
    elif hand_length == 6:
        return 'plo6', 6
    elif hand_length == 4:
        return 'plo4', 4
    else:
        raise ValueError('Invalid number of cards in the hand.')


def preprocess_hand(hand_str):
    """Converts a hand string into a list of cards.

    Args:
        hand_str (str): The hand in string format.

    Returns:
        list: The hand as a list of cards.
    """
    return [hand_str[i:i + 2] for i in range(0, len(hand_str), 2)]


def preprocess_hands(hands_list):
    """Preprocesses multiple hands from strings to lists of cards.

    Args:
        hands_list (list): The list of hands in string format.

    Returns:
        list: A list of hands, each as a list of cards.
    """
    return [preprocess_hand(hand) for hand in hands_list]


def best_possible_score(board):
    deck = scores.generate_deck(board)
    all_possible_hands = list(combinations(deck, 2))
    best_score = 0
    for hand in all_possible_hands:
        hand_score = best_score_of_hand_on_board(hand, board, scores.score_dict)
        best_score = max(best_score, hand_score)
    return best_score


def best_category_of_hand_on_board(hand, board):
    best_score = 0
    best_combo = ''
    # Generate all possible 2-card combinations from the hand and 3-card combinations from the board
    for hand_combo in combinations(hand, 2):
        for board_combo in combinations(board, 3):
            # Combine hand and board cards to form a 5-card poker hand
            full_hand = hand_combo + board_combo
            # Convert to a string representation or any format that matches the score_dict keys
            hand_str = scores.sort_hand(''.join(full_hand))
            # Update best score if this combo's score is higher
            new_score = scores.score_dict.get(hand_str, 0)
            best_score = max(best_score, new_score)
    return scores.scores_to_category_map.get(best_score, 0)


def test_best_score_function():
    # let's assume some hands and a board
    hand = ['As', 'Ks', 'Qs', 'Js', 'Ts', '9s']
    board = ['2s', '3s', 'Ac', 'Ad', '6c']
    strength = best_category_of_hand_on_board(hand, board)
    # assert strength is 'Three of a Kind'
    assert strength == 'Three of a Kind'


def generate_deck(exclude_cards):
    suits = 'shdc'
    ranks = '23456789TJQKA'
    deck = [r + s for r in ranks for s in suits if (r + s) not in exclude_cards]
    random.shuffle(deck)
    return deck


def plo5equities(hands, number_of_trials, board=[], opponent_hands=[]):
    '''
    :param hands: list of lists of strings, each list of strings is a hand
    :param number_of_trials:
    :param board:
    :return:
    '''
    # print("Hands: ", hands, " Type: ", type(hands))
    # print("Number of Trials: ", number_of_trials, " Type: ", type(number_of_trials))
    # print("Board: ", board, " Type: ", type(board))
    # print("Opponent Hands: ", len(opponent_hands), " Type: ", type(opponent_hands))
    try:
        all_cards_in_play = [card for hand in hands for card in hand]
        if len(board):
            all_cards_in_play += board
        deck = generate_deck(all_cards_in_play)
        win_frequencies = {''.join(hand): 0 for hand in hands}
        win_frequencies_pairs = {''.join(a) + '_' + ''.join(b): 0 for a, b in combinations(hands, 2)}
        for _ in range(number_of_trials):
            # Pick a random hand from the deck
            if len(opponent_hands):
                random_hand = random.sample(opponent_hands, 1)[0]
                deck = generate_deck(all_cards_in_play + random_hand)
                random_board = board + random.sample(deck, 5 - len(board))
            else:
                random.shuffle(deck)
                random_hand = deck[:5]
                random_board = board + deck[6:(11 - len(board))]
            # Pick a random board from the deck

            random_hand_best_score = best_score_of_hand_on_board(random_hand, random_board, scores.score_dict)
            # random_hand_category = best_category_of_hand_on_board(random_hand, random_board)
            # print(random_hand, random_board, random_hand_category)
            # Calculate best score for each hand on the random board
            hand_scores = {''.join(hand): best_score_of_hand_on_board(hand, random_board, scores.score_dict) for hand in
                           hands}
            # hand_categories = {''.join(hand): best_category_of_hand_on_board(hand, random_board) for hand in hands}
            # print('Your Hand Categories: ', hand_categories)
            # check which hands have score higher than the random hand, if the score is higher, increment the win frequency, if it is equal, increment the win frequency by 0.5
            for hand, score in hand_scores.items():
                if score > random_hand_best_score:
                    win_frequencies[hand] += 1
                elif score == random_hand_best_score:
                    win_frequencies[hand] += 0.5
            # check which pair of hands have score higher than the random hand (either one could be higher), if the score is higher, increment the win frequency, if it is equal, increment the win frequency by 0.5, if all three are equal, increment the win frequency by 0.66
            pairs = combinations(hands, 2)
            for pair in pairs:
                key = ''.join(pair[0]) + '_' + ''.join(pair[1])
                if hand_scores[''.join(pair[0])] > random_hand_best_score or hand_scores[
                    ''.join(pair[1])] > random_hand_best_score:
                    win_frequencies_pairs[key] += 1
                elif hand_scores[''.join(pair[0])] == random_hand_best_score and hand_scores[
                    ''.join(pair[1])] == random_hand_best_score:
                    win_frequencies_pairs[key] += 0.66
                elif hand_scores[''.join(pair[0])] == random_hand_best_score or hand_scores[
                    ''.join(pair[1])] == random_hand_best_score:
                    win_frequencies_pairs[key] += 0.5
    except:
        breakpoint()
    # divide the win frequency by the number of trials to get the win frequency
    for hand in win_frequencies.keys():
        win_frequencies[hand] /= number_of_trials
    for pair in win_frequencies_pairs.keys():
        win_frequencies_pairs[pair] /= number_of_trials

    # print(f"Process Win Frequencies: {win_frequencies}")
    # print(f"Process Win Frequencies Pairs: {win_frequencies_pairs}")
    return win_frequencies, win_frequencies_pairs


def plo5equities_3h(hands, number_of_trials, board=[], opponent_hands=[]):
    '''
    :param hands: list of lists of strings, each list of strings is a hand
    :param number_of_trials:
    :param board: list
    :return:
    '''
    # print("Hands: ", hands, " Type: ", type(hands))
    # print("Number of Trials: ", number_of_trials, " Type: ", type(number_of_trials))
    # print("Board: ", board, " Type: ", type(board))
    # print("Opponent Hands: ", len(opponent_hands), " Type: ", type(opponent_hands))
    if type(hands[0]) == str:
        hands = [preprocess_hand(hand) for hand in hands]
    try:
        all_cards_in_play = [card for hand in hands for card in hand]
        if len(board):
            all_cards_in_play += board
        deck = generate_deck(all_cards_in_play)
        win_frequencies = {''.join(hand): 0 for hand in hands}
        win_frequencies_pairs = {''.join(a) + '_' + ''.join(b): 0 for a, b in combinations(hands, 2)}
        for _ in range(number_of_trials):
            # Pick a random hand from the deck
            if len(opponent_hands):
                random_hands = random.sample(opponent_hands, 2)
                random_hand = random_hands[0]
                random_hand2 = random_hands[1]
                if type(random_hand) == str:
                    random_hand = preprocess_hand(random_hand)
                    random_hand2 = preprocess_hand(random_hands[1])
                deck = generate_deck(all_cards_in_play + random_hand + random_hand2)
                random_board = board + random.sample(deck, 5 - len(board))
            else:
                random.shuffle(deck)
                random_hand = deck[:5]
                random_hand2 = deck[5:10]
                # Pick a random board from the deck
                random_board = board + deck[10:(15 - len(board))]
            random_hand_best_scores = [best_score_of_hand_on_board(h, random_board, scores.score_dict) for h in
                                       [random_hand, random_hand2]]
            hand_scores = {''.join(hand): best_score_of_hand_on_board(hand, random_board, scores.score_dict) for hand in
                           hands}
            # hand_categories = {''.join(hand): best_category_of_hand_on_board(hand, random_board) for hand in hands}
            # print('Your Hand Categories: ', hand_categories)
            # check which hands have score higher than the random hand, if the score is higher, increment the win frequency, if it is equal, increment the win frequency by 0.5
            for hand, score in hand_scores.items():
                if score < max(random_hand_best_scores):
                    pass
                elif score > max(random_hand_best_scores):
                    win_frequencies[hand] += 1
                elif max(random_hand_best_scores) == score and score == min(random_hand_best_scores):
                    win_frequencies[hand] += 0.33

            # check which pair of hands have score higher than the random hand (either one could be higher), if the score is higher, increment the win frequency, if it is equal, increment the win frequency by 0.5, if all three are equal, increment the win frequency by 0.66
            pairs = combinations(hands, 2)
            for pair in pairs:
                key = ''.join(pair[0]) + '_' + ''.join(pair[1])
                if hand_scores[''.join(pair[0])] > random_hand_best_scores[0] or hand_scores[
                    ''.join(pair[1])] > random_hand_best_scores[0]:
                    win_frequencies_pairs[key] += 1/2
                elif hand_scores[''.join(pair[0])] == random_hand_best_scores[0] and hand_scores[
                    ''.join(pair[1])] == random_hand_best_scores[0]:
                    win_frequencies_pairs[key] += 0.66/2
                elif hand_scores[''.join(pair[0])] == random_hand_best_scores[0] or hand_scores[
                    ''.join(pair[1])] == random_hand_best_scores[0]:
                    win_frequencies_pairs[key] += 0.5/2

                if hand_scores[''.join(pair[0])] > random_hand_best_scores[1] or hand_scores[
                    ''.join(pair[1])] > random_hand_best_scores[1]:
                    win_frequencies_pairs[key] += 1/2
                elif hand_scores[''.join(pair[0])] == random_hand_best_scores[1] and hand_scores[
                    ''.join(pair[1])] == random_hand_best_scores[1]:
                    win_frequencies_pairs[key] += 0.66/2
                elif hand_scores[''.join(pair[0])] == random_hand_best_scores[1] or hand_scores[
                    ''.join(pair[1])] == random_hand_best_scores[1]:
                    win_frequencies_pairs[key] += 0.5/2
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        # breakpoint()
    # divide the win frequency by the number of trials to get the win frequency
    for hand in win_frequencies.keys():
        win_frequencies[hand] /= number_of_trials
    for pair in win_frequencies_pairs.keys():
        win_frequencies_pairs[pair] /= number_of_trials

    # print(f"Process Win Frequencies: {win_frequencies}")
    # print(f"Process Win Frequencies Pairs: {win_frequencies_pairs}")
    return win_frequencies, win_frequencies_pairs



def plo6equities(hands, number_of_trials, board=[], opponent_hands=[]):
    '''
    :param hands: list of lists of strings, each list of strings is a hand
    :param number_of_trials:
    :param board: list
    :return:
    '''
    # print("Hands: ", hands, " Type: ", type(hands))
    # print("Number of Trials: ", number_of_trials, " Type: ", type(number_of_trials))
    # print("Board: ", board, " Type: ", type(board))
    # print("Opponent Hands: ", len(opponent_hands), " Type: ", type(opponent_hands))
    if type(hands[0]) == str:
        hands = [preprocess_hand(hand) for hand in hands]
    try:
        all_cards_in_play = [card for hand in hands for card in hand]
        if len(board):
            all_cards_in_play += board
        deck = generate_deck(all_cards_in_play)
        win_frequencies = {''.join(hand): 0 for hand in hands}
        win_frequencies_pairs = {''.join(a) + '_' + ''.join(b): 0 for a, b in combinations(hands, 2)}
        for _ in range(number_of_trials):
            # Pick a random hand from the deck
            if len(opponent_hands):
                random_hand = random.sample(opponent_hands, 1)[0]
                if type(random_hand) == str:
                    random_hand = preprocess_hand(random_hand)
                deck = generate_deck(all_cards_in_play + random_hand)
                random_board = board + random.sample(deck, 5 - len(board))
            else:
                random.shuffle(deck)
                random_hand = deck[:6]
                # Pick a random board from the deck
                random_board = board + deck[6:(11 - len(board))]
            random_hand_best_score = best_score_of_hand_on_board(random_hand, random_board, scores.score_dict)
            # random_hand_category = best_category_of_hand_on_board(random_hand, random_board)
            # print(random_hand, random_board, random_hand_category)
            # Calculate best score for each hand on the random board
            hand_scores = {''.join(hand): best_score_of_hand_on_board(hand, random_board, scores.score_dict) for hand in
                           hands}
            # hand_categories = {''.join(hand): best_category_of_hand_on_board(hand, random_board) for hand in hands}
            # print('Your Hand Categories: ', hand_categories)
            # check which hands have score higher than the random hand, if the score is higher, increment the win frequency, if it is equal, increment the win frequency by 0.5
            for hand, score in hand_scores.items():
                if score > random_hand_best_score:
                    win_frequencies[hand] += 1
                elif score == random_hand_best_score:
                    win_frequencies[hand] += 0.5
            # check which pair of hands have score higher than the random hand (either one could be higher), if the score is higher, increment the win frequency, if it is equal, increment the win frequency by 0.5, if all three are equal, increment the win frequency by 0.66
            pairs = combinations(hands, 2)
            for pair in pairs:
                key = ''.join(pair[0]) + '_' + ''.join(pair[1])
                if hand_scores[''.join(pair[0])] > random_hand_best_score or hand_scores[
                    ''.join(pair[1])] > random_hand_best_score:
                    win_frequencies_pairs[key] += 1
                elif hand_scores[''.join(pair[0])] == random_hand_best_score and hand_scores[
                    ''.join(pair[1])] == random_hand_best_score:
                    win_frequencies_pairs[key] += 0.66
                elif hand_scores[''.join(pair[0])] == random_hand_best_score or hand_scores[
                    ''.join(pair[1])] == random_hand_best_score:
                    win_frequencies_pairs[key] += 0.5
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        # breakpoint()
    # divide the win frequency by the number of trials to get the win frequency
    for hand in win_frequencies.keys():
        win_frequencies[hand] /= number_of_trials
    for pair in win_frequencies_pairs.keys():
        win_frequencies_pairs[pair] /= number_of_trials

    # print(f"Process Win Frequencies: {win_frequencies}")
    # print(f"Process Win Frequencies Pairs: {win_frequencies_pairs}")
    return win_frequencies, win_frequencies_pairs


def plo6equities_3h(hands, number_of_trials, board=[], opponent_hands=[]):
    '''
    :param hands: list of lists of strings, each list of strings is a hand
    :param number_of_trials:
    :param board: list
    :return:
    '''
    # print("Hands: ", hands, " Type: ", type(hands))
    # print("Number of Trials: ", number_of_trials, " Type: ", type(number_of_trials))
    # print("Board: ", board, " Type: ", type(board))
    # print("Opponent Hands: ", len(opponent_hands), " Type: ", type(opponent_hands))
    if type(hands[0]) == str:
        hands = [preprocess_hand(hand) for hand in hands]
    try:
        all_cards_in_play = [card for hand in hands for card in hand]
        if len(board):
            all_cards_in_play += board
        deck = generate_deck(all_cards_in_play)
        win_frequencies = {''.join(hand): 0 for hand in hands}
        win_frequencies_pairs = {''.join(a) + '_' + ''.join(b): 0 for a, b in combinations(hands, 2)}
        for _ in range(number_of_trials):
            # Pick a random hand from the deck
            if len(opponent_hands):
                random_hands = random.sample(opponent_hands, 2)
                random_hand = random_hands[0]
                random_hand2 = random_hands[1]
                if type(random_hand) == str:
                    random_hand = preprocess_hand(random_hand)
                    random_hand2 = preprocess_hand(random_hands[1])
                deck = generate_deck(all_cards_in_play + random_hand + random_hand2)
                random_board = board + random.sample(deck, 5 - len(board))
            else:
                random.shuffle(deck)
                random_hand = deck[:6]
                random_hand2 = deck[6:12]
                # Pick a random board from the deck
                random_board = board + deck[12:(17 - len(board))]
            random_hand_best_scores = [best_score_of_hand_on_board(h, random_board, scores.score_dict) for h in
                                       [random_hand, random_hand2]]
            hand_scores = {''.join(hand): best_score_of_hand_on_board(hand, random_board, scores.score_dict) for hand in
                           hands}
            # hand_categories = {''.join(hand): best_category_of_hand_on_board(hand, random_board) for hand in hands}
            # print('Your Hand Categories: ', hand_categories)
            # check which hands have score higher than the random hand, if the score is higher, increment the win frequency, if it is equal, increment the win frequency by 0.5
            for hand, score in hand_scores.items():
                if score < max(random_hand_best_scores):
                    pass
                elif score > max(random_hand_best_scores):
                    win_frequencies[hand] += 1
                elif max(random_hand_best_scores) == score and score == min(random_hand_best_scores):
                    win_frequencies[hand] += 0.33

            # check which pair of hands have score higher than the random hand (either one could be higher), if the score is higher, increment the win frequency, if it is equal, increment the win frequency by 0.5, if all three are equal, increment the win frequency by 0.66
            pairs = combinations(hands, 2)
            for pair in pairs:
                key = ''.join(pair[0]) + '_' + ''.join(pair[1])
                if hand_scores[''.join(pair[0])] > random_hand_best_scores[0] or hand_scores[
                    ''.join(pair[1])] > random_hand_best_scores[0]:
                    win_frequencies_pairs[key] += 1/2
                elif hand_scores[''.join(pair[0])] == random_hand_best_scores[0] and hand_scores[
                    ''.join(pair[1])] == random_hand_best_scores[0]:
                    win_frequencies_pairs[key] += 0.66/2
                elif hand_scores[''.join(pair[0])] == random_hand_best_scores[0] or hand_scores[
                    ''.join(pair[1])] == random_hand_best_scores[0]:
                    win_frequencies_pairs[key] += 0.5/2

                if hand_scores[''.join(pair[0])] > random_hand_best_scores[1] or hand_scores[
                    ''.join(pair[1])] > random_hand_best_scores[1]:
                    win_frequencies_pairs[key] += 1/2
                elif hand_scores[''.join(pair[0])] == random_hand_best_scores[1] and hand_scores[
                    ''.join(pair[1])] == random_hand_best_scores[1]:
                    win_frequencies_pairs[key] += 0.66/2
                elif hand_scores[''.join(pair[0])] == random_hand_best_scores[1] or hand_scores[
                    ''.join(pair[1])] == random_hand_best_scores[1]:
                    win_frequencies_pairs[key] += 0.5/2
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        # breakpoint()
    # divide the win frequency by the number of trials to get the win frequency
    for hand in win_frequencies.keys():
        win_frequencies[hand] /= number_of_trials
    for pair in win_frequencies_pairs.keys():
        win_frequencies_pairs[pair] /= number_of_trials

    # print(f"Process Win Frequencies: {win_frequencies}")
    # print(f"Process Win Frequencies Pairs: {win_frequencies_pairs}")
    return win_frequencies, win_frequencies_pairs


def ploequities_for_hand_ranks(hand, number_of_trials, board=[]):
    '''
    :param hand: list of strings, each string is a card: Ex: Ah
    :param number_of_trials:
    :param board: list
    :return:
    '''
    # print("Hands: ", hands, " Type: ", type(hands))
    # print("Number of Trials: ", number_of_trials, " Type: ", type(number_of_trials))
    # print("Board: ", board, " Type: ", type(board))
    # print("Opponent Hands: ", len(opponent_hands), " Type: ", type(opponent_hands))
    if type(hand) == str:
        hand = preprocess_hand(hand)
    all_cards_in_play = [card for card in hand]
    if len(board):
        all_cards_in_play += board
    deck = generate_deck(all_cards_in_play)
    game_type = determine_game_type(hand)
    win_frequency = 0

    random_boards = [board + random.sample(deck, 5 - len(board)) for _ in range(number_of_trials)]
    all_hand_scores = [best_score_of_hand_on_board(hand, random_board, scores.score_dict) for random_board in
                       random_boards]
    # check if the first score is the highest score, if yes, add 1 to the win frequency
    return sum(all_hand_scores) / number_of_trials


def Hand_category_frequencies(board, hero_hands, num_sims=10000):
    t1 = time.time()
    if isinstance(board, str):
        board = preprocess_hand(board)
    example_hand = hero_hands[0]
    if isinstance(example_hand, str):
        hero_hands = preprocess_hands(hero_hands)
    # Deduce game type from the number of cards in hero hands
    game_type, num_cards_per_hand = determine_game_type(hero_hands[0])
    # Step 1: Generate deck by removing the board, and generate 10k random hands for opponent
    deck = generate_deck([card for hand in hero_hands for card in hand] + board)
    num_opponent_hands = num_sims
    # generate 10k random hands for the opponent, each hand contains num_cards_per_hand cards
    opponent_hands_and_boards = [random.sample(deck, num_cards_per_hand + (5 - len(board))) for _ in
                                 range(num_opponent_hands)]
    # Step 2: Filter out top x% by score, note the cutoff score
    category_list = [
        best_category_of_hand_on_board(hand[:num_cards_per_hand], board + hand[-(5 - len(board)):]) for hand in
        opponent_hands_and_boards]

    # Get frequency of each hand category, and save it in a dictionary
    opponent_hist = {category: category_list.count(category) for category in set(category_list)}
    opponent_percentages = {k: (v / sum(opponent_hist.values())) * 100 for k, v in opponent_hist.items()}
    df_opponent = pd.DataFrame(list(opponent_percentages.items()), columns=['Category', 'Opponent'])
    df_opponent.set_index('Category', inplace=True)
    all_categories = set(df_opponent.index)  # Start with opponent categories

    # Calculate hero hand categories, by getting the category for each hero hand, in for all the possible turns and rivers
    all_possible_runouts = [board + list(runout) for runout in list(combinations(deck, 5 - len(board)))]
    for hero_hand in hero_hands:
        category_list = [best_category_of_hand_on_board(hero_hand, runout) for runout in all_possible_runouts]
        hist = {category: category_list.count(category) for category in all_categories}  # Use all_categories
        total = sum(hist.values())
        all_categories.update(hist.keys())
        for category in all_categories:
            if category not in df_opponent.index:
                df_opponent.loc[category] = [0]

        percentages = {category: (hist.get(category, 0) / total * 100) if total else 0 for category in all_categories}
        df_opponent[''.join(hero_hand)] = df_opponent.index.map(percentages).fillna(0)
    df_opponent.index = pd.CategoricalIndex(df_opponent.index, categories=list(reversed(scores.hierarchy)),
                                            ordered=True)
    # Add a column 'Atleast One', in this column we will calculate the probability that at least one opponent has a hand of this category
    num_opponents = 6 - len(hero_hands)
    df_opponent['Atleast One'] = 1 - (1 - df_opponent['Opponent'] / 100) ** num_opponents
    df_opponent['Atleast One'] = df_opponent['Atleast One'] * 100
    # round to two decimal places
    df_opponent = df_opponent.round(2)

    # Sort the DataFrame by its index
    df_opponent.sort_index(ascending=True, inplace=True)
    return df_opponent


def aggregate_results(results, number_of_processes):
    aggregated_win_frequencies = {}
    aggregated_win_frequencies_pairs = {}

    # print("Starting aggregation of results...")

    # Sum up the frequencies from each result
    for win_freqs, win_freqs_pairs in results:
        for hand, freq in win_freqs.items():
            if hand not in aggregated_win_frequencies:
                aggregated_win_frequencies[hand] = []
            aggregated_win_frequencies[hand].append(freq)

        for pair, freq in win_freqs_pairs.items():
            if pair not in aggregated_win_frequencies_pairs:
                aggregated_win_frequencies_pairs[pair] = []
            aggregated_win_frequencies_pairs[pair].append(freq)

    # Calculate the average win rate across all processes
    for hand, freqs in aggregated_win_frequencies.items():
        aggregated_win_frequencies[hand] = sum(freqs) / len(freqs)
        # print(f"Averaged {hand}: {aggregated_win_frequencies[hand]}")

    for pair, freqs in aggregated_win_frequencies_pairs.items():
        aggregated_win_frequencies_pairs[pair] = sum(freqs) / len(freqs)
        # print(f"Averaged {pair}: {aggregated_win_frequencies_pairs[pair]}")

    # print("\nFinished calculating averages.")
    return aggregated_win_frequencies, aggregated_win_frequencies_pairs


def run_parallel_plo6equities(hands, total_number_of_trials, number_of_processes, board=[], opponent_hands=[]):
    """Run PLO6 equity calculation using persistent global process pool."""
    trials_per_process = total_number_of_trials // number_of_processes
    executor = get_global_executor(number_of_processes)
    
    futures = [executor.submit(plo6equities, hands, trials_per_process, board, opponent_hands) 
               for _ in range(number_of_processes)]
    results = [future.result() for future in futures]

    return aggregate_results(results, total_number_of_trials)

def run_parallel_plo6equities_3h(hands, total_number_of_trials, number_of_processes, board=[], opponent_hands=[]):
    """Run PLO6 3-handed equity calculation using persistent global process pool."""
    trials_per_process = total_number_of_trials // number_of_processes
    executor = get_global_executor(number_of_processes)
    
    futures = [executor.submit(plo6equities_3h, hands, trials_per_process, board, opponent_hands) 
               for _ in range(number_of_processes)]
    results = [future.result() for future in futures]

    return aggregate_results(results, total_number_of_trials)

def run_parallel_plo5equities_3h(hands, total_number_of_trials, number_of_processes, board=[], opponent_hands=[]):
    """Run PLO5 3-handed equity calculation using persistent global process pool."""
    trials_per_process = total_number_of_trials // number_of_processes
    executor = get_global_executor(number_of_processes)
    
    futures = [executor.submit(plo5equities_3h, hands, trials_per_process, board, opponent_hands) 
               for _ in range(number_of_processes)]
    results = [future.result() for future in futures]

    return aggregate_results(results, total_number_of_trials)


def run_parallel_plo5equities(hands, total_number_of_trials, number_of_processes, board=[], opponent_hands=[]):
    """Run PLO5 equity calculation using persistent global process pool."""
    trials_per_process = total_number_of_trials // number_of_processes
    executor = get_global_executor(number_of_processes)
    
    futures = [executor.submit(plo5equities, hands, trials_per_process, board, opponent_hands) 
               for _ in range(number_of_processes)]
    results = [future.result() for future in futures]

    return aggregate_results(results, total_number_of_trials)


@profile
def score_hands(hands, boards):
    # return a df where index is board and columns are hands and values are best scores of hand on board
    rows = []
    processed = dict()
    for board in boards:
        if ''.join(board) in processed:
            row = processed[''.join(board)]
            rows.append(row)
            continue
        else:
            row = {'board': ''.join(board)}
            hand_scores = [best_score_of_hand_on_board(hand, board, scores.score_dict) for hand in hands]
            for i, hand in enumerate(hands):
                row[''.join(hand)] = hand_scores[i]
            processed[''.join(board)] = row
        rows.append(row)

    return pd.DataFrame(rows).set_index('board')



def calculate_equity(df1, df2, hero_hand):

    if type(hero_hand) == list:
        hero_hand = ''.join(hero_hand)
    # equity = 0.0
    # for i in range(len(df1)):
    #     board1 = df1.index[i]
    #     board2 = df2.index[i]
    #     # print(f"Board 1: {board1}, Board 2: {board2}")
    #
    #     max_score_board1 = df1.iloc[i].max()
    #     max_score_board2 = df2.iloc[i].max()
    #     num_max_values_board1 = (df1.iloc[i] == max_score_board1).astype(int).sum()
    #     num_max_values_board2 = (df2.iloc[i] == max_score_board2).astype(int).sum()
    #     hero_score_board1 = df1.at[board1, hero_hand].values[0]
    #     hero_score_board2 = df2.at[board2, hero_hand].values[0]
    #
    #     # print(f"Hero score on Board 1: {hero_score_board1}, Hero score on Board 2: {hero_score_board2}")
    #
    #     if max_score_board1 <= hero_score_board1:
    #         equity += 0.5 / num_max_values_board1
    #     if max_score_board2 <= hero_score_board2:
    #         equity += 0.5 / num_max_values_board2

    # return equity / len(df1)

    max_scores_board1 = df1.max(axis=1).to_numpy()
    max_scores_board2 = df2.max(axis=1).to_numpy()
    num_max_values_board1 = (df1.to_numpy() == max_scores_board1[:, None]).sum(axis=1)
    num_max_values_board2 = (df2.to_numpy() == max_scores_board2[:, None]).sum(axis=1)
    hero_scores_board1 = df1[hero_hand].to_numpy()
    hero_scores_board2 = df2[hero_hand].to_numpy()

    equity_board1 = (max_scores_board1 <= hero_scores_board1) * (0.5 / num_max_values_board1)
    equity_board2 = (max_scores_board2 <= hero_scores_board2) * (0.5 / num_max_values_board2)

    equity = equity_board1.sum() + equity_board2.sum()
    return equity / len(df1)


@profile
def calculate_scenarios(street, board1, board2, hero, dead_cards, num_scenarios, num_opponents):
    if isinstance(board1, str):
        board1 = preprocess_hand(board1)
    if isinstance(board2, str):
        board2 = preprocess_hand(board2)
    if isinstance(hero, str):
        hero = preprocess_hand(hero)

    assert not any(card in board1 for card in board2), "Board 1 and Board 2 have common cards"
    assert not any(card in hero for card in board1), "Hero hand and Board 1 have common cards"
    assert not any(card in hero for card in board2), "Hero hand and Board 2 have common cards"
    assert not any(card in dead_cards for card in board1), "Dead cards and Board 1 have common cards"
    assert not any(card in dead_cards for card in board2), "Dead cards and Board 2 have common cards"
    assert not any(card in dead_cards for card in hero), "Dead cards and Hero hand have common cards"


    if street == 'flop':
        assert len(board1) == 3, "Board 1 should have 3 cards for flop"
        assert len(board2) == 3, "Board 2 should have 3 cards for flop"
    elif street == 'turn':
        assert len(board1) == 4, "Board 1 should have 4 cards for turn"
        assert len(board2) == 4, "Board 2 should have 4 cards for turn"
    else:
        return "Invalid street"

    all_cards_in_play = board1 + board2 + hero + dead_cards
    equities = []
    deck = generate_deck(all_cards_in_play)
    for _ in range(num_scenarios):
        random.shuffle(deck)
        opponent_hands = [deck[i:i + len(hero)] for i in range(0, len(hero) * num_opponents, len(hero))]
        remaining_deck = generate_deck(all_cards_in_play + [card for hand in opponent_hands for card in hand])
        board1_combos = []
        board2_combos = []
        if street == 'flop':
            b1combinations = list(combinations(remaining_deck, 2))
            for combo in b1combinations:
                now_remaining_deck = [card for card in remaining_deck if card not in list(combo)]
                for combo2 in list(combinations(now_remaining_deck, 2)):
                    board1_combos.append(board1 + list(combo))
                    board2_combos.append(board2 + list(combo2))
        elif street == 'turn':
            b1combinations = list(combinations(remaining_deck, 1))
            for combo in b1combinations:
                now_remaining_deck = [card for card in remaining_deck if card not in list(combo)]
                for combo2 in list(combinations(now_remaining_deck, 1)):
                    board1_combos.append(board1 + list(combo))
                    board2_combos.append(board2 + list(combo2))

        # print(f"Number of board combinations: {len(board1_combos)}")
        df1 = score_hands([hero] + opponent_hands, board1_combos)
        df2 = score_hands([hero] + opponent_hands, board2_combos)

        equity = calculate_equity(df1, df2, ''.join(hero))
        equities.append(equity)
        # print('Hero Hand: ', ''.join(hero), 'Equity: ', equity)

    return equities

def calculate_scenarios_multithreaded(street, board1, board2, hero, dead_cards, num_scenarios, num_opponents, num_workers=8):
    scenarios_per_worker = num_scenarios // num_workers
    futures = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for _ in range(num_workers):
            futures.append(executor.submit(calculate_scenarios, street, board1, board2, hero, dead_cards, scenarios_per_worker, num_opponents))

    equities = []
    for future in futures:
        equities.extend(future.result())

    return equities

def generate_random_hand(deck, num_cards):
    return random.sample(deck, num_cards)


def generate_random_scenario():
    deck = generate_deck([])  # Generate a full deck of cards
    hero_hand = generate_random_hand(deck, 6)
    dead_cards = []
    remaining_deck = generate_deck(hero_hand + dead_cards)
    flop1 = generate_random_hand(remaining_deck, 3)
    remaining_deck = generate_deck(hero_hand + dead_cards + flop1)
    flop2 = generate_random_hand(remaining_deck, 3)
    return hero_hand, flop1, flop2

def test_random_scenario():

    for i in range(25):
        hero, board1, board2 = generate_random_scenario()
        # pick a random opponent hand
        deck = generate_deck(hero + board1 + board2)
        ally_hand = generate_random_hand(deck, 6)
        dead_cards = []
        print(f"Hero Hand: {hero}")
        print(f"Flop 1: {board1}")
        print(f"Flop 2: {board2}")
        t1 = time.time()
        equities = calculate_scenarios_multithreaded('flop', board1, board2, hero, dead_cards = dead_cards,  num_scenarios=150, num_opponents=5, num_workers=12)
        # equities = calculate_scenarios('flop', board1, board2, hero, dead_cards = dead_cards,  num_scenarios=32, num_opponents=5)
        average_equity = sum(equities) / len(equities)
        print(f"Average Equity: {average_equity*100}, time taken: {time.time() - t1}")

    sorted_equities = sorted(equities)
    equity_threshold = 0.4
    # print graph
'''
TODO: Comment out line profiler import
remove @profile tags

'''

if __name__ == '__main__':

    test_random_scenario()

    # deck = generate_deck([])
    # n = 4
    # num_cards_per_hand = 5
    # hands = [deck[i * num_cards_per_hand:(i + 1) * num_cards_per_hand] for i in range(n)]
    # # get last 3 cards from the deck as board
    # # board = deck[-3:]
    # board = []
    # print([''.join(hand) for hand in hands])
    # print(board)
    # total_number_of_trials = 5000
    # number_of_processes = 10  # Adjust based on the number of available CPU cores
    # remaining_deck = generate_deck([card for hand in hands for card in hand] + board)
    # opponent_hands = [random.sample(remaining_deck, num_cards_per_hand) for _ in range(total_number_of_trials)]
    # data = {
    #     "board": board,
    #     "hero_hands": hands,
    #     "sprlist": [0.5, 1, 2, 3, 4],
    #     "xlist": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    #     "total_number_of_trials": total_number_of_trials,
    #     "number_of_processes": number_of_processes
    # }
    # board = data.get('board')
    # hero_hands = data.get('hero_hands')
    # sprlist = data.get('sprlist')
    # xlist = data.get('xlist')
    # total_number_of_trials = data.get('total_number_of_trials')
    # number_of_processes = data.get('number_of_processes')
    # t1 = time.time()
    # # for i in range(10):
    # #     hero_equities_against_calling_range, foldequity = run_parallel_Equity_vs_best_x_hands_and_fold_equity(board, 0.65, hands, 10000, 12)
    # #     print(hero_equities_against_calling_range)
    # #     print(foldequity)
    # # df = Hand_category_frequencies(board,hero_hands=hands,num_sims=5000)
    # hu_equities = run_parallel_plo5equities(hands, total_number_of_trials, number_of_processes, board,
    #                                               opponent_hands)
    # print(hu_equities)
    # print('Time taken: ', time.time() - t1)
    # t1 = time.time()
    # threeway_equities = run_parallel_plo5equities_3h(hands, total_number_of_trials, number_of_processes, board,
    #                                               opponent_hands)
    # print(threeway_equities)
    #
    # print('Time taken: ', time.time() - t1)
    # # print(df)
