board_categories_to_hand_strengths = {
    "Trips Board": ["Quads", "AA,KK", "TT-QQ", "22-99", "Others"],
    "Paired Board": ["Quads", "Full House", "Trips", "Others"],
    "Unpaired Board - Monotone": ["Nut Flush", "Flush", "Set", "Others"],
    "Unpaired Board - Two Tone (Straight)": ["Nut Straight", "Straight", "Set", "Flush Draw", "Wrap", "Others"],
    "Unpaired Board - Two Tone (Non-Straight)": ["Set", "Top Two", "Nut Flush Draw", "Flush Draw", "Wrap", "Others"],
    "Unpaired Board - Rainbow (Straight)": ["Nut Straight", "Straight", "Set", "Wrap", "Others"],
    "Unpaired Board - Rainbow (Non-Straight)": ["Set", "Top Two", "Wrap", "Others"]
}
from multithread_ploequities2 import best_category_of_hand_on_board
from concurrent.futures import ProcessPoolExecutor

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
    num_straights = sum([1 for rank in ranks_not_in_board if best_category_of_hand_on_board(hand_, board_ + [rank + 'c']) == 'Straight'])
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
    if max(rank_indices) - min(rank_indices) <=4 and len(set(rank_indices)) == 3:
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

def break_hands_into_categories(hands, board):
    # Example: Break a list of hands into categories based on the board
    board_category = categorize_board(board)
    hand_categories = [hand_strength_category(hand, board, board_category) for hand in hands]
    return_dict = {}
    for hand, category in zip(hands, hand_categories):
        if category not in return_dict.keys():
            return_dict[category] = [hand]
        else:
            return_dict[category].append(hand)
    return return_dict

def categorize_hands_chunk(hands_chunk, board):
    """
    Helper function to categorize a chunk of hands.
    """
    board_category = categorize_board(board)
    hand_categories = [(hand, hand_strength_category(hand, board, board_category)) for hand in hands_chunk]
    return hand_categories

def break_hands_into_categories_mp(hands, board, num_processes = 4):
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


if __name__ == "__main__":
    # Example usage of the board categorization and hand evaluation functions
    ranks = "A23456789TJQK"
    suits = "cdhs"
    deck = [rank + suit for rank in ranks for suit in suits]
    import random
    random.shuffle(deck)
    board = deck[:3]
    hand = deck[3:8]
    board_category = categorize_board(board)
    print("Board:", board, "Category:", board_category)
    print("Hand:", hand)
    print("Hand Strength:", hand_strength_category(hand, board, board_category))
    remaining_deck = [card for card in deck if card not in board]
    # generate 1000 random 5 card hands
    hands = [random.sample(remaining_deck, 5) for _ in range(50000)]
    import time
    t1 = time.time()
    hand_categories = break_hands_into_categories(hands, board)
    t2 = time.time()
    print("Time taken for sequential categorization:", t2 - t1)
    t3 = time.time()
    hand_categories_mp = break_hands_into_categories_mp(hands, board, num_processes=4)
    t4 = time.time()
    print("Time taken for parallel categorization:", t4 - t3)
    # check if the results are same
    assert hand_categories == hand_categories_mp


