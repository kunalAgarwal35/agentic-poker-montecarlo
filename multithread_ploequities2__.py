# In this file we will have multiple functions to calculate frequencies of hand strengths by the river
from concurrent.futures import ProcessPoolExecutor
import random
import generating_list as scores
from itertools import combinations, product
import time
import pandas as pd


# from line_profiler_pycharm import profile

def best_score_of_hand_on_board(hand, board, score_dict):
    from itertools import combinations
    best_score = 0
    # Generate all possible 2-card combinations from the hand and 3-card combinations from the board
    for hand_combo in combinations(hand, 2):
        for board_combo in combinations(board, 3):
            # Combine hand and board cards to form a 5-card poker hand
            full_hand = hand_combo + board_combo
            # Convert to a string representation or any format that matches the score_dict keys
            hand_str = scores.sort_hand(full_hand)
            # Update best score if this combo's score is higher
            best_score = max(best_score, score_dict.get(hand_str, 0))
    return best_score


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


def best_category_of_hand_on_board(hand, board, ):
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


# Assuming `generate_list_of_hands` and `sort_hand` are available and correct
# Assuming `calculate_hand_score` returns a numeric score where higher is better

def generate_deck(exclude_cards):
    suits = 'shdc'
    ranks = '23456789TJQKA'
    deck = [r + s for r in ranks for s in suits if (r + s) not in exclude_cards]
    random.shuffle(deck)
    return deck


def plo5equities(hands, number_of_trials, board=[]):
    '''
    :param hands: list of lists of strings, each list of strings is a hand
    :param number_of_trials:
    :param board:
    :return:
    '''
    try:
        all_cards_in_play = [card for hand in hands for card in hand]
        if len(board):
            all_cards_in_play += board
        deck = generate_deck(all_cards_in_play)
        win_frequencies = {''.join(hand): 0 for hand in hands}
        win_frequencies_pairs = {''.join(a) + '_' + ''.join(b): 0 for a, b in combinations(hands, 2)}
        for _ in range(number_of_trials):
            # Pick a random hand from the deck
            random.shuffle(deck)
            random_hand = deck[:5]
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


def plo6equities(hands, number_of_trials, board=[]):
    '''
    :param hands: list of lists of strings, each list of strings is a hand
    :param number_of_trials:
    :param board:
    :return:
    '''
    try:
        all_cards_in_play = [card for hand in hands for card in hand]
        if len(board):
            all_cards_in_play += board
        deck = generate_deck(all_cards_in_play)
        win_frequencies = {''.join(hand): 0 for hand in hands}
        win_frequencies_pairs = {''.join(a) + '_' + ''.join(b): 0 for a, b in combinations(hands, 2)}
        for _ in range(number_of_trials):
            # Pick a random hand from the deck
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


def run_parallel_plo6equities(hands, total_number_of_trials, number_of_processes, board=[]):
    trials_per_process = total_number_of_trials // number_of_processes
    with ProcessPoolExecutor(max_workers=number_of_processes) as executor:
        futures = [executor.submit(plo6equities, hands, trials_per_process, board) for _ in range(number_of_processes)]
        results = [future.result() for future in futures]

    return aggregate_results(results, total_number_of_trials)


# def run_parallel_plo5equities(hands, total_number_of_trials, number_of_processes):
#     trials_per_process = total_number_of_trials // number_of_processes
#     with ProcessPoolExecutor(max_workers=number_of_processes) as executor:
#         futures = [executor.submit(plo5equities, hands, trials_per_process) for _ in range(number_of_processes)]
#         results = [future.result() for future in futures]
#
#     return aggregate_results(results, total_number_of_trials)

def run_parallel_plo5equities(hands, total_number_of_trials, number_of_processes, board=[]):
    trials_per_process = total_number_of_trials // number_of_processes
    with ProcessPoolExecutor(max_workers=number_of_processes) as executor:
        futures = [executor.submit(plo5equities, hands, trials_per_process, board) for _ in range(number_of_processes)]
        results = [future.result() for future in futures]

    return aggregate_results(results, total_number_of_trials)


if __name__ == '__main__':
    deck = generate_deck([])
    n = 4
    num_cards_per_hand = 5
    hands = [deck[i * num_cards_per_hand:(i + 1) * num_cards_per_hand] for i in range(n)]
    # get last 3 cards from the deck as board
    board = deck[-3:]
    print([''.join(hand) for hand in hands])
    print(board)
    dead_cards = []
    total_number_of_trials = 5000
    number_of_processes = 10  # Adjust based on the number of available CPU cores
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
    t1 = time.time()
    # for i in range(10):
    #     hero_equities_against_calling_range, foldequity = run_parallel_Equity_vs_best_x_hands_and_fold_equity(board, 0.65, hands, 10000, 12)
    #     print(hero_equities_against_calling_range)
    #     print(foldequity)
    # df = Hand_category_frequencies(board,hero_hands=hands,num_sims=5000)
    plo5_equities_old = run_parallel_plo5equities(hands, total_number_of_trials, number_of_processes, board)

    print(plo5_equities_old)
    print('Time taken: ', time.time() - t1)
    # print(df)
