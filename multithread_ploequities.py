# In this file we will have multiple functions to calculate frequencies of hand strengths by the river
from concurrent.futures import ProcessPoolExecutor
import random
import generating_list as scores
from itertools import combinations, product
import time


def best_score_of_hand_on_board(hand, board, score_dict):
    from itertools import combinations
    best_score = 0
    # Generate all possible 2-card combinations from the hand and 3-card combinations from the board
    for hand_combo in combinations(hand, 2):
        for board_combo in combinations(board, 3):
            # Combine hand and board cards to form a 5-card poker hand
            full_hand = hand_combo + board_combo
            # Convert to a string representation or any format that matches the score_dict keys
            hand_str = scores.sort_hand(''.join(sorted(full_hand)))
            # Update best score if this combo's score is higher
            best_score = max(best_score, score_dict.get(hand_str, 0))
    return best_score

def best_category_of_hand_on_board(hand, board,):
    from itertools import combinations
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
            best_score = max(best_score, scores.score_dict.get(hand_str, 0))
            if best_score == scores.score_dict.get(hand_str, 0):
                best_combo = hand_str
                # print(best_combo, best_score)
            # get the hand category, for that get the hand_str for the best score

    return scores.category_dict.get(best_combo, 0)

def test_best_score_function():
    # lets assume some hands and a board
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

def plo5equities(hands, dead_cards, number_of_trials):
    try:
        all_cards_in_play = [card for hand in hands for card in hand]
        deck = generate_deck(all_cards_in_play)
        win_frequencies = {''.join(hand): 0 for hand in hands}
        win_frequencies_pairs = {''.join(a)+'_'+''.join(b): 0 for a, b in combinations(hands, 2)}
        for _ in range(number_of_trials):
            # Pick a random hand from the deck
            random.shuffle(deck)
            random_hand = deck[:5]
            # Pick a random board from the deck
            random_board = deck[6:11]
            random_hand_best_score = best_score_of_hand_on_board(random_hand, random_board, scores.score_dict)
            # random_hand_category = best_category_of_hand_on_board(random_hand, random_board)
            # print(random_hand, random_board, random_hand_category)
            # Calculate best score for each hand on the random board
            hand_scores = {''.join(hand): best_score_of_hand_on_board(hand, random_board, scores.score_dict) for hand in hands}
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
                if hand_scores[''.join(pair[0])] > random_hand_best_score or hand_scores[''.join(pair[1])] > random_hand_best_score:
                    win_frequencies_pairs[key] += 1
                elif hand_scores[''.join(pair[0])] == random_hand_best_score and hand_scores[''.join(pair[1])] == random_hand_best_score:
                    win_frequencies_pairs[key] += 0.66
                elif hand_scores[''.join(pair[0])] == random_hand_best_score or hand_scores[''.join(pair[1])] == random_hand_best_score:
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



def plo6equities(hands, dead_cards, number_of_trials):
    try:
        all_cards_in_play = [card for hand in hands for card in hand]
        deck = generate_deck(all_cards_in_play)
        win_frequencies = {''.join(hand): 0 for hand in hands}
        win_frequencies_pairs = {''.join(a)+'_'+''.join(b): 0 for a, b in combinations(hands, 2)}
        for _ in range(number_of_trials):
            # Pick a random hand from the deck
            random.shuffle(deck)
            random_hand = deck[:6]
            # Pick a random board from the deck
            random_board = deck[6:11]
            random_hand_best_score = best_score_of_hand_on_board(random_hand, random_board, scores.score_dict)
            # random_hand_category = best_category_of_hand_on_board(random_hand, random_board)
            # print(random_hand, random_board, random_hand_category)
            # Calculate best score for each hand on the random board
            hand_scores = {''.join(hand): best_score_of_hand_on_board(hand, random_board, scores.score_dict) for hand in hands}
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
                if hand_scores[''.join(pair[0])] > random_hand_best_score or hand_scores[''.join(pair[1])] > random_hand_best_score:
                    win_frequencies_pairs[key] += 1
                elif hand_scores[''.join(pair[0])] == random_hand_best_score and hand_scores[''.join(pair[1])] == random_hand_best_score:
                    win_frequencies_pairs[key] += 0.66
                elif hand_scores[''.join(pair[0])] == random_hand_best_score or hand_scores[''.join(pair[1])] == random_hand_best_score:
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


def run_parallel_plo6equities(hands, dead_cards, total_number_of_trials, number_of_processes):
    trials_per_process = total_number_of_trials // number_of_processes
    with ProcessPoolExecutor(max_workers=number_of_processes) as executor:
        futures = [executor.submit(plo6equities, hands, dead_cards, trials_per_process) for _ in range(number_of_processes)]
        results = [future.result() for future in futures]

    return aggregate_results(results, total_number_of_trials)

def run_parallel_plo5equities(hands, dead_cards, total_number_of_trials, number_of_processes):
    trials_per_process = total_number_of_trials // number_of_processes
    with ProcessPoolExecutor(max_workers=number_of_processes) as executor:
        futures = [executor.submit(plo5equities, hands, dead_cards, trials_per_process) for _ in range(number_of_processes)]
        results = [future.result() for future in futures]

    return aggregate_results(results, total_number_of_trials)



if __name__ == '__main__':
    deck = generate_deck([])
    n = 4
    num_cards_per_hand = 5
    hands = [deck[i * num_cards_per_hand:(i + 1) * num_cards_per_hand] for i in range(n)]
    print([''.join(hand) for hand in hands])
    dead_cards = []
    total_number_of_trials = 5000
    number_of_processes = 10  # Adjust based on the number of available CPU cores
    for i in range(10):
        t1 = time.time()
        win_frequencies, win_frequencies_pairs = run_parallel_plo5equities(hands, dead_cards, total_number_of_trials,
                                                                           number_of_processes)
        print(
            f"Sample win frequencies in process: {dict(list(win_frequencies.items())[:2])}")  # Print a sample of the outcomes for debugging

        print(win_frequencies, win_frequencies_pairs)
        print('Time taken: ', time.time() - t1)