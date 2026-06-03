# In this file we will have multiple functions to calculate frequencies of hand strengths by the river

import random
import generating_list as scores
from itertools import combinations, product
import time
# PLO6 Equity calculate function: Inputs: list of hands, board, unknown_range, dead cards, number of trials
# List of hands will be a list of strings each string will contain 6 cards: hole cards
# dead cards are cards that are not in the deck, they are not in the hands and not in the board
# board is a list of cards, it can have 3, 4 or 5 cards
# unknown_range is a percentage expression, currently we only support 100%, so the trials will be run against a random hand from the deck
# number of trials is the number of times we will run the simulation
# In the beginning, we will initiate multiple variables, lets assume there are 4 hands in the list of hands
# Initiating variables: H1vsRandom, H2vsRandom, H3vsRandom, H4vsRandom, H1andH2vsRandom, H1andH3vsRandom, H1andH4vsRandom, H2andH3vsRandom, H2andH4vsRandom, H3andH4vsRandom
# each of these variables are keys in a dictionary, and the value at each key is number_of_wins
# number_of_wins is the number of times the hand won against the random hand, and in case of two hands, it is the number of times the team of two hands won against the random hand
# We will then run the simulation for the number of trials, and for each trial, we will pick a random hand from the deck, and a random board from the deck, to do this we will randomly pick 11 cards from a shuffled remaining deck
# We will assume first 5 cards to be the board and the remaining 6 cards to be the random hand
# To check which hand won, we need to calculate the best score of each hand on the board, and then we can compare the best scores for each hand to determine who won
# to calcualte the best score of each hand (Create separate function for this: best_score_of_hand_on_board)
# best_score_of_hand_on_board will take a hand and a board, and will return the best score of the hand on the board
# to calculate best score on board, we will generate all possible 2 card combinations of the hand, and all possible 3 card combinations of the board, we will pair all two cards with all 3 cards
# this will give us a string of 5 cards, and we can look up the score of this string in the score_dict
# we will then pick the highest score from all possible combinations


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
    return win_frequencies, win_frequencies_pairs


# Example usage
#select 4 random hands with 6 cards each

deck = generate_deck([])
n = 4
num_cards_per_hand = 6
hands = list()
for i in range(n):
    hands.append(deck[i * num_cards_per_hand:(i + 1) * num_cards_per_hand])
print(hands)


dead_cards = []
number_of_trials = 10000  # Adjust based on desired accuracy vs. runtime
for i in range(10):
    t1 = time.time()
    win_frequencies = plo6equities(hands, dead_cards, number_of_trials)
    print(win_frequencies)
    print('Time taken: ', time.time() - t1)
