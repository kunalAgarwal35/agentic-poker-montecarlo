import pickle
import os

score_file = 'score_dict.pkl'
category_file = 'category_dict.pkl'

# Function to save a dictionary to a pickle file
def save_to_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

# Function to load a dictionary from a pickle file
def load_from_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# def sort_hand(hand):
#     rank_of_cards = ['As', 'Ah', 'Ad', 'Ac', 'Ks', 'Kh', 'Kd', 'Kc', 'Qs', 'Qh', 'Qd', 'Qc',
#                      'Js', 'Jh', 'Jd', 'Jc', 'Ts', 'Th', 'Td', 'Tc', '9s', '9h', '9d', '9c',
#                      '8s', '8h', '8d', '8c', '7s', '7h', '7d', '7c', '6s', '6h', '6d', '6c',
#                      '5s', '5h', '5d', '5c', '4s', '4h', '4d', '4c', '3s', '3h', '3d', '3c',
#                      '2s', '2h', '2d', '2c']
#     if type(hand) == str:
#         hand = [hand[i:i+2] for i in range(0, len(hand), 2)]
#     return ''.join(sorted(hand, key=lambda x: rank_of_cards.index(x)))

# card_rank_indices = {card: index for index, card in enumerate([
#     'As', 'Ah', 'Ad', 'Ac', 'Ks', 'Kh', 'Kd', 'Kc', 'Qs', 'Qh', 'Qd', 'Qc',
#     'Js', 'Jh', 'Jd', 'Jc', 'Ts', 'Th', 'Td', 'Tc', '9s', '9h', '9d', '9c',
#     '8s', '8h', '8d', '8c', '7s', '7h', '7d', '7c', '6s', '6h', '6d', '6c',
#     '5s', '5h', '5d', '5c', '4s', '4h', '4d', '4c', '3s', '3h', '3d', '3c',
#     '2s', '2h', '2d', '2c'
# ])}

# def sort_hand(hand):
#     """
#     Sorts a hand based on the predefined card ranks.
#
#     Args:
#         hand (str or list): The poker hand to sort. Can be a string or a list of strings.
#
#     Returns:
#         str: A sorted hand as a string.
#     """
#     if isinstance(hand, str):
#         hand = [hand[i:i+2] for i in range(0, len(hand), 2)]
#     sorted_hand = sorted(hand, key=lambda x: card_rank_indices[x])
#     return ''.join(sorted_hand)

def sort_hand(hand):
    if type(hand) == str:
        hand = [hand[i:i+2] for i in range(0, len(hand), 2)]
    return ''.join(sorted(hand))

# Check if pickle files exist, and load them if they do
if os.path.exists(score_file) and os.path.exists(category_file):
    score_dict = load_from_pickle(score_file)
    category_dict = load_from_pickle(category_file)
else:
    from ranking import calculate_hand_score

    def generate_list_of_hands():
        # generate a list of all possible 5 card combos from a 52 card deck
        from itertools import combinations
        from ranking import Card
        deck = [Card(rank+suit) for rank in Card.ranks for suit in Card.suits]
        return list(combinations(deck, 5))


    list_of_hands = generate_list_of_hands()
    list_of_hands = [sort_hand(''.join([str(card) for card in hand])) for hand in list_of_hands]

    category_score_tuples = [(calculate_hand_score(hand), hand) for hand in list_of_hands]
    score_dict = {hand: score[1] for score, hand in category_score_tuples}
    category_dict = {hand: score[0] for score, hand in category_score_tuples}

    # Save the dictionaries to pickle files
    save_to_pickle(score_dict, score_file)
    save_to_pickle(category_dict, category_file)

#for each key in category_dict, get the score for the key, and this would be the key of the following dictionary. The value of the following dictionary would be the value of the category_dict
scores_to_category_map = {score_dict[key]: category_dict[key] for key in category_dict.keys()}
hierarchy  = ['High Card', 'One Pair', 'Two Pair', 'Three of a Kind', 'Straight', 'Flush', 'Full House', 'Four of a Kind', 'Straight Flush', 'Royal Flush']

