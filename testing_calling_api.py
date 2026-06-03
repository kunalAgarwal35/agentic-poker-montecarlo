import unittest
from unittest.mock import patch
import calling_api
from api_host_db_local import health_check
import itertools
import random
import time


def test_health_check():
    print(calling_api.health_check())

def test_individual_strength():
    print("Testing individual_strengths Function")
    deck = [card for card in itertools.product('23456789TJQKA', 'shdc')]
    deck = ["".join(card) for card in deck]
    random.shuffle(deck)
    n = 4
    num_cards_per_hand = 5
    hands = [deck[i * num_cards_per_hand:(i + 1) * num_cards_per_hand] for i in range(n)]
    # get last 3 cards from the deck as board
    print([''.join(hand) for hand in hands])
    dead_cards = []
    t1 = time.time()
    # input type is list of strings:
    hands = [''.join(hand) for hand in hands]
    results = calling_api.individual_strengths(hands)
    t2 = time.time()
    print(results)
    print("Time taken for individual_strengths: ", t2-t1)
    # Now testing with a board
    board = ''.join(deck[-3:])
    print('Board: ',board)
    t1 = time.time()
    results = calling_api.individual_strengths(hands, board)
    t2 = time.time()
    print(results)
    print("Time taken for individual_strengths with board: ", t2-t1)



def test_pairwise_strengths():
    print("Testing pairwise_strengths Function")
    deck = [card for card in itertools.product('23456789TJQKA', 'shdc')]
    deck = ["".join(card) for card in deck]
    random.shuffle(deck)
    n = 4
    num_cards_per_hand = 5
    hands = [deck[i * num_cards_per_hand:(i + 1) * num_cards_per_hand] for i in range(n)]
    # get last 3 cards from the deck as board
    print([''.join(hand) for hand in hands])
    dead_cards = []
    t1 = time.time()
    # input type is list of strings:
    hands = [''.join(hand) for hand in hands]
    results = calling_api.pairwise_results(hands)
    t2 = time.time()

    print("Time taken for pairwise_strengths: ", t2-t1)
    print(results)

    # Now testing with a board
    board = ''.join(deck[-3:])
    print('Board: ',board)
    t1 = time.time()
    results = calling_api.pairwise_results(hands, board)
    t2 = time.time()
    print(results)
    print("Time taken for pairwise_strengths with board: ", t2-t1)

def test_preflop_results():
    print("Testing preflop_results Function")
    deck = [card for card in itertools.product('23456789TJQKA', 'shdc')]
    deck = ["".join(card) for card in deck]
    random.shuffle(deck)
    n = 4
    num_cards_per_hand = 5
    hands = [deck[i * num_cards_per_hand:(i + 1) * num_cards_per_hand] for i in range(n)]
    # get last 3 cards from the deck as board
    print([''.join(hand) for hand in hands])
    dead_cards = []
    t1 = time.time()
    # input type is list of strings:
    hands = [''.join(hand) for hand in hands]
    results = calling_api.preflop_results(hands)
    t2 = time.time()
    print(results)
    print("Time taken for preflop_results: ", t2-t1)

def test_preflop_multithread_plo6():
    print("Testing preflop_multithread_plo6 Function")
    deck = [card for card in itertools.product('23456789TJQKA', 'shdc')]
    deck = ["".join(card) for card in deck]
    random.shuffle(deck)
    n = 4
    num_cards_per_hand = 6
    hands = [deck[i * num_cards_per_hand:(i + 1) * num_cards_per_hand] for i in range(n)]
    # get last 3 cards from the deck as board
    print([''.join(hand) for hand in hands])
    dead_cards = []
    t1 = time.time()
    # input type is list of strings:
    hands = [''.join(hand) for hand in hands]
    results = calling_api.preflop_multithread_plo6(hands)
    t2 = time.time()
    print(results)
    print("Time taken for preflop_multithread_plo6: ", t2-t1)

def test_db_bompot():
    print('Testing db_bompot Function')
    deck = [card for card in itertools.product('23456789TJQKA', 'shdc')]
    deck = ["".join(card) for card in deck]
    random.shuffle(deck)
    n = 3
    num_cards_per_hand = 6
    hands = [deck[i * num_cards_per_hand:(i + 1) * num_cards_per_hand] for i in range(n)]
    board1 = ''.join(deck[-3:])
    board2 = ''.join(deck[-6:-3])
    num_iterations = 200
    num_opponents = 2
    t1 = time.time()
    hands = [''.join(hand) for hand in hands]
    results = calling_api.db_bompot(hands, board1, board2, num_scenarios=num_iterations, num_opponents=num_opponents)
    t2 = time.time()
    print(results)
    print("Time taken for db_bompot: ", t2-t1)
    return results


test_health_check()
while True:
    try:
        test_preflop_multithread_plo6()
        test_db_bompot()
        test_preflop_results()
    except Exception as e:
        time.sleep(4)


