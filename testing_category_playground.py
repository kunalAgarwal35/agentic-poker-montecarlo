from category_playground import is_quads, is_full_house, is_nut_flush_or_flush, is_AA_KK, is_nut_straight_or_straight, is_nutflushdraw_or_flushdraw, is_wrap, is_top_two


def test_is_quads():
    test_cases = [
        # PLO5 examples
        (["Ah", "Ad", "Kh", "Kd", "Qh"], ["Ac", "As", "2d"], True),  # Quads of Aces
        (["Ah", "Ad", "Kh", "Kd", "Qh"], ["Ac", "9s", "2d", "3c", "4h"], False),  # No quads

        # PLO6 examples
        (["Ah", "Ad", "Kh", "Kd", "Qh", "Jh"], ["Ac", "As", "2d"], True),  # Quads of Aces
        (["Ah", "Ad", "Kh", "Kd", "Qh", "Jh"], ["Ac", "9s", "2d", "3c"], False),  # No quads

        # Testing with set on the board and hero holds the fourth one
        (["9h", "6d", "5c", "5d", "6s"], ["9c", "9s", "9d", "Kc", "Qd"], True),  # Quads of 9s, Hero holds fourth 9
        (["9h", "Td", "5c", "5d", "6s"], ["9c", "9s", "9d", "Kc", "Qd"], True),
        # Set on board, no quads, Hero doesn't hold fourth 9

        # False positives
        (["5h", "6h", "7h", "8h", "9h"], ["Th", "Jh", "Qh"], False),  # Straight flush, not quads

        # Additional test cases as requested
        (["Kd", "Qd", "Jd", "Td", "9d"], ["Kh", "Ks", "Kc"], True),  # Quads of Kings, Hero holds fourth King
        (["Ad", "Ac", "5s", "5h", "6h"], ["Ah", "As", "2d", "3c", "4h"], True),
        # Quads of Aces, Hero holds both remaining Aces
    ]

    for hand, board, expected in test_cases:
        result = is_quads(hand, board)
        assert result == expected, f"Test failed for hand {hand} and board {board}. Expected {expected}, got {result}."

    print("All tests for is_quads passed!")


def test_is_full_house():
    test_cases = [
        # Hand, Board, Expected result
        (["Ah", "Kh", "5s", "5h", "6s"], ["Ac", "Ad", "2d", "3c", "4h"], False),  # Full house, Aces over Fives
        (["Kh", "Kd", "5s", "5h", "6s"], ["9c", "9d", "9s", "3c"], True),  # Full house, Nines over Kings
        (["Ah", "Kh", "Qh", "Jh", "Th"], ["Kc", "Ks", "Qd", "Qs"], True),  # Full house, Kings over Queens
        (["4h", "4d", "5s", "5h", "6s"], ["4c", "7s", "7d"], True),  # Full house, Fours over Sevens
        (["8h", "8d", "Ts", "Jh", "Qs"], ["8c", "Tc", "Td"], True),  # Full house, Tens over Eights
        (["6h", "6d", "As", "Ah", "Ks"], ["Ac", "Ad", "Kd"], False),  # Quads Aces, not Full house
        (["2h", "2d", "3s", "3h", "4s"], ["5c", "5d", "5s"], True),  # Full house, Fives over Twos
        (["7h", "7d", "8s", "8h", "9s"], ["9c", "9d", "Ts"], False),  # Full house, Nines over Sevens
        (["Jh", "Jd", "Qs", "Qh", "Ks"], ["Kc", "Kd", "As"], False),  # Full house, Kings over Jacks
        (["3h", "3d", "4s", "4h", "5s"], ["5c", "6d", "7s"], False),  # No Full house
        (["9h", "Th", "Js", "Qh", "Ks"], ["Qc", "Qd", "Kd", "Kc"], True),  # Full house, Kings over Queens
        (["2h", "3d", "4s", "5h", "6s"], ["7c", "7d", "7s", "8c"], False),  # Full house, Sevens over Twos
        (["Ah", "Ad", "Kh", "Kd", "Qs"], ["Qc", "Qd", "Js"], False),  # Full house, Queens over Aces
        (["6h", "7d", "8s", "9h", "Ts"], ["9c", "9d", "Th", "Jc"], True),  # Full house, Nines over Tens
        (["4h", "5d", "6s", "7h", "8s"], ["7c", "7d", "8d", "8c"], True),  # Full house, Eights over Sevens
        (["Jh", "Jd", "Qs", "Qh", "As"], ["Ac", "Ad", "5d", "5c"], False),  # Quads Aces, not Full house
        (["2h", "2d", "2s", "3h", "3s"], ["3c", "4d", "5s"], False),  # Incorrect, can't use 3 cards from hand
        (["9h", "9d", "Ts", "Th", "Js"], ["Jc", "Jd", "Qs"], False),  # Full house, Jacks over Tens
        (["7h", "7d", "8s", "8h", "9s"], ["Tc", "Td", "Ts", "Jc"], True),  # Full house, Tens over Eights
        (["Ah", "Ad", "As", "Kh", "Kd"], ["Kc", "Qd", "Js"], False),  # Incorrect, can't use 3 cards from hand
    ]

    for hand, board, expected in test_cases:
        result = is_full_house(hand, board)
        assert result == expected, f"Test failed for hand {hand} and board {board}. Expected {expected}, got {result}."

    print("All tests for is_full_house passed!")


def test_is_nut_flush():
    test_cases = [
        # Format: (hand, board, expected_result)
        (["Ah", "Ks", "5s", "5h", "6s"], ["2h", "3h", "4h"], (True, True)),  # Nut flush with Ah on flop
        (["Ks", "Qs", "5s", "5h", "6s"], ["2s", "3s", "Js", "8d"], (False, True)),  # Nut flush with Ks on turn
        (["Ks", "Qs", "5s", "5h", "6s"], ["2s", "3s", "As", "8d", "9d"], (True, True)),  # As on board, no nut flush
        (["Ah", "Kh", "Qh", "Jh", "Th"], ["9h", "8h", "2d"], (False, False)),  # Nut flush with Ah, all hearts in hand
        (["Ac", "Kc", "5c", "4c", "3c"], ["2c", "9d", "Td", "Jd", "Qd"], (False, False)),  # Nut flush with Ac on river

        # More scenarios with different suits and stages of the game
        (["As", "3s", "4s", "5s", "6s"], ["7s", "8s", "9s", "Td"], (True, True)),
        (["Ah", "2h", "Th", "Jh", "Qh"], ["Kh", "5h", "6h"], (True, True)),  # Nut flush with Ah, Kh on board
        (["Ad", "Kd", "Qd", "Jd", "Td"], ["9d", "8d", "7h"], (False, False)),  # Straight flush in hand, not just nut flush
        (["Ah", "Kh", "4s", "5s", "6s"], ["Qh", "Jh", "6h", "3s", "4s"], (True, True)),  # Nut flush on turn with Ah
        # Add more test cases as needed to cover various scenarios
         (["7h", "Kh", "4s", "5s", "6s"], ["Qh", "Jh", "6h", "3s", "4s"], (False, True)),  # Nut flush on turn with Ah
         (["8h", "7h", "4s", "5s", "6s"], ["Ah", "Jh", "6h", "3s", "4s"], (False, True)),  # Nut flush on turn with Ah

    ]

    for hand, board, expected in test_cases:
        result = is_nut_flush_or_flush(hand, board)
        assert result == expected, f"Test failed for hand {hand} and board {board}. Expected {expected}, got {result}."

    print("All tests for is_nut_flush_or_flush passed!")

def test_is_AA_KK():
    test_cases = [
        # Hand, Board, Expected result
        (["Ah", "Kh", "5s", "5h", "6s"], ["Ac", "Ad", "2d", "3c", "4h"], False),  # Full house, Aces over Fives
        (["Kh", "Kd", "5s", "5h", "6s"], ["9c", "9d", "9s", "3c"], True),  # Full house, Nines over Kings
        (["Ah", "Kh", "Qh", "Jh", "Ah"], ["Kc", "Ks", "Qd", "Qs"], True),  # Full house, Kings over Queens
        (["4h", "4d", "5s", "6h", "6s"], ["4c", "7s", "7d"], False),  # Full house, Fours over Sevens
        (["8h", "8d", "Ts", "Jh", "Qs"], ["8c", "Tc", "Td"], False),  # Full house, Tens over Eights
        (["6h", "6d", "As", "Ah", "Ks"], ["Ac", "Ad", "Kd"], True)]
    for hand, board, expected in test_cases:
        result = is_AA_KK(hand, board)
        assert result == expected, f"Test failed for hand {hand} and board {board}. Expected {expected}, got {result}."
    print("All tests for is_AA_KK passed!")


def test_is_nut_straight_or_straight():
    test_cases = [
        # Format: (hand, board, (is_nut_straight, is_straight))
        (["Ah", "Kh", "5d", "5h", "6s"], ["Qh", "Jd", "Td"], (True, True)),  # Nut straight with Broadway
        (["4h", "3d", "5h", "6h", "7h"], ["5s", "6d", "7h", "9d"], (False, True)),
        # Straight, but not nut due to higher possibilities
        (["Ac", "2d", "3h", "4h", "5s"], ["3h", "4d", "5s"], (False, True)),  # Wheel straight (nut in this case)
        (["Ah", "3s", "4h", "5h", "6h"], ["4d", "5h", "6c"], (False, False)),  # No straight completed
        (["Ah", "Kd", "Qh", "Jh", "Th"], ["Qc", "Jh", "Th"], (True, True)),  # Nut straight with Broadway

        # Regular straight, not nut due to potential higher straights
        (["9h", "8d", "5d", "5h", "6s"], ["7h", "Jd", "Td"], (True, True)),  # Regular straight
        (["5h", "3d", "2s", "9h", "Ts"], ["4d", "5c", "6d"], (False, True)),  # Regular straight, low end
        (["7h", "8h", "Ts", "Jd", "Qc"], ["9d", "Td", "Jc"], (False, True)),  # Regular straight, missing high end

        # Not forming any straight
        (["2h", "3d", "5s", "9h", "Ts"], ["4d", "6c", "8d", "2d"], (False, True)),  # No straight possible
        (["Jh", "Qd", "Ks", "Ad", "2c"], ["3d", "4s", "5h"], (False, True)),  # Disjointed, no straight

        # Tests for edge cases
        (["As", "2d", "3h", "4c", "5s"], ["6h", "7c", "8s"], (False, True)),  # Regular straight (low), not nut
        (["As", "Ks", "Qs", "Js", "Ts"], ["9s", "8s", "7s"], (False, False)),  # Flush, but straight tested

        # Additional tests covering broader scenarios
        # (hand, board, expected outcome)
        # ...
    ]

    for hand, board, expected in test_cases:
        result = is_nut_straight_or_straight(hand, board)
        assert result == expected, f"Test failed for hand {hand} and board {board}. Expected {expected}, got {result}."

    print("All tests for is_nut_straight_or_straight passed!")

def test_is_nutflushdraw_or_flushdraw():
    test_cases = [
        # Format: (hand, board, (is_nut_flush_draw, is_flush_draw))
        (["Ah", "Kh", "5h", "4h", "2s"], ["Jh", "Th", "3s"], (True, True)),  # Nut flush draw with hearts
        (["Qs", "Js", "5h", "4h", "2s"], ["9s", "Ts", "3s", "4d"], (False, False)),  # Regular flush draw with spades
        (["Ah", "Kh", "Qh", "Jh", "Th"], ["2d", "3d", "4c", "5c", "6h"], (False, False)),  # No flush draw, already a flush
        (["2h", "3h", "Ad", "Kd", "Qd"], ["Jd", "Td", "5s", "6s", "7s"], (True, True)),  # Nut flush draw with diamonds
        # Add more test cases to cover various scenarios and board textures
    ]

    for hand, board, expected in test_cases:
        result = is_nutflushdraw_or_flushdraw(hand, board)
        assert result == expected, f"Test failed for hand {hand} and board {board}. Expected {expected}, got {result}."

    print("All tests for is_nutflushdraw_or_flushdraw passed!")


def test_is_wrap():
    test_cases = [
        # Format: (hand, board, expected_result)
        # Correct wrap scenarios
        (["9d", "Td", "Jd", "5h", "2h"], ["3c", "7c", "8c"], True),  # Wrap with 6, 9, T, J completing straights
        (["7c", "8d", "9h", "3h", "2h"], ["6c", "Td", "Ad"], True),  # Wrap with 5, 7, 8, 9, J, Q completing straights
        (["Kd", "Qd", "Jh", "3s", "2d"], ["Tc", "9d", "8s"], True),  # Wrap with 7, J, Q, K completing straights
        (["7h", "4d", "5s", "6h", "3s"], ["8c", "9d", "Ad"], True),  # Wrap with 6, 7, J completing straights

        # Non-wrap scenarios
        (["Ah", "Kh", "5d", "5h", "2s"], ["Qh", "Jd", "9d"], False),  # No wrap, straight already on board
        (["2h", "3d", "4s", "8h", "9s"], ["5c", "Qd", "7s"], False),  # No wrap, straight already on board
        (["2h", "3d", "6s", "8h", "9s"], ["5c", "Qd", "7s"], True),  # No wrap, straight already on board
        (["Ah", "Kh", "Qd", "9s", "3d"], ["4h", "5c", "6d"], False),  # No wrap, needs specific cards for straight
        (["Ah", "Kh", "7d", "8s", "4d"], ["2h", "5c", "6d"], True),  # No wrap, needs specific cards for straight

        # Edge cases and borderline wraps
        (["Ah", "3h", "4d", "5d", "2h"], ["2c", "7d", "8s"], False),  # No wrap, despite sequential hand cards
        (["9h", "Jh", "Qs", "Ks", "As"], ["Td", "8d", "6c"], True),  # Wrap with 9, J, Q, K
        (["9h", "Jh", "Qs", "Ks", "As"], ["Td", "Ad", "6c"], True),  # Wrap with 9, J, Q, K
        (["9h", "Jh", "Qs", "Ks", "As"], ["Td", "2d", "6c"], False),  # Wrap with 9, J, Q, K
        (["7h", "8h", "Ts", "Js", "Qs"], ["9d", "6c", "4s"], True),  # Wrap with 7, 8, T, J

        # More test cases to ensure comprehensive coverage
        # ...
    ]

    for hand, board, expected in test_cases:
        result = is_wrap(hand, board)
        assert result == expected, f"Test failed for hand {hand} and board {board}. Expected {expected}, got {result}."

    print("All tests for is_wrap passed!")


def test_is_top_two():
    test_cases = [
        # Format: (hand, board, expected_result)
        (["Ah", "Kh", "5d", "5h", "2s"], ["Ah", "Kd", "3c", "4d", "5c"], True),  # Hand has top two pairs A and K
        (["Qh", "Jd", "5d", "5h", "2s"], ["Ah", "Kd", "Qc", "Jc", "Td"], False),  # Hand has top two pairs A and K, Q and J are not top
        (["Ah", "Qh", "5d", "5h", "2s"], ["Ad", "Kc", "3c"], False),  # Hand does not have exactly one of each top rank
        (["Ah", "Kc", "Qh", "Jh", "7h"], ["Qc", "Jd", "7s"], True),  # Hand has Q and J, but they are not top two on board
        (["2h", "3d", "4s", "5h", "6s"], ["2c", "3c", "Ac"], False),  # No pair of top two ranks in hand
        # Add more test cases to cover various scenarios
    ]

    for hand, board, expected in test_cases:
        result = is_top_two(hand, board)
        assert result == expected, f"Test failed for hand {hand} and board {board}. Expected {expected}, got {result}."

    print("All tests for is_top_two passed!")

# Run the test function
test_is_top_two()


# Run the test function
test_is_wrap()

# Run the test function
test_is_nutflushdraw_or_flushdraw()



# Run the test function
test_is_nut_straight_or_straight()

# Run the test function
test_is_quads()

test_is_full_house()

test_is_nut_flush()

test_is_AA_KK()