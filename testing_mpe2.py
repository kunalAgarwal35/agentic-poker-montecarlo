import multithread_ploequities2 as mpl

# List of test cases, each case is a dictionary with 'board' and 'expected' category
test_cases = [
    {"board": "AhAsAc", "expected": "Trips Board"},
    {"board": "KhKsQd", "expected": "Paired Board"},
    {"board": "Th3h5h", "expected": "Unpaired Board - Monotone"},
    {"board": "Ah3h4d", "expected": "Unpaired Board - Two Tone (Straight)"},
    {"board": "Ah7h8d", "expected": "Unpaired Board - Two Tone (Non-Straight)"},
    {"board": "7h3d4s", "expected": "Unpaired Board - Rainbow (Straight)"},
    {"board": "2h7dJs", "expected": "Unpaired Board - Rainbow (Non-Straight)"},
    # Additional test cases can be added here without limitation
]
# Extending the test cases with the new boards provided
extended_test_cases = [
    {"board": "8c8d8s", "expected": "Trips Board"},
    {"board": "9s7dAc", "expected": "Unpaired Board - Rainbow (Non-Straight)"},
    {"board": "KsJd9s", "expected": "Unpaired Board - Two Tone (Straight)"},
    {"board": "5s6s9s", "expected": "Unpaired Board - Monotone"},
    {"board": "5c6d2h", "expected": "Unpaired Board - Rainbow (Straight)"},
    {"board": "AcKdTs", "expected": "Unpaired Board - Rainbow (Straight)"},
    {"board": "8s8cAs", "expected": "Unpaired Board - Two Tone (Non-Straight)"},
    {"board": "2c3dAs", "expected": "Unpaired Board - Rainbow (Straight)"},
    {"board": "9d9cKd", "expected": "Paired Board"},
    {"board": "4s5s9c", "expected": "Unpaired Board - Two Tone (Non-Straight)"},
]

# Combine the original and extended test cases for comprehensive testing
all_test_cases = test_cases + extended_test_cases

# Run the test function with all the test cases



def test_board_classification(test_cases):
    errors = []
    for case in test_cases:
        board, expected = case["board"], case["expected"]
        result = mpl.categorize_board(board)
        if result != expected:
            errors.append(f"Test failed for {board}, expected {expected}, got {result}")

    if errors:
        error_message = "\n".join(errors)
        raise AssertionError(error_message)
    else:
        print("All tests passed!")


# Run the test function with the list of test cases
test_board_classification(test_cases)