"""Hand-category constants shared by the category array and PQL exactHandType."""

# Index order = poker strength, low -> high. Names match category_dict.pkl values.
CATEGORY_NAMES = [
    'High Card', 'One Pair', 'Two Pair', 'Three of a Kind', 'Straight',
    'Flush', 'Full House', 'Four of a Kind', 'Straight Flush',
]
# PQL tokens (lowercase, no spaces) accepted by exactHandType and used in aliases.
CATEGORY_TOKENS = [
    'highcard', 'pair', 'twopair', 'trips', 'straight',
    'flush', 'fullhouse', 'quads', 'straightflush',
]

NAME_TO_INDEX = {name: i for i, name in enumerate(CATEGORY_NAMES)}
TOKEN_TO_INDEX = {tok: i for i, tok in enumerate(CATEGORY_TOKENS)}
