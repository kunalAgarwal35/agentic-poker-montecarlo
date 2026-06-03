"""Convert category_dict.pkl -> category_array.npy (uint8 category index per 5-card hand)."""
import numpy as np
import pickle
import os

from card_encoding import hand_str_to_ints
from hand_indexing import hand_to_index, TOTAL_5CARD_HANDS
from hand_categories import NAME_TO_INDEX


def build(in_path='category_dict.pkl', out_path='category_array.npy') -> np.ndarray:
    # category_dict.pkl is a repo-internal precomputed artifact (not user input) — pickle is safe here.
    with open(in_path, 'rb') as f:
        category_dict = pickle.load(f)
    arr = np.zeros(TOTAL_5CARD_HANDS, dtype=np.uint8)
    for hand_str, name in category_dict.items():
        idx = hand_to_index(hand_str_to_ints(hand_str))
        arr[idx] = NAME_TO_INDEX[name]
    np.save(out_path, arr)
    print(f"Wrote {out_path}: {arr.shape}, {arr.nbytes} bytes")
    return arr


if __name__ == '__main__':
    build(os.path.join(os.path.dirname(__file__), 'category_dict.pkl'),
          os.path.join(os.path.dirname(__file__), 'category_array.npy'))
