import numpy as np
from pql.parser.parse import parse
from pql.scenario import build_scenario
from pql.runtime.sampler import complete_boards


def _scn(board):
    q = parse(
        f"select avg(riverEquity(PLAYER_1)) from game='omahahi5', "
        f"board='{board}', PLAYER_1='AsAhKsKhQs', PLAYER_2='JdTd9d8d7d'"
    )
    return build_scenario(q)


def test_complete_boards_shape_and_determinism():
    s = _scn("2c3c4c")  # 3 board cards -> need 2 more
    b1 = complete_boards(s, num_trials=1000, seed=42)
    b2 = complete_boards(s, num_trials=1000, seed=42)
    assert b1.shape == (1000, 5)
    assert np.array_equal(b1, b2)              # seed -> deterministic
    # First 3 columns are the fixed board, repeated each trial
    assert np.array_equal(np.unique(b1[:, :3], axis=0), np.array([[3, 7, 11]]))


def test_completed_cards_are_legal():
    s = _scn("2c3c4c")
    used = set([3, 7, 11])                          # board
    used |= {48, 49, 44, 45, 40}                    # PLAYER_1
    used |= {CARD for CARD in __import__("card_encoding").hand_str_to_ints("JdTd9d8d7d").tolist()}
    b = complete_boards(s, num_trials=500, seed=1)
    drawn = b[:, 3:]
    # No drawn card is a used (dead-by-presence) card, and each trial's 5 are distinct
    assert not set(np.unique(drawn).tolist()) & used
    for row in b:
        assert len(set(row.tolist())) == 5


from pql.runtime.sampler import sample_trials


def test_sample_trials_fixed_hands_shapes_and_repeat():
    s = _scn('2c3c4c')  # existing helper: omahahi5, fixed AsAhKsKhQs vs JdTd9d8d7d
    hands, boards = sample_trials(s, num_trials=20, seed=5)
    assert hands.shape == (20, 2, 5)
    assert boards.shape == (20, 5)
    assert (hands[:, 0, :] == hands[0, 0, :]).all()        # fixed player repeats
    assert (boards[:, :3] == boards[0, :3]).all()          # fixed flop repeats
    for t in range(20):
        allcards = list(hands[t].flatten()) + list(boards[t])
        assert len(set(allcards)) == len(allcards)         # no collisions
