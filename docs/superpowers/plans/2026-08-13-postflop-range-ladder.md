# Postflop Range Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute, for a hero on a given board, its equity against the top
5/15/25/40/60/100% of villain hands ranked by strength *on that board*, plus the
weakest hand in each slice.

**Architecture:** One evaluation pass. Sample N villain hands and R runouts, then
evaluate every villain and every hero on every runout — `O((N + heroes) × R)`,
never pairwise. Villain strength, the percentile buckets, hero equity per bucket
and the boundary hands all fall out of that single score table. The villain
population depends only on board + dead cards, so it is shared across all heroes
at the table.

**Tech Stack:** Python 3, numpy, Flask (`server.py`), pytest. Reuses the repo's
existing vectorised evaluator — no new evaluator.

**Spec:** `C:\PycharmProjects\postfloper\docs\superpowers\specs\2026-08-13-postflop-range-ladder-design.md`

## Global Constraints

- Buckets are percentages of the villain population: `[5, 15, 25, 40, 60, 100]`.
- Ranking is **hero-independent** — a villain's strength is its equity against the
  sampled population, never against hero. Ranking by equity-vs-hero is degenerate
  (see spec §6) and must not be implemented.
- Hero equity is **heads-up** against one hand drawn from the slice, never multiway.
- Default `N = 10000` villain hands.
- Latency budget **2–3 s** per board. `R` is chosen by measurement in Task 8, not guessed.
- Accuracy target: bucket equities within **±1 percentage point** run-to-run across seeds.
- River (`board_len == 5`) is **exact** — one runout of zero cards, no sampling.
- Card strings are 2-char, rank-then-suit, e.g. `"6s"`, hands concatenated `"Ts7d6h5c"`.
- All new code lives in `range_ladder.py` at the repo root, matching the flat layout
  of `hand_rank_evaluator.py` / `card_encoding.py`.

**Existing primitives this plan consumes (do not reimplement):**

```python
from card_encoding import hand_str_to_ints, generate_deck_ints, ints_to_hand_str
# hand_str_to_ints("Ts7d6h5c") -> np.ndarray (K,) int32
# generate_deck_ints(exclude)  -> np.ndarray of remaining card ints, int32
# ints_to_hand_str(arr)        -> "Ts7d6h5c"

from hand_indexing import BINOMIAL                    # (53, 7) int64
from hand_rank_evaluator import (
    _batch_best_score,   # (hands (N,K), boards (N,5), hand_combos, board_combos,
                         #  score_array, binomial) -> (N,) float64 ; higher = better
    get_score_array,     # () -> np.ndarray
    detect_game_type,    # (num_cards) -> 'plo4' | 'plo5' | 'plo6'
    _HAND_COMBOS,        # {'plo4': (C,2), 'plo5': ..., 'plo6': ...}
    _BOARD_COMBOS,       # (10, 3) — always 3-from-5
)
```

---

### Task 1: Villain population sampling

**Files:**
- Create: `range_ladder.py`
- Test: `tests/test_range_ladder.py`

**Interfaces:**
- Consumes: `card_encoding.generate_deck_ints`
- Produces: `sample_villains(deck: np.ndarray, num_cards: int, n: int, rng: np.random.Generator) -> np.ndarray` returning `(M, num_cards) int32`, `M <= n`, every row sorted ascending, no duplicate rows, no card repeated within a row.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from range_ladder import sample_villains

def test_sample_villains_returns_distinct_legal_hands():
    deck = np.arange(20, 52, dtype=np.int32)      # 32 cards available
    rng = np.random.default_rng(7)
    out = sample_villains(deck, num_cards=6, n=500, rng=rng)

    assert out.shape == (500, 6)
    assert out.dtype == np.int32
    # every card comes from the deck
    assert np.isin(out, deck).all()
    # no card repeats inside a hand
    assert all(len(set(row.tolist())) == 6 for row in out)
    # rows are sorted, so identical hands are byte-identical
    assert (np.diff(out, axis=1) > 0).all()
    # no duplicate hands
    assert len({tuple(r) for r in out.tolist()}) == 500

def test_sample_villains_is_seed_deterministic():
    deck = np.arange(20, 52, dtype=np.int32)
    a = sample_villains(deck, 6, 100, np.random.default_rng(1))
    b = sample_villains(deck, 6, 100, np.random.default_rng(1))
    assert np.array_equal(a, b)

def test_sample_villains_caps_at_the_available_space():
    deck = np.arange(0, 7, dtype=np.int32)        # C(7,6) = 7 possible hands
    out = sample_villains(deck, 6, 500, np.random.default_rng(3))
    assert out.shape[0] == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_range_ladder.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'range_ladder'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Postflop range ladder: hero equity vs board-strength percentile slices.

See docs/superpowers/plans/2026-08-13-postflop-range-ladder.md
"""
from itertools import combinations

import numpy as np

from card_encoding import generate_deck_ints, hand_str_to_ints, ints_to_hand_str
from hand_indexing import BINOMIAL
from hand_rank_evaluator import (
    _BOARD_COMBOS,
    _HAND_COMBOS,
    _batch_best_score,
    detect_game_type,
    get_score_array,
)


def sample_villains(deck, num_cards, n, rng):
    """Distinct legal villain hands drawn from `deck`, as sorted rows.

    Falls back to exhaustive enumeration when the space is smaller than `n`,
    so tiny decks return every hand exactly once instead of looping forever.
    """
    deck = np.asarray(deck, dtype=np.int32)
    total = len(deck)
    if total < num_cards:
        return np.empty((0, num_cards), dtype=np.int32)

    # Exhaustive when the space is small enough to enumerate cheaply.
    space = 1
    for i in range(num_cards):
        space = space * (total - i) // (i + 1)
    if space <= max(n, 1):
        rows = [sorted(c) for c in combinations(deck.tolist(), num_cards)]
        return np.array(rows, dtype=np.int32)

    seen = set()
    rows = []
    while len(rows) < n:
        need = n - len(rows)
        draw = rng.random((need * 2, total)).argsort(axis=1)[:, :num_cards]
        cand = np.sort(deck[draw], axis=1)
        for row in cand:
            key = tuple(row.tolist())
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            if len(rows) == n:
                break
    return np.array(rows, dtype=np.int32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_range_ladder.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add range_ladder.py tests/test_range_ladder.py
git commit -m "feat(ladder): distinct villain-hand sampling"
```

---

### Task 2: Runout sampling and villain eligibility

**Files:**
- Modify: `range_ladder.py`
- Test: `tests/test_range_ladder.py`

**Interfaces:**
- Produces:
  - `sample_runouts(deck, board_len, r, rng) -> np.ndarray` of shape `(R, 5 - board_len) int32`. River (`board_len == 5`) returns shape `(1, 0)` — one empty runout, which makes the river the exact case with no special-casing downstream.
  - `eligible_mask(villains, runout) -> np.ndarray` of shape `(N,) bool`, False where the villain holds a runout card.

- [ ] **Step 1: Write the failing test**

```python
from range_ladder import sample_runouts, eligible_mask

def test_sample_runouts_draws_the_missing_board_cards():
    deck = np.arange(10, 52, dtype=np.int32)
    out = sample_runouts(deck, board_len=3, r=200, rng=np.random.default_rng(5))
    assert out.shape == (200, 2)
    assert np.isin(out, deck).all()
    assert (out[:, 0] != out[:, 1]).all()

def test_river_yields_one_empty_runout():
    deck = np.arange(10, 52, dtype=np.int32)
    out = sample_runouts(deck, board_len=5, r=200, rng=np.random.default_rng(5))
    assert out.shape == (1, 0)

def test_eligible_mask_excludes_villains_holding_a_runout_card():
    villains = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    runout = np.array([4, 9], dtype=np.int32)     # card 4 sits in villain 0
    mask = eligible_mask(villains, runout)
    assert mask.tolist() == [False, True]

def test_empty_runout_leaves_every_villain_eligible():
    villains = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    mask = eligible_mask(villains, np.array([], dtype=np.int32))
    assert mask.tolist() == [True, True]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_range_ladder.py -q`
Expected: FAIL — `ImportError: cannot import name 'sample_runouts'`

- [ ] **Step 3: Write minimal implementation**

```python
def sample_runouts(deck, board_len, r, rng):
    """`r` random completions of the board. River -> a single empty runout."""
    deck = np.asarray(deck, dtype=np.int32)
    need = 5 - board_len
    if need <= 0:
        return np.empty((1, 0), dtype=np.int32)
    draw = rng.random((r, len(deck))).argsort(axis=1)[:, :need]
    return deck[draw].astype(np.int32)


def eligible_mask(villains, runout):
    """False for villains holding a card that the runout also uses.

    Those pairings are impossible and must never be scored. Skipping them
    evaluates each villain over exactly the runouts compatible with it, which
    is the correct conditional distribution -- unbiased. See spec 4.3.
    """
    if runout.size == 0:
        return np.ones(villains.shape[0], dtype=bool)
    return ~np.isin(villains, runout).any(axis=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_range_ladder.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add range_ladder.py tests/test_range_ladder.py
git commit -m "feat(ladder): runout sampling and villain/runout collision mask"
```

---

### Task 3: Scoring one runout

**Files:**
- Modify: `range_ladder.py`
- Test: `tests/test_range_ladder.py`

**Interfaces:**
- Produces: `score_hands(hands, board5, game, score_array, chunk=2000) -> np.ndarray` of shape `(N,) float64`, higher is better. Chunks the call to `_batch_best_score` because its intermediate is `(N, C_h * C_b, 5)` int32 — at N=10000, PLO6 that is ~30 MB per temporary, and several exist at once.

- [ ] **Step 1: Write the failing test**

```python
from card_encoding import hand_str_to_ints
from hand_rank_evaluator import get_score_array
from range_ladder import score_hands

def test_score_hands_ranks_a_flush_above_a_pair():
    board5 = hand_str_to_ints("6s7s4s2h9d")
    hands = np.stack([
        hand_str_to_ints("AsKs9h2c"),   # nut flush
        hand_str_to_ints("6h6d3c2d"),   # trips/pair
    ])
    out = score_hands(hands, board5, "plo4", get_score_array())
    assert out.shape == (2,)
    assert out[0] > out[1]

def test_score_hands_chunking_matches_unchunked():
    rng = np.random.default_rng(11)
    deck = generate_deck_ints(["6s", "7s", "4s", "2h", "9d"])
    hands = sample_villains(deck, 4, 300, rng)
    board5 = hand_str_to_ints("6s7s4s2h9d")
    sa = get_score_array()
    a = score_hands(hands, board5, "plo4", sa, chunk=10_000)
    b = score_hands(hands, board5, "plo4", sa, chunk=7)
    assert np.array_equal(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_range_ladder.py -q`
Expected: FAIL — `ImportError: cannot import name 'score_hands'`

- [ ] **Step 3: Write minimal implementation**

```python
def score_hands(hands, board5, game, score_array, chunk=2000):
    """Best 5-card score for each hand on one complete board. Higher = better."""
    hand_combos = _HAND_COMBOS[game]
    n = hands.shape[0]
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk):
        block = hands[start:start + chunk]
        boards = np.broadcast_to(board5, (block.shape[0], 5))
        out[start:start + chunk] = _batch_best_score(
            block, np.ascontiguousarray(boards),
            hand_combos, _BOARD_COMBOS, score_array, BINOMIAL,
        )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_range_ladder.py -q`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add range_ladder.py tests/test_range_ladder.py
git commit -m "feat(ladder): chunked per-runout hand scoring"
```

---

### Task 4: The evaluation pass — villain strength and hero win counts

**Files:**
- Modify: `range_ladder.py`
- Test: `tests/test_range_ladder.py`

**Interfaces:**
- Produces: `evaluate_population(villains, heroes, board_ints, runouts, game, score_array) -> tuple[np.ndarray, np.ndarray]`
  - `strength`: `(N,) float64` — each villain's mean share of the eligible field it beats (win + ½ tie), i.e. its equity against the population. NaN-free: villains eligible on zero runouts get `0.0`.
  - `hero_equity`: `(H, N) float64` — hero `h`'s equity against villain `i`, averaged over the runouts where `i` was eligible.

  This is the single `O((N + H) × R)` pass; everything later is bookkeeping on
  these two arrays.

- [ ] **Step 1: Write the failing test**

```python
from range_ladder import evaluate_population

def test_strength_orders_a_made_flush_above_air_on_the_river():
    board = hand_str_to_ints("6s7s4s2h9d")          # river: exact, no sampling
    villains = np.stack([
        hand_str_to_ints("AsKs9h2c"),               # nut flush
        hand_str_to_ints("Jd3d2d4d"),               # air
        hand_str_to_ints("6h6d3c5d"),               # pair
    ])
    heroes = np.stack([hand_str_to_ints("QsJs8c3h")])   # queen-high flush
    runouts = np.empty((1, 0), dtype=np.int32)
    strength, hero_equity = evaluate_population(
        villains, heroes, board, runouts, "plo4", get_score_array()
    )
    assert strength[0] > strength[2] > strength[1]
    assert hero_equity.shape == (1, 3)
    assert hero_equity[0, 0] == 0.0      # loses to the nut flush
    assert hero_equity[0, 1] == 1.0      # beats air

def test_a_villain_holding_a_runout_card_is_skipped_not_scored():
    # Board 6s7s4s; the only runout is 2h9d. Villain 0 holds 2h, so the pairing
    # is impossible -- it must be skipped, not counted as a loss.
    board = hand_str_to_ints("6s7s4s")
    villains = np.stack([
        hand_str_to_ints("As2hKd3c"),
        hand_str_to_ints("AhKh9c8c"),
    ])
    heroes = np.stack([hand_str_to_ints("QsJs8d3d")])
    runouts = np.stack([hand_str_to_ints("2h9d")])
    strength, hero_equity = evaluate_population(
        villains, heroes, board, runouts, "plo4", get_score_array()
    )
    assert strength[0] == 0.0            # never eligible -> no data
    assert hero_equity[0, 0] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_range_ladder.py -q`
Expected: FAIL — `ImportError: cannot import name 'evaluate_population'`

- [ ] **Step 3: Write minimal implementation**

```python
def evaluate_population(villains, heroes, board_ints, runouts, game, score_array):
    """The single O((N + H) x R) pass. Returns (strength, hero_equity)."""
    n = villains.shape[0]
    h = heroes.shape[0]

    beat_sum = np.zeros(n, dtype=np.float64)     # share of field beaten, summed
    beat_cnt = np.zeros(n, dtype=np.float64)     # runouts this villain was eligible for
    hero_sum = np.zeros((h, n), dtype=np.float64)

    for runout in runouts:
        board5 = np.concatenate([board_ints, runout]).astype(np.int32)
        mask = eligible_mask(villains, runout)
        idx = np.flatnonzero(mask)
        if idx.size < 2:
            continue

        vs = score_hands(villains[idx], board5, game, score_array)
        hs = score_hands(heroes, board5, game, score_array)

        # Share of the eligible field each villain beats: win + 1/2 tie.
        order = np.sort(vs)
        lower = np.searchsorted(order, vs, side="left")          # strictly worse
        upper = np.searchsorted(order, vs, side="right")
        ties = upper - lower - 1                                  # excluding self
        share = (lower + 0.5 * ties) / (idx.size - 1)
        beat_sum[idx] += share
        beat_cnt[idx] += 1.0

        # Hero vs each eligible villain, heads-up.
        for j in range(h):
            hero_sum[j, idx] += np.where(hs[j] > vs, 1.0, np.where(hs[j] == vs, 0.5, 0.0))

    seen = beat_cnt > 0
    strength = np.zeros(n, dtype=np.float64)
    strength[seen] = beat_sum[seen] / beat_cnt[seen]
    hero_equity = np.zeros((h, n), dtype=np.float64)
    hero_equity[:, seen] = hero_sum[:, seen] / beat_cnt[seen]
    return strength, hero_equity
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_range_ladder.py -q`
Expected: PASS (11 passed)

- [ ] **Step 5: Commit**

```bash
git add range_ladder.py tests/test_range_ladder.py
git commit -m "feat(ladder): single-pass villain strength and hero equity"
```

---

### Task 5: Buckets, per-bucket equity, and boundary hands

**Files:**
- Modify: `range_ladder.py`
- Test: `tests/test_range_ladder.py`

**Interfaces:**
- Produces: `build_rungs(strength, hero_equity_row, villains, buckets, board5_for_category) -> list[dict]` — one dict per bucket, `{"bucket": int, "equity": float, "edge": {"cards": str, "category": str}}`. Villains eligible on zero runouts (`strength == 0` and never seen) are excluded upstream by Task 6, so this function assumes every row carries data.

  `category` naming is deferred to Task 6's helper `describe_category`.

- [ ] **Step 1: Write the failing test**

```python
from range_ladder import build_rungs

def test_rungs_slice_by_strength_and_report_the_weakest_hand_in_each():
    # 10 villains with strengths 0.0 .. 0.9; hero beats exactly the weak half.
    villains = np.stack([hand_str_to_ints("AsKs9h2c")] * 10)
    strength = np.linspace(0.0, 0.9, 10)
    hero = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.float64)

    rungs = build_rungs(strength, hero, villains, [20, 50, 100],
                        hand_str_to_ints("6s7s4s2h9d"))

    assert [r["bucket"] for r in rungs] == [20, 50, 100]
    # top 20% = the 2 strongest, which hero loses to
    assert rungs[0]["equity"] == 0.0
    # top 100% = everyone; hero beats 5 of 10
    assert rungs[2]["equity"] == 0.5
    # equity never decreases as the bucket widens
    eq = [r["equity"] for r in rungs]
    assert eq == sorted(eq)

def test_every_bucket_holds_at_least_one_hand():
    villains = np.stack([hand_str_to_ints("AsKs9h2c")] * 3)
    strength = np.array([0.1, 0.5, 0.9])
    hero = np.array([1.0, 1.0, 0.0])
    rungs = build_rungs(strength, hero, villains, [5, 100],
                        hand_str_to_ints("6s7s4s2h9d"))
    assert rungs[0]["equity"] == 0.0        # top 5% rounds up to 1 hand
    assert rungs[0]["edge"]["cards"] != ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_range_ladder.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_rungs'`

- [ ] **Step 3: Write minimal implementation**

```python
def build_rungs(strength, hero_equity_row, villains, buckets, board5_for_category):
    """One rung per bucket: hero equity vs that slice + the slice's weakest hand."""
    order = np.argsort(-strength, kind="stable")     # strongest first
    n = order.size
    rungs = []
    for pct in buckets:
        take = max(1, int(round(n * pct / 100.0)))
        sl = order[:take]
        edge_idx = int(sl[-1])                        # weakest hand in the slice
        rungs.append({
            "bucket": pct,
            "equity": float(hero_equity_row[sl].mean()),
            "edge": {
                "cards": ints_to_hand_str(villains[edge_idx]),
                "category": describe_category(villains[edge_idx], board5_for_category),
            },
        })
    return rungs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_range_ladder.py -q`
Expected: FAIL — `NameError: name 'describe_category' is not defined`. Add the
stub below, then re-run and expect PASS (13 passed).

```python
def describe_category(hand_ints, board5):
    """Human label for a hand's made category. Filled in by Task 6."""
    return ""
```

- [ ] **Step 5: Commit**

```bash
git add range_ladder.py tests/test_range_ladder.py
git commit -m "feat(ladder): percentile buckets, per-bucket equity, boundary hands"
```

---

### Task 6: Category naming and the public entry point

**Files:**
- Modify: `range_ladder.py`
- Test: `tests/test_range_ladder.py`

**Interfaces:**
- Produces:
  - `describe_category(hand_ints, board5) -> str` — one of `"high card"`, `"pair"`, `"two pair"`, `"trips"`, `"straight"`, `"flush"`, `"full house"`, `"quads"`, `"straight flush"`. Derived from the existing `hand_categories` module so naming matches the rest of the engine.
  - `compute_range_ladder(board: str, dead: list[str], heroes: list[dict], buckets=(5,15,25,40,60,100), hands=10000, runouts=None, seed=None) -> dict` — the whole feature, returning the spec §5 response body.

- [ ] **Step 1: Write the failing test**

```python
from range_ladder import compute_range_ladder

def test_river_ladder_is_exact_and_monotonic():
    out = compute_range_ladder(
        board="6s7s4s2h9d",
        dead=["AsKs9h2c", "QsJs8c3h"],
        heroes=[{"id": "nuts", "cards": "AsKs9h2c"},
                {"id": "air",  "cards": "QsJs8c3h"}],
        hands=2000, seed=42,
    )
    assert out["exact"] is True
    assert out["runouts"] == 0
    assert out["population"] > 0

    nuts = next(l for l in out["ladders"] if l["id"] == "nuts")["rungs"]
    air = next(l for l in out["ladders"] if l["id"] == "air")["rungs"]

    # monotonic: equity never falls as the bucket widens
    for rungs in (nuts, air):
        eq = [r["equity"] for r in rungs]
        assert eq == sorted(eq), eq

    # the whole point: nuts and air must look different vs the top 5%
    assert nuts[0]["equity"] > 0.85
    assert air[0]["equity"] < 0.15

    # boundary hands are shared -- ranking is hero-independent
    assert [r["edge"]["cards"] for r in nuts] == [r["edge"]["cards"] for r in air]

def test_hero_missing_from_dead_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        compute_range_ladder(
            board="6s7s4s",
            dead=["AsKs9h2c"],
            heroes=[{"id": "x", "cards": "QsJs8c3h"}],   # not in dead
            hands=100, seed=1,
        )

def test_describe_category_names_a_flush():
    assert describe_category(hand_str_to_ints("AsKs9h2c"),
                             hand_str_to_ints("6s7s4s2h9d")) == "flush"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_range_ladder.py -q`
Expected: FAIL — `ImportError: cannot import name 'compute_range_ladder'`

- [ ] **Step 3: Write minimal implementation**

First inspect the category helper so the labels match the engine's own naming:

```bash
grep -n "^def \|^CATEGORY\|^[A-Z_]* = " hand_categories.py | head -20
```

Then implement, mapping whatever `hand_categories` exposes onto the nine labels
above:

```python
DEFAULT_BUCKETS = (5, 15, 25, 40, 60, 100)


def compute_range_ladder(board, dead, heroes, buckets=DEFAULT_BUCKETS,
                         hands=10000, runouts=None, seed=None):
    board_ints = hand_str_to_ints(board)
    board_len = len(board_ints)
    if board_len not in (3, 4, 5):
        raise ValueError(f"board must be 3, 4 or 5 cards, got {board_len}")

    dead_cards = []
    for d in dead:
        dead_cards.extend(ints_to_hand_str(hand_str_to_ints(d))[i:i + 2]
                          for i in range(0, len(d), 2))
    dead_set = set(dead_cards)

    for hero in heroes:
        cards = [hero["cards"][i:i + 2] for i in range(0, len(hero["cards"]), 2)]
        missing = [c for c in cards if c not in dead_set]
        if missing:
            raise ValueError(
                f"hero {hero['id']} cards {missing} absent from `dead`; "
                "villains would be allowed to hold them"
            )

    hero_arrays = np.stack([hand_str_to_ints(h["cards"]) for h in heroes])
    num_cards = hero_arrays.shape[1]
    game = detect_game_type(num_cards)
    score_array = get_score_array()

    deck = generate_deck_ints(list(dead_set) + [board[i:i + 2]
                                                for i in range(0, len(board), 2)])
    rng = np.random.default_rng(seed)

    villains = sample_villains(deck, num_cards, hands, rng)
    if villains.shape[0] == 0:
        raise ValueError("no legal villain hands remain")

    r = 1 if board_len == 5 else (runouts if runouts else 800)
    runout_rows = sample_runouts(deck, board_len, r, rng)

    strength, hero_equity = evaluate_population(
        villains, hero_arrays, board_ints, runout_rows, game, score_array
    )

    # A villain eligible on zero runouts carries no data; drop it rather than
    # letting a 0.0 masquerade as "weakest hand in the population".
    seen = np.zeros(villains.shape[0], dtype=bool)
    for runout in runout_rows:
        seen |= eligible_mask(villains, runout)
    villains, strength, hero_equity = villains[seen], strength[seen], hero_equity[:, seen]

    ladders = []
    for j, hero in enumerate(heroes):
        # Category labels use the median runout so a flop board still names a
        # complete 5-card hand; on the river this is the real board.
        board5 = np.concatenate([board_ints, runout_rows[0]]).astype(np.int32)
        ladders.append({
            "id": hero["id"],
            "rungs": build_rungs(strength, hero_equity[j], villains, list(buckets), board5),
        })

    return {
        "population": int(villains.shape[0]),
        "runouts": 0 if board_len == 5 else int(runout_rows.shape[0]),
        "exact": board_len == 5,
        "ladders": ladders,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_range_ladder.py -q`
Expected: PASS (16 passed)

- [ ] **Step 5: Commit**

```bash
git add range_ladder.py tests/test_range_ladder.py
git commit -m "feat(ladder): category labels and compute_range_ladder entry point"
```

---

### Task 7: Correctness properties

**Files:**
- Create: `tests/test_range_ladder_properties.py`

**Interfaces:**
- Consumes: `compute_range_ladder`, `evaluate_population` from Task 6/4.

These are the assertions that catch a silently-wrong ladder. Each targets a
distinct failure: inverted ranking, mis-sliced buckets, and a broken equity
denominator.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest

from card_encoding import hand_str_to_ints
from range_ladder import compute_range_ladder

FLOP = "6s7s4s"
DEAD = ["AsKs9h2c", "QhJhTd3d"]
HEROES = [{"id": "nuts", "cards": "AsKs9h2c"}, {"id": "air", "cards": "QhJhTd3d"}]


def _ladder(**kw):
    return compute_range_ladder(board=FLOP, dead=DEAD, heroes=HEROES,
                                hands=3000, runouts=300, seed=kw.pop("seed", 5), **kw)


def test_equity_is_monotonic_in_bucket_width():
    """A dip means the ranking is inverted or the slices are wrong."""
    out = _ladder()
    for lad in out["ladders"]:
        eq = [r["equity"] for r in lad["rungs"]]
        assert eq == sorted(eq), f"{lad['id']}: {eq}"


def test_top_100_percent_equals_equity_versus_a_random_hand():
    """Independent check of the equity denominator."""
    out = _ladder()
    for lad in out["ladders"]:
        rung100 = [r for r in lad["rungs"] if r["bucket"] == 100][0]
        # recompute directly: mean hero equity over the whole population
        assert 0.0 <= rung100["equity"] <= 1.0
    nuts = [l for l in out["ladders"] if l["id"] == "nuts"][0]["rungs"]
    air = [l for l in out["ladders"] if l["id"] == "air"][0]["rungs"]
    top100 = {r["bucket"]: r["equity"] for r in nuts}[100]
    assert top100 > {r["bucket"]: r["equity"] for r in air}[100]


def test_boundary_strength_falls_as_buckets_widen():
    out = _ladder()
    rungs = out["ladders"][0]["rungs"]
    assert len({r["edge"]["cards"] for r in rungs}) > 1


def test_ranking_is_hero_independent():
    """Both heroes must see identical boundary hands -- the whole design rests
    on the ranking not depending on who is asking."""
    out = _ladder()
    edges = [[r["edge"]["cards"] for r in lad["rungs"]] for lad in out["ladders"]]
    assert edges[0] == edges[1]


def test_nuts_and_air_are_distinguishable_against_the_tightest_slice():
    """If this fails the feature is pointless -- see spec section 6."""
    out = _ladder()
    by_id = {l["id"]: {r["bucket"]: r["equity"] for r in l["rungs"]} for l in out["ladders"]}
    assert by_id["nuts"][5] - by_id["air"][5] > 0.4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_range_ladder_properties.py -q`
Expected: All five run. Any failure here is a real defect in Tasks 1–6 — fix the
implementation, never the assertion.

- [ ] **Step 3: Fix whatever the properties expose**

No new code is written for this task if Tasks 1–6 are correct. If a property
fails, the likeliest causes in order: the `eligible_mask` denominator in
`evaluate_population` (dividing by `beat_cnt` of the wrong villain), the sort
direction in `build_rungs` (`-strength` sorts strongest-first), and `take`
rounding to 0 for a small population.

- [ ] **Step 4: Run the whole suite**

Run: `python -m pytest tests/ -q`
Expected: PASS, with no regression in the pre-existing tests.

- [ ] **Step 5: Commit**

```bash
git add tests/test_range_ladder_properties.py
git commit -m "test(ladder): monotonicity, hero-independence and discrimination properties"
```

---

### Task 8: Choose R by measurement

**Files:**
- Create: `bench_range_ladder.py`

**Interfaces:**
- Produces: a printed table of `(game, runouts) -> wall seconds, equity spread across seeds`, and a chosen `DEFAULT_RUNOUTS` constant written into `range_ladder.py`.

The spec deliberately does not fix R. Measure, then decide.

- [ ] **Step 1: Write the benchmark**

```python
"""Pick the runout count: largest R inside the 2-3s budget that holds +/-1pp."""
import time

import numpy as np

from range_ladder import compute_range_ladder

CASES = {
    "plo4": (["AsKs9h2c", "QhJhTd3d"], [{"id": "h", "cards": "AsKs9h2c"}]),
    "plo6": (["AsKs9h2c3d4d", "QhJhTd3s5s8s"], [{"id": "h", "cards": "AsKs9h2c3d4d"}]),
}

for game, (dead, heroes) in CASES.items():
    for r in (100, 200, 400, 800, 1600):
        eqs = []
        t0 = time.perf_counter()
        for seed in (1, 2, 3):
            out = compute_range_ladder(board="6s7s4s", dead=dead, heroes=heroes,
                                       hands=10000, runouts=r, seed=seed)
            eqs.append([x["equity"] for x in out["ladders"][0]["rungs"]])
        elapsed = (time.perf_counter() - t0) / 3
        spread = float(np.ptp(np.array(eqs), axis=0).max())
        print(f"{game:5s} R={r:5d}  {elapsed:6.2f}s/board  max spread {spread*100:5.2f}pp")
```

- [ ] **Step 2: Run it**

Run: `python bench_range_ladder.py`
Expected: a table. PLO6 is the slow case — 15 hand combos x 10 board combos per
hand versus PLO4's 6 x 10.

- [ ] **Step 3: Set the constant**

Pick the largest R whose PLO6 time stays under 2.5 s and whose spread is ≤ 1.0 pp,
then in `range_ladder.py` replace the literal `800` in `compute_range_ladder` with:

```python
DEFAULT_RUNOUTS = <measured value>   # Task 8: <X>s/board PLO6, +/-<Y>pp across seeds
```

If no R satisfies both, the budget wins: take the largest R inside 2.5 s, record
the true spread in the comment, and round displayed equities to whole percent so
the number is not stated more precisely than it is known.

- [ ] **Step 4: Re-run the property tests at the chosen R**

Run: `python -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bench_range_ladder.py range_ladder.py
git commit -m "perf(ladder): measure runout count against the 2-3s budget"
```

---

### Task 9: Flask endpoint

**Files:**
- Modify: `server.py`
- Test: `tests/test_range_ladder_endpoint.py`

**Interfaces:**
- Consumes: `compute_range_ladder` from Task 6.
- Produces: `POST /range_ladder`, request/response bodies exactly as spec §5.

- [ ] **Step 1: Write the failing test**

```python
import json

import pytest

from server import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_range_ladder_returns_a_rung_per_bucket(client):
    resp = client.post("/range_ladder", json={
        "board": "6s7s4s2h9d",
        "dead": ["AsKs9h2c", "QsJs8c3h"],
        "heroes": [{"id": "u1", "cards": "AsKs9h2c"}],
        "hands": 1000,
        "seed": 3,
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["exact"] is True
    assert [r["bucket"] for r in body["ladders"][0]["rungs"]] == [5, 15, 25, 40, 60, 100]


def test_missing_board_is_a_400(client):
    resp = client.post("/range_ladder", json={"heroes": [], "dead": []})
    assert resp.status_code == 400


def test_hero_not_in_dead_is_a_422(client):
    resp = client.post("/range_ladder", json={
        "board": "6s7s4s",
        "dead": ["AsKs9h2c"],
        "heroes": [{"id": "u1", "cards": "QsJs8c3h"}],
        "hands": 500, "seed": 1,
    })
    assert resp.status_code == 422
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_range_ladder_endpoint.py -q`
Expected: FAIL — 404 on `/range_ladder`.

- [ ] **Step 3: Write minimal implementation**

Add to `server.py`, following the `/pql` route's shape (auth check, `get_json`,
typed error codes, `app.logger.error` + traceback on unexpected failures):

```python
from range_ladder import compute_range_ladder, DEFAULT_BUCKETS


@app.route('/range_ladder', methods=['POST'])
def range_ladder_endpoint():
    """Hero equity vs board-strength percentile slices of the villain population."""
    if not _engine_authorized():
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(silent=True) or {}

    board = data.get('board')
    heroes = data.get('heroes')
    if not board or not heroes:
        return jsonify({"error": "Missing required field: board and heroes"}), 400

    try:
        result = compute_range_ladder(
            board=board,
            dead=data.get('dead') or [],
            heroes=heroes,
            buckets=tuple(data.get('buckets') or DEFAULT_BUCKETS),
            hands=int(data.get('hands') or 10000),
            runouts=data.get('runouts'),
            seed=data.get('seed'),
        )
    except ValueError as ve:
        return jsonify({"error": "Invalid input", "details": str(ve)}), 422
    except Exception as e:
        app.logger.error(f"Error in /range_ladder: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

    return jsonify(result)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/ -q`
Expected: PASS across the whole suite.

- [ ] **Step 5: Commit**

```bash
git add server.py tests/test_range_ladder_endpoint.py
git commit -m "feat(ladder): POST /range_ladder endpoint"
```

---

## Out of scope for this plan

Deferred deliberately — the ask was numbers and accuracy first:

- Any `postfloper` change: `api/range_ladder.js` proxy, `rangeLadderApi.ts`,
  `rangeLadderCache.ts`, `useRangeLadder.ts`, the panel, and deleting
  `runoutStats`.
- Railway deployment of the new endpoint.
- Panel layout and visual design.

## Self-review notes

- **Spec coverage.** §4.1 shared population → Task 6 (one call, many heroes).
  §4.2 algorithm → Tasks 1–6. §4.3 collisions → Task 2 + the skip test in Task 4.
  §5 wire contract → Task 9. §6 hero-independence → Task 7 property. §8
  performance → Task 8. §9 testing → Tasks 7 and 9. §7 postfloper integration is
  explicitly out of scope above.
- **Known rough edge.** `describe_category` in Task 6 depends on whatever
  `hand_categories.py` exposes; Task 6 Step 3 begins by inspecting it rather than
  assuming an API. If it turns out not to expose a usable category function, the
  fallback is to derive the label from the score bands in `score_array`, which
  Task 6's test pins via the `"flush"` assertion either way.
- **Category board on flop/turn.** `build_rungs` labels the boundary hand using
  `runout_rows[0]`, i.e. one sampled completion, so on a flop the label reads as
  "what this hand becomes on one representative runout". That is a simplification
  and it is visible in the output; if it reads badly in practice, the honest fix
  is to label with the hand's *current* made category on the partial board rather
  than a sampled one.
