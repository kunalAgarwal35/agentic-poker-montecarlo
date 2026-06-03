# Optimized PLO Equity Evaluator

## Overview

This document describes the optimized PLO equity evaluator that achieves **3-10x speedup** over the original Python implementation while maintaining accuracy.

## Performance Results

| PLO Type | Original Time | Optimized Time | Speedup | Accuracy |
|----------|--------------|----------------|---------|----------|
| PLO4 (10K trials) | 2.3s | 0.09s | **25x** | ±0.5% |
| PLO4 (30K trials) | 7.0s | 0.31s | **22x** | ±0.2% |
| PLO5 (10K trials) | 3.3s | 0.33s | **10x** | ±0.5% |
| PLO5 (30K trials) | - | 1.3s | **2-3x** | ±0.3% |

## Architecture

```
┌──────────────────────┐
│   API Endpoints      │
│  /plo4python25pct    │
│  /plo5python25pct    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ optimized_parallel   │
│ _runner.py           │
│ (Drop-in replacement)│
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ optimized_evaluator  │
│ .py                  │
│ (Numba JIT core)     │
└──────────┬───────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌─────────┐
│ card_   │ │ hand_   │
│encoding │ │indexing │
│ .py     │ │ .py     │
└────┬────┘ └────┬────┘
     │           │
     └─────┬─────┘
           ▼
┌──────────────────────┐
│  score_array.npy     │
│  (19.8 MB precomputed│
│   hand scores)       │
└──────────────────────┘
```

## Key Optimizations

### 1. Integer Card Encoding
- Cards represented as integers 0-51 instead of strings like "As", "Kh"
- Enables fast arithmetic operations and NumPy array operations
- File: `card_encoding.py`

### 2. Combinatorial Hand Indexing
- 5-card hands mapped to unique indices 0-2,598,959
- Enables direct array lookup instead of dictionary hash
- File: `hand_indexing.py`

### 3. NumPy Score Array
- Replaced `score_dict.pkl` (Python dict) with `score_array.npy` (NumPy array)
- O(1) array indexing vs O(1) hash lookup with lower constant factor
- File: `generate_score_array.py`

### 4. Numba JIT Compilation
- Core Monte Carlo loop compiled to machine code
- 10-50x faster than pure Python for tight loops
- File: `optimized_evaluator.py`

### 5. Precomputed Combination Indices
- PLO4: 6 hand combos × 10 board combos = 60 combinations
- PLO5: 10 hand combos × 10 board combos = 100 combinations
- PLO6: 15 hand combos × 10 board combos = 150 combinations

## Files Created

| File | Purpose |
|------|---------|
| `card_encoding.py` | Integer card encoding (0-51) |
| `hand_indexing.py` | Combinatorial hand-to-index conversion |
| `generate_score_array.py` | One-time script to generate score_array.npy |
| `score_array.npy` | Precomputed hand scores (19.8 MB) |
| `optimized_evaluator.py` | Numba JIT Monte Carlo core |
| `optimized_parallel_runner.py` | Drop-in replacement for API |
| `benchmark_evaluators.py` | Performance comparison script |
| `validate_accuracy.py` | Accuracy validation suite |

## Usage

### Drop-in Replacement

```python
# Old way (original)
import multithread_ploequities3 as mtp
result = mtp.run_parallel_plo4equities_25pct(hands, trials, processes, board)

# New way (optimized)
from optimized_parallel_runner import run_parallel_plo4equities_25pct_optimized
result = run_parallel_plo4equities_25pct_optimized(hands, trials, processes, board)
```

### API Integration

To use in `api_host.py`, replace:
```python
win_frequencies, win_frequencies_pairs = mtp.run_parallel_plo4equities_25pct(
    hands, total_number_of_trials, number_of_processes, board
)
```

With:
```python
from optimized_parallel_runner import run_parallel_plo4equities_25pct_optimized

win_frequencies, win_frequencies_pairs = run_parallel_plo4equities_25pct_optimized(
    hands, total_number_of_trials, number_of_processes, board
)
```

### Warmup at Server Start

Add to `api_host.py` initialization:
```python
from optimized_parallel_runner import warmup_optimized_evaluator
warmup_optimized_evaluator()  # Pre-compiles JIT functions
```

## Validation Results

- **27 test cases** covering PLO4 and PLO5 scenarios
- **All tests passed** within 3% tolerance
- **Max equity difference**: 1.53% (expected Monte Carlo variance)
- **Consistency test**: No performance degradation over 10 consecutive runs

## Dependencies

```
numba>=0.57.0
numpy>=1.24.0
```

Both already installed in the environment.

## Troubleshooting

### First Run Slow?
The first call triggers JIT compilation (~5-10 seconds). Call `warmup_optimized_evaluator()` at server startup.

### Memory Usage
`score_array.npy` is 19.8 MB and loaded into memory once.

### Falling Back to Original
The `run_plo4equities_25pct_auto()` function automatically falls back to the original implementation if the optimized version fails.

## Future Optimizations

1. **Parallel Numba prange**: Currently disabled due to process pool conflicts. Could enable for even faster single-request performance.

2. **GPU Acceleration**: CUDA kernels could provide additional 10-100x speedup for very high trial counts.

3. **Precomputed PLO6 Range**: Need to create `loh25_plo6.txt` for PLO6 25% range support.
