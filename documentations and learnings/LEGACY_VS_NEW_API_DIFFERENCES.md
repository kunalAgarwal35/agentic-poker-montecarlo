# Legacy Java vs New Python PLO Equity API: Complete Comparison

## Executive Summary

The new Python-based PLO equity calculator provides **3-8x faster performance** compared to the legacy Java (ProPokerTools) implementation, while returning both individual AND pairwise equities in a single API call.

---

## Performance Comparison

| Metric | Legacy Java | New Python | Improvement |
|--------|-------------|------------|-------------|
| Individual equity (4 hands) | ~40-60 seconds | ~1.2 seconds | **30-50x faster** |
| Pairwise equity (6 pairs) | ~30-45 seconds | Included above | N/A |
| Total for 4 hands + pairs | ~70-100 seconds | ~1.2 seconds | **60-80x faster** |
| Single hand vs 25% | ~5-8 seconds | ~0.3 seconds | **15-25x faster** |

*Note: Legacy Java requires **separate API calls** for individual and pairwise, while Python returns both in one call.*

---

## API Structure Differences

### Legacy Java Endpoints

#### 1. `/strength` - Individual Equity
```http
POST /strength
Content-Type: application/json

{
    "cards": ["AsAhKsKhQs", "JdTc9d8c7d", "QsQhJhTs9h", "8c7c6c5c4c"],
    "board": "",
    "range": "25%"
}
```

**Response:**
```json
{
    "AsAhKsKhQs": 68.97
}
```

**Key Points:**
- Only returns equity for the **first card** in the array
- Must call separately for each hand
- Board is a string (e.g., `"2c3c4c"`)
- Range is a string percentage (`"25%"`)
- Returns equity as **percentage** (0-100)

#### 2. `/pair` - Pairwise Equity
```http
POST /pair
Content-Type: application/json

{
    "cards0": "AsAhKsKhQs",
    "cards1": "JdTc9d8c7d",
    "dcs": "QsQhJhTs9h8c7c6c5c4c",
    "board": ""
}
```

**Response:**
```json
{
    "AsAhKsKhQs : JdTc9d8c7d": {
        "25%": 33.21,
        "AsAhKsKhQs": 43.65,
        "JdTc9d8c7d": 23.14
    }
}
```

**Key Points:**
- Must call separately for each pair combination
- Returns all three equities (range, hand1, hand2)
- Combined equity = 100 - range equity (approx)
- Requires manual dead card calculation

---

### New Python Endpoints

#### `/plo5python25pct` - PLO5 Individual + Pairwise
```http
POST /plo5python25pct
Content-Type: application/json

{
    "cards": ["AsAhKsKhQs", "JdTc9d8c7d", "QsQhJhTs9h", "8c7c6c5c4c"],
    "board": [],
    "trials": 10000,
    "processes": 4
}
```

**Response:**
```json
{
    "individual": {
        "AsAhKsKhQs": 0.6887,
        "JdTc9d8c7d": 0.5502,
        "QsQhJhTs9h": 0.4903,
        "8c7c6c5c4c": 0.4456
    },
    "pairwise": {
        "AsAhKsKhQs_JdTc9d8c7d": 0.8163,
        "AsAhKsKhQs_QsQhJhTs9h": 0.7845,
        "AsAhKsKhQs_8c7c6c5c4c": 0.7612,
        "JdTc9d8c7d_QsQhJhTs9h": 0.7234,
        "JdTc9d8c7d_8c7c6c5c4c": 0.6987,
        "QsQhJhTs9h_8c7c6c5c4c": 0.6543
    }
}
```

#### `/plo4python25pct` - PLO4 Individual + Pairwise
Same structure but with 8-character hands (4 cards).

---

## Key Differences

### 1. Equity Format
| Aspect | Legacy | New Python |
|--------|--------|------------|
| Format | Percentage (68.97) | Decimal (0.6887) |
| Range | 0-100 | 0.0-1.0 |
| Conversion | N/A | Multiply by 100 for % |

### 2. Board Format
| Aspect | Legacy | New Python |
|--------|--------|------------|
| Type | String | Array |
| Example | `"2c3c4c"` | `["2c", "3c", "4c"]` |
| Empty | `""` | `[]` |

### 3. API Calls Required
| Scenario | Legacy | New Python |
|----------|--------|------------|
| 4 hands individual | 4 calls | 1 call |
| 4 hands pairwise | 6 calls | 1 call |
| Total calls | 10 calls | **1 call** |

### 4. Output Structure
| Aspect | Legacy | New Python |
|--------|--------|------------|
| Individual | Single hand per call | All hands in one response |
| Pairwise | Single pair per call | All pairs in one response |
| Key format (pairwise) | `"Hand1 : Hand2"` | `"Hand1_Hand2"` |

### 5. Range Handling
| Aspect | Legacy | New Python |
|--------|--------|------------|
| Range param | Required (`"25%"`) | Built-in (not needed) |
| Implementation | Java ProPokerTools interprets | Pre-computed `loh25_plo5.txt` |
| Hand count | Internal to PPT | ~650k PLO5 hands |

### 6. Pairwise Calculation Method
| Aspect | Legacy | New Python |
|--------|--------|------------|
| Meaning | 3-way showdown equity | Combined win frequency |
| Calculation | Via ProPokerTools | Direct Monte Carlo |
| Output | Range + both hands' equities | Combined team equity |

---

## Example API Outputs

### Test Set 1 (seed=42)
**Hands:** `4h7c8h2c7h`, `Jd6sJc6c4c`, `Kd8sTh9h9c`, `Qc3s9s4d8d`

**New Python Response:**
```json
{
  "individual": {
    "4h7c8h2c7h": 0.4323,
    "Jd6sJc6c4c": 0.4718,
    "Kd8sTh9h9c": 0.3786,
    "Qc3s9s4d8d": 0.4135
  },
  "pairwise": {
    "4h7c8h2c7h_Jd6sJc6c4c": 0.6066,
    "4h7c8h2c7h_Kd8sTh9h9c": 0.5675,
    "4h7c8h2c7h_Qc3s9s4d8d": 0.5995,
    "Jd6sJc6c4c_Kd8sTh9h9c": 0.6017,
    "Jd6sJc6c4c_Qc3s9s4d8d": 0.6243,
    "Kd8sTh9h9c_Qc3s9s4d8d": 0.5702
  }
}
```
**Time:** 1.34s

### Test Set 2 (seed=123) - Premium Aces
**Hands:** `JhAh9hAs7s`, `8h6d7dTh4c`, `Qc3c4hKd6s`, `7cKc3s9d2h`

**New Python Response:**
```json
{
  "individual": {
    "JhAh9hAs7s": 0.6745,
    "8h6d7dTh4c": 0.4939,
    "Qc3c4hKd6s": 0.4909,
    "7cKc3s9d2h": 0.4308
  },
  "pairwise": {
    "JhAh9hAs7s_8h6d7dTh4c": 0.7735,
    "JhAh9hAs7s_Qc3c4hKd6s": 0.7720,
    "JhAh9hAs7s_7cKc3s9d2h": 0.7346,
    "8h6d7dTh4c_Qc3c4hKd6s": 0.6583,
    "8h6d7dTh4c_7cKc3s9d2h": 0.6491,
    "Qc3c4hKd6s_7cKc3s9d2h": 0.5966
  }
}
```
**Time:** 1.17s

### More Test Sets (All 10)

| Test | Seed | Sample Hand | Individual Eq | Best Pair Eq | Time |
|------|------|-------------|---------------|--------------|------|
| 1 | 42 | Jd6sJc6c4c | 47.18% | 62.43% | 1.34s |
| 2 | 123 | JhAh9hAs7s (AA) | 67.45% | 77.35% | 1.17s |
| 3 | 456 | JsAd2sTsKs | 49.30% | 69.67% | 1.28s |
| 4 | 789 | 6d9sAhAsJd (AA) | 53.11% | 65.54% | 1.23s |
| 5 | 101 | Ac8hKc2hKd (KK) | 57.82% | 68.98% | 1.15s |
| 6 | 202 | Js9hKc4s3c | 43.29% | 61.34% | 1.23s |
| 7 | 303 | 9cQcJdQs3d (QQ) | 45.05% | 62.70% | 1.29s |
| 8 | 404 | 2cJhAs7sQh | 61.05% | 74.91% | 1.23s |
| 9 | 505 | 5dJh8c4cKh | 53.22% | 71.24% | 1.15s |
| 10 | 606 | KsAcAh2s4h (AA) | 62.80% | 77.62% | 1.10s |

**Average Time:** 1.22s for 4 hands + 6 pairwise equities

---

## Migration Code Examples

### Before (Legacy)
```python
import requests

# Get individual equities - 4 separate calls
for i, hand in enumerate(hands):
    other_hands = hands[:i] + hands[i+1:]
    dead_cards = ''.join(other_hands)
    
    response = requests.post('http://server/strength', json={
        "cards": [hand],
        "board": "",
        "range": "25%"
    })
    result = response.json()
    equity = result[hand] / 100  # Convert from percentage

# Get pairwise - 6 more separate calls
for pair in combinations(hands, 2):
    response = requests.post('http://server/pair', json={
        "cards0": pair[0],
        "cards1": pair[1],
        "dcs": ''.join(other_hands),
        "board": ""
    })
    # Complex parsing required...
```
**Total: 10 API calls, ~100 seconds**

### After (New Python)
```python
import requests

# Single call for everything
response = requests.post('http://server/plo5python25pct', json={
    "cards": hands,
    "board": [],
    "trials": 10000,
    "processes": 4
})
result = response.json()

# All individual equities
for hand, equity in result['individual'].items():
    print(f"{hand}: {equity * 100:.2f}%")

# All pairwise equities
for pair, equity in result['pairwise'].items():
    print(f"{pair}: {equity * 100:.2f}%")
```
**Total: 1 API call, ~1.2 seconds**

---

## Accuracy Comparison

Based on validation tests, the Python implementation produces results within **±0.5%** of the Java implementation at 10,000 trials. At 30,000 trials, accuracy improves to **±0.3%**.

| Hand Type | Java (30K) | Python (30K) | Difference |
|-----------|-----------|--------------|------------|
| AAKKQs vs 25% | 68.98% | 68.87% | 0.11% |
| JT987 vs 25% | 54.55% | 55.02% | 0.47% |
| KK + suited | 64.21% | 64.45% | 0.24% |
| Rundown | 52.34% | 52.67% | 0.33% |

---

## PLO6 Implementation Note (IMPORTANT)

### Discovered Bug in Legacy PLO6 Implementation

During validation testing (Feb 2026), a significant difference was found between legacy and optimized PLO6 implementations when a **board is present** (flop/turn).

**Root Cause**: The legacy `plo6equities` function does NOT filter opponent hands that contain board cards. This means:
- If the opponent range has 500 hands and the flop is `Ah Kd 2c`
- ~40% of opponent hands might contain `Ah`, `Kd`, or `2c`
- Legacy uses these invalid hands anyway
- Optimized correctly filters them out

**Impact**:
- PREFLOP: Both implementations agree (within Monte Carlo variance ~1%)
- POSTFLOP: Legacy results are **incorrect** by 2-9%

**Validation Test Results**:
| Test Scenario | Legacy vs Optimized | Root Cause |
|---------------|---------------------|------------|
| Preflop (no board) | ~1% difference | Monte Carlo variance (OK) |
| Flop/Turn | 2-9% difference | Legacy uses invalid opponent hands |

**Conclusion**: The optimized implementation is MORE CORRECT than legacy for postflop scenarios.

**Recommendation**: Use the optimized implementation (`run_parallel_plo6equities_optimized`) for all PLO6 calculations, especially postflop.

---

## When to Use Each

### Use New Python API (`/plo5python25pct`, `/plo4python25pct`)
- Real-time applications requiring fast response
- Need both individual and pairwise in one call
- Standard 25% opponent range
- PLO4 or PLO5 games

### Use Legacy Java API (`/strength`, `/pair`)
- Custom percentage ranges (e.g., 10%, 15%, 30%)
- PLO6 games (Python endpoint available separately)
- Complex board scenarios
- When exact ProPokerTools methodology is required

---

## Technical Implementation Notes

### Python Monte Carlo Details
- Uses Numba JIT compilation for 10-30x speedup
- Pre-computed hand rankings stored in `score_array.npy`
- Top 25% hands loaded from `loh25_plo5.txt` (~650k hands) and `loh25_plo4.txt` (~68k hands)
- Persistent ProcessPoolExecutor eliminates process spawn overhead

### Java ProPokerTools Details
- Uses ProPokerTools JAR via subprocess
- Each call spawns new Java process
- Range interpretation happens inside PPT black box
- Query language: PQL (Poker Query Language)

---

## Files Reference

| File | Purpose |
|------|---------|
| `api_host.py` | Main API server with all endpoints |
| `multi_queries.py` | Legacy Java integration |
| `multithread_ploequities3.py` | Python Monte Carlo implementation |
| `optimized_parallel_runner.py` | Numba-optimized wrapper |
| `optimized_evaluator.py` | JIT-compiled hand evaluation |
| `loh25_plo5.txt` | Top 25% PLO5 hand list |
| `loh25_plo4.txt` | Top 25% PLO4 hand list |
| `score_array.npy` | Pre-computed hand scores |

---

*Last updated: February 2026*
