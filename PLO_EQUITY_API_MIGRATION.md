# PLO4/PLO5 Equity API Migration Guide

## Overview

This document describes the migration from Java-based ProPokerTools equity calculations to new Python-based endpoints for PLO4 and PLO5 games. The new endpoints provide **3-8x faster performance** while maintaining accuracy.

### Why Migrate?

| Aspect | Old (Java) | New (Python) |
|--------|-----------|--------------|
| Response Time | ~8 seconds | ~1-3 seconds |
| Technology | Java subprocess (ProPokerTools) | Native Python Monte Carlo |
| Output | Single hand equity | Individual + Pairwise equities |
| Range | "25%" string | Precomputed top 25% hands |

---

## Endpoint Reference

### New Endpoints

#### 1. `/plo5python25pct` - PLO5 Equity vs Top 25% Range

**Method:** `POST`

**Description:** Calculates equity for PLO5 hands against the top 25% of opponent starting hands.

**Request Body:**
```json
{
    "cards": ["AsAhKsKhQs", "JdTc9d8c7d"],
    "board": ["2c", "3c", "4c"],
    "trials": 10000,
    "processes": 4
}
```

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `cards` | array | Yes | - | Array of PLO5 hands (10 characters each) |
| `board` | array | No | `[]` | Board cards (0-5 cards) |
| `trials` | int | No | `10000` | Number of Monte Carlo trials |
| `processes` | int | No | `4` | Number of parallel processes |

**Response:**
```json
{
    "individual": {
        "AsAhKsKhQs": 0.6887,
        "JdTc9d8c7d": 0.5502
    },
    "pairwise": {
        "AsAhKsKhQs_JdTc9d8c7d": 0.8163
    }
}
```

**Response Fields:**
- `individual`: Equity of each hand vs the 25% range (0.0 to 1.0)
- `pairwise`: Combined equity when either hand in the pair wins (for multi-way scenarios)

---

#### 2. `/plo4python25pct` - PLO4 Equity vs Top 25% Range

**Method:** `POST`

**Description:** Calculates equity for PLO4 hands against the top 25% of opponent starting hands.

**Request Body:**
```json
{
    "cards": ["AsAhKsKh", "JdTc9d8c"],
    "board": [],
    "trials": 10000,
    "processes": 4
}
```

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `cards` | array | Yes | - | Array of PLO4 hands (8 characters each) |
| `board` | array | No | `[]` | Board cards (0-5 cards) |
| `trials` | int | No | `10000` | Number of Monte Carlo trials |
| `processes` | int | No | `4` | Number of parallel processes |

**Response:**
```json
{
    "individual": {
        "AsAhKsKh": 0.7646,
        "JdTc9d8c": 0.5263
    },
    "pairwise": {
        "AsAhKsKh_JdTc9d8c": 0.8512
    }
}
```

---

## Migration from Old Endpoints

### Old Endpoint: `/strength`

The `/strength` endpoint used Java ProPokerTools and only returned single-hand equity.

**Old Request:**
```json
{
    "cards": ["AsAhKsKhQs", "JdTc9d8c7d"],
    "board": "",
    "range": "25%"
}
```

**Old Response:**
```json
{
    "AsAhKsKhQs": 68.97
}
```

### Migration Mapping

| Old Endpoint | New Endpoint | When to Use |
|--------------|--------------|-------------|
| `/strength` (8-char cards) | `/plo4python25pct` | PLO4 (4-card Omaha) |
| `/strength` (10-char cards) | `/plo5python25pct` | PLO5 (5-card Omaha) |

---

## Code Migration Examples

### Before (Python client using old endpoint)

```python
import requests

# Old approach - single hand at a time
response = requests.post('http://server/strength', json={
    "cards": ["AsAhKsKhQs", "JdTc9d8c7d"],
    "board": "",
    "range": "25%"
})
result = response.json()
# Returns: {"AsAhKsKhQs": 68.97}
equity = result["AsAhKsKhQs"] / 100  # Convert to decimal
```

### After (Python client using new endpoint)

```python
import requests

# New approach - multiple hands with pairwise
response = requests.post('http://server/plo5python25pct', json={
    "cards": ["AsAhKsKhQs", "JdTc9d8c7d"],
    "board": [],
    "trials": 10000,
    "processes": 4
})
result = response.json()
# Returns: {"individual": {"AsAhKsKhQs": 0.6887, ...}, "pairwise": {...}}
equity = result["individual"]["AsAhKsKhQs"]  # Already decimal (0.0 to 1.0)
```

### Key Differences

| Aspect | Old `/strength` | New `/plo5python25pct` |
|--------|----------------|------------------------|
| Equity format | Percentage (68.97) | Decimal (0.6887) |
| Board format | String `"2c3c4c"` | Array `["2c", "3c", "4c"]` |
| Output | Single hand | All hands + pairwise |
| Range parameter | Required (`"25%"`) | Not needed (built-in) |

---

## Performance Comparison

Benchmarks run with same hands, warm process pool:

| Endpoint | Trials | Time | Accuracy vs Java |
|----------|--------|------|------------------|
| `/strength` (Java) | 30,000 | ~5-8s | Baseline |
| `/plo5python25pct` | 10,000 | ~2.5s | ±0.5% |
| `/plo5python25pct` | 30,000 | ~3.9s | ±0.3% |
| `/plo4python25pct` | 10,000 | ~0.9s | ±0.5% |
| `/plo4python25pct` | 30,000 | ~1.5s | ±0.3% |

### Accuracy Validation

Tested with same hands against both implementations:

| Hand | Java (30K) | Python (30K) | Difference |
|------|-----------|--------------|------------|
| AAKKQs vs 25% | 68.98% | 68.87% | 0.11% |
| JT987 vs 25% | 54.55% | 55.02% | 0.47% |

---

## Deprecation Notice

The `/strength` endpoint remains available but is **deprecated** for PLO4/PLO5 calculations against the 25% range. We recommend migrating to the new Python endpoints for:

- Faster response times
- Individual AND pairwise equities in one call
- More consistent results

---

## Questions?

Contact the backend team for migration support.
