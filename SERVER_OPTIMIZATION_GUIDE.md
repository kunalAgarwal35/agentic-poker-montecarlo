# Server-Side Optimization Guide

## Current Bottleneck

The `api_host.py` server uses `ProcessPoolExecutor` which creates new processes for **every request**:

```python
# Current (slow) - creates processes per request
def run_parallel_plo6equities(...):
    with ProcessPoolExecutor(max_workers=number_of_processes) as executor:
        futures = [executor.submit(plo6equities, ...) for _ in range(number_of_processes)]
```

This adds ~5-6 seconds of overhead per request.

## Solution: Persistent Worker Pool

### Option 1: Global ProcessPoolExecutor (Simplest)

In `multithread_ploequities3.py`, create a global executor that persists:

```python
from concurrent.futures import ProcessPoolExecutor
import atexit

# Create pool once at module load
_EXECUTOR = None
_NUM_WORKERS = 4

def get_executor():
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ProcessPoolExecutor(max_workers=_NUM_WORKERS)
        atexit.register(_EXECUTOR.shutdown)
    return _EXECUTOR

def run_parallel_plo6equities(hands, total_number_of_trials, number_of_processes, board=[], opponent_hands=[]):
    trials_per_process = total_number_of_trials // number_of_processes
    executor = get_executor()  # Reuse existing pool!
    
    futures = [executor.submit(plo6equities, hands, trials_per_process, board, opponent_hands) 
               for _ in range(number_of_processes)]
    results = [future.result() for future in futures]
    
    return aggregate_results(results, total_number_of_trials)
```

### Option 2: Celery Task Queue (Production-Grade)

For production, use Celery with Redis/RabbitMQ:

```bash
pip install celery redis
```

```python
# tasks.py
from celery import Celery

app = Celery('tasks', broker='redis://localhost:6379')

@app.task
def calculate_plo6_equity(hands, trials, board, opponent_hands):
    return plo6equities(hands, trials, board, opponent_hands)
```

### Option 3: Pre-forked Workers with Gunicorn

Run Flask with Gunicorn's pre-fork model:

```bash
pip install gunicorn
gunicorn -w 4 --preload api_host:app
```

The `--preload` flag loads the app before forking, so score_dict is shared across workers.

## Expected Performance After Fix

| Configuration | Before | After |
|--------------|--------|-------|
| 1k trials | ~6.5s | ~1.5s |
| 2.5k trials | ~7s | ~2s |
| 10k trials | ~9-10s | ~4s |

## Quick Test

After implementing Option 1, run:

```python
import calling_api
import time

t1 = time.time()
result = calling_api.preflop_multithread_plo6(hands, total_number_of_trials=2500)
print(f"Time: {time.time()-t1:.2f}s")  # Should be ~2s
```
