"""The one persistent ProcessPoolExecutor, and nothing else.

This module exists to be *importable from a slim container*. It was carved out
of `multithread_ploequities3`, which still re-exports every name below so
existing callers are unaffected and there is still exactly one pool per
process -- the whole point of a global executor.

Why carve it out: `range_ladder` needed exactly one function from
`multithread_ploequities3` (`get_global_executor`), but importing it dragged in
`pandas`, and then `generating_list` -> `display_scenario` -> `cv2`. Neither
pandas nor OpenCV is in `requirements-engine.txt`, and neither belongs in an
engine image -- so the deployed container crashed on import even after its
COPY list was fixed, and would have needed ~100MB of wheels that compute
nothing on this path. Splitting the pool out drops pandas, cv2,
`generating_list`, `ranking` and `display_scenario` from the engine image's
dependency closure entirely.

Keeping the pool here (rather than duplicating it) preserves the invariant that
matters: `multithread_ploequities3.get_global_executor` and
`range_ladder`'s are the SAME object, so the two never race to build competing
pools in a process that imports both.

Creating a ProcessPoolExecutor costs ~5 seconds, which is why it is created
once and reused rather than per request.
"""

import atexit
import os
from concurrent.futures import ProcessPoolExecutor

_GLOBAL_EXECUTOR = None
_DEFAULT_WORKERS = 4  # Match server core count


def get_global_executor(num_workers=None):
    """Get or create a persistent ProcessPoolExecutor.

    The FIRST caller's `num_workers` wins for the life of the process -- the
    pool is not resized on later calls. Callers that care about the size
    (range_ladder does) should therefore be the ones to create it.
    """
    global _GLOBAL_EXECUTOR
    if _GLOBAL_EXECUTOR is None:
        workers = num_workers or _DEFAULT_WORKERS
        print(f"[ProcessPool] Initializing global executor with {workers} workers...")
        _GLOBAL_EXECUTOR = ProcessPoolExecutor(max_workers=workers)
        atexit.register(shutdown_executor)
        print(f"[ProcessPool] Global executor ready (PID: {os.getpid()})")
    return _GLOBAL_EXECUTOR


def shutdown_executor():
    """Cleanup function called at exit."""
    global _GLOBAL_EXECUTOR
    if _GLOBAL_EXECUTOR is not None:
        print("[ProcessPool] Shutting down global executor...")
        _GLOBAL_EXECUTOR.shutdown(wait=True)
        _GLOBAL_EXECUTOR = None


def _dummy_warmup_task():
    """Simple task used to warm up process pool workers."""
    return 1


def warmup_executor():
    """Pre-warm the executor by running a dummy task on each worker."""
    executor = get_global_executor()
    # Submit dummy tasks to ensure all workers are spawned
    futures = [executor.submit(_dummy_warmup_task) for _ in range(_DEFAULT_WORKERS)]
    for f in futures:
        f.result()
    print("[ProcessPool] Workers warmed up and ready")
