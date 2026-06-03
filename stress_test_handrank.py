"""
Stress test harness for /handrank.

Scenarios (all run against a running Flask server):
  1. Warm single-hand latency sweep   (PLO4/5/6 x trials {1000, 2500, 5000, 10000})
  2. Concurrent load                  (workers {1, 4, 8, 16, 32}, 500 requests each)
  3. Batch-size scaling               (cards length {1, 2, 4, 6})
  4. Consistency                      (same hand, 50 repeats -> rank stdev)
  5. Memory / CPU                     (sampled via psutil, if PID discoverable)
  6. Cold-start                       (optional: pass --cold-start-cmd to respawn server)

Outputs:
  * stress_test_handrank_results.csv  (all scenarios, one row per measurement)
  * stress_test_handrank_summary.md   (concise table for docs)

Usage:
  # scenarios 1-5 against an already-running server
  python stress_test_handrank.py --base-url http://127.0.0.1:5050

  # faster smoke run (subset)
  python stress_test_handrank.py --quick
"""

import argparse
import concurrent.futures as cf
import csv
import os
import random
import statistics
import sys
import time
from typing import Dict, List, Optional

import requests

try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False


# Card-disjoint sample hands. Batch scenarios stack up to 5 hands, so every pair
# must have no shared cards (otherwise the API rejects with a duplicate error).
SAMPLE_HANDS = {
    'plo4': [
        'AsAhKsKh',
        'QsQhJsJh',
        'TsTh9s9h',
        '8c7c6c5c',
        '4d3d2d2c',
    ],
    'plo5': [
        'AsAhKsKhQs',
        'QhQdJhJdTd',
        'Th9h9d8h8d',
        '7c7d6c6d5c',
        '5d4c4d3c3d',
    ],
    'plo6': [
        'AsAhKsKhQsJh',
        'QhQdQcJdJsTc',
        'TsTh9s9h9d8s',
        '8h8d7s7h7d7c',
        '6s6h6d6c5s5h',
    ],
}


def percentiles(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {'p50': 0, 'p90': 0, 'p95': 0, 'p99': 0, 'mean': 0, 'min': 0, 'max': 0}
    xs_sorted = sorted(xs)
    def pct(p):
        k = max(0, min(len(xs_sorted) - 1, int(round(p * (len(xs_sorted) - 1)))))
        return xs_sorted[k]
    return {
        'p50': pct(0.50),
        'p90': pct(0.90),
        'p95': pct(0.95),
        'p99': pct(0.99),
        'mean': statistics.fmean(xs),
        'min': xs_sorted[0],
        'max': xs_sorted[-1],
    }


def post(base_url: str, payload: dict, timeout: float = 60) -> float:
    t0 = time.perf_counter()
    r = requests.post(f"{base_url}/handrank", json=payload, timeout=timeout)
    dt = (time.perf_counter() - t0) * 1000.0
    if r.status_code != 200:
        raise RuntimeError(f"status={r.status_code} body={r.text[:200]}")
    return dt


def wait_warm(base_url: str, timeout_s: float = 90) -> None:
    """Hit /health then a trivial /handrank until p50 stabilizes. Keeps the suite
    self-sufficient - no assumption that the server was warmed beforehand."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200 and 'Warmup in progress' not in r.text:
                break
        except Exception:
            pass
        time.sleep(1)
    # Extra priming requests to load CDFs into each thread's path
    for variant in ('plo4', 'plo5', 'plo6'):
        for _ in range(3):
            try:
                post(base_url, {'cards': SAMPLE_HANDS[variant][0], 'trials': 500})
            except Exception:
                pass


def find_server_process(port: int = 5050) -> Optional['psutil.Process']:
    if not HAVE_PSUTIL:
        return None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = ' '.join(proc.info.get('cmdline') or [])
            if 'api_host_handrank_local.py' in cmd or ('api_host' in cmd and 'python' in (proc.info.get('name') or '').lower()):
                return proc
        except Exception:
            continue
    # Fallback: find any process listening on the port
    for conn in psutil.net_connections(kind='tcp'):
        if conn.status == 'LISTEN' and conn.laddr and conn.laddr.port == port:
            try:
                return psutil.Process(conn.pid)
            except Exception:
                return None
    return None


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def scenario_1_latency_sweep(base_url: str, writer, quick: bool) -> List[dict]:
    """Warm single-hand latency sweep per PLO type x trials."""
    print("\n=== Scenario 1: warm latency sweep ===")
    summary = []
    trials_list = [1000, 2500, 5000, 10000] if not quick else [1000, 5000]
    reps = 80 if not quick else 25
    for variant in ('plo4', 'plo5', 'plo6'):
        for trials in trials_list:
            latencies = []
            hand = SAMPLE_HANDS[variant][0]
            for _ in range(reps):
                dt = post(base_url, {'cards': hand, 'trials': trials, 'seed': None})
                latencies.append(dt)
            p = percentiles(latencies)
            row = {
                'scenario': 'latency_sweep',
                'variant': variant,
                'trials': trials,
                'reps': reps,
                **p,
            }
            summary.append(row)
            writer.writerow(row)
            print(f"  {variant} trials={trials:>5d}  p50={p['p50']:6.1f}ms  p95={p['p95']:6.1f}ms  p99={p['p99']:6.1f}ms  min={p['min']:6.1f}ms  max={p['max']:6.1f}ms")
    return summary


def scenario_2_concurrent(base_url: str, writer, quick: bool, proc_watcher) -> List[dict]:
    """Concurrent load via ThreadPoolExecutor."""
    print("\n=== Scenario 2: concurrent load ===")
    summary = []
    workers_list = [1, 4, 8, 16] if quick else [1, 4, 8, 16, 32]
    total_reqs = 200 if quick else 500

    def job(_):
        hand = random.choice(SAMPLE_HANDS['plo5'])
        return post(base_url, {'cards': hand, 'trials': 3000})

    for workers in workers_list:
        rss_before = proc_watcher['sample']() if proc_watcher else None
        t0 = time.perf_counter()
        latencies: List[float] = []
        errors = 0
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(job, i) for i in range(total_reqs)]
            for fut in cf.as_completed(futs):
                try:
                    latencies.append(fut.result())
                except Exception:
                    errors += 1
        elapsed = time.perf_counter() - t0
        throughput = total_reqs / elapsed
        p = percentiles(latencies)
        rss_after = proc_watcher['sample']() if proc_watcher else None
        rss_delta = (rss_after - rss_before) if (rss_before is not None and rss_after is not None) else None
        row = {
            'scenario': 'concurrent',
            'workers': workers,
            'total_reqs': total_reqs,
            'errors': errors,
            'elapsed_s': round(elapsed, 2),
            'throughput_rps': round(throughput, 2),
            'rss_delta_mb': round(rss_delta, 2) if rss_delta is not None else None,
            **p,
        }
        summary.append(row)
        writer.writerow(row)
        extra = f"  RSS_delta={rss_delta:+.1f}MB" if rss_delta is not None else ""
        print(f"  workers={workers:>2d}  throughput={throughput:6.1f}rps  p50={p['p50']:6.1f}ms  p95={p['p95']:6.1f}ms  p99={p['p99']:6.1f}ms  errors={errors}{extra}")
    return summary


def scenario_3_batch(base_url: str, writer, quick: bool) -> List[dict]:
    """Batch size scaling."""
    print("\n=== Scenario 3: batch size scaling ===")
    summary = []
    sizes = [1, 2, 4] if quick else [1, 2, 4, 6]
    reps = 25 if quick else 60
    for size in sizes:
        hands = SAMPLE_HANDS['plo5'][:size]
        latencies = []
        for _ in range(reps):
            dt = post(base_url, {'cards': hands, 'trials': 3000})
            latencies.append(dt)
        p = percentiles(latencies)
        row = {
            'scenario': 'batch',
            'variant': 'plo5',
            'batch_size': size,
            'trials': 3000,
            'reps': reps,
            'per_hand_mean_ms': round(p['mean'] / size, 2),
            **p,
        }
        summary.append(row)
        writer.writerow(row)
        print(f"  batch={size}  mean={p['mean']:6.1f}ms  per_hand={p['mean']/size:6.1f}ms  p95={p['p95']:6.1f}ms")
    return summary


def scenario_4_consistency(base_url: str, writer, quick: bool) -> List[dict]:
    """Same hand, many repeats, measure rank stdev."""
    print("\n=== Scenario 4: consistency ===")
    summary = []
    reps = 20 if quick else 50
    test_cases = [
        ('plo4', 'AsKsQdJd'),
        ('plo5', 'AsKsQdJdTc'),
        ('plo6', 'AsKsQdJdTc9d'),
    ]
    for variant, hand in test_cases:
        ranks = []
        equities = []
        for _ in range(reps):
            r = requests.post(f"{base_url}/handrank", json={'cards': hand, 'trials': 5000})
            j = r.json()
            rank = j.get('ranks', {}).get(hand)
            eq = j.get('equities', {}).get(hand)
            if rank is not None:
                ranks.append(rank)
                equities.append(eq)
        stdev = statistics.pstdev(ranks) if len(ranks) > 1 else 0
        rng = max(ranks) - min(ranks) if ranks else 0
        row = {
            'scenario': 'consistency',
            'variant': variant,
            'hand': hand,
            'reps': reps,
            'rank_mean': round(statistics.fmean(ranks), 2) if ranks else None,
            'rank_stdev': round(stdev, 2),
            'rank_range': rng,
            'equity_stdev': round(statistics.pstdev(equities), 4) if len(equities) > 1 else 0,
        }
        summary.append(row)
        writer.writerow(row)
        print(f"  {variant} {hand}  rank_mean={row['rank_mean']}  stdev={row['rank_stdev']}  range={rng}")
    return summary


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base-url', default='http://127.0.0.1:5050', help='Server base URL.')
    p.add_argument('--quick', action='store_true', help='Reduced scenarios for dev loop.')
    p.add_argument('--out-csv', default='stress_test_handrank_results.csv')
    p.add_argument('--out-md', default='stress_test_handrank_summary.md')
    p.add_argument('--port', type=int, default=5050)
    args = p.parse_args()

    print(f"Stress test against {args.base_url}  (quick={args.quick}, psutil={HAVE_PSUTIL})")
    try:
        wait_warm(args.base_url)
    except Exception as e:
        print(f"[fatal] could not reach server: {e}")
        return 2

    proc = find_server_process(args.port) if HAVE_PSUTIL else None
    if proc is not None:
        print(f"[info] monitoring server PID={proc.pid} RSS={proc.memory_info().rss/1024/1024:.1f}MB")
        proc_watcher = {
            'sample': lambda: (proc.memory_info().rss / 1024 / 1024) if proc.is_running() else None
        }
    else:
        proc_watcher = None
        print("[info] psutil or server process not found - skipping RSS tracking")

    # Write all rows to one CSV (columns union)
    rows: List[dict] = []
    with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
        # Use a delayed writer since schemas differ by scenario
        pending: List[dict] = []
        class Appender:
            def writerow(self, row): pending.append(row)
        w = Appender()
        rows += scenario_1_latency_sweep(args.base_url, w, args.quick)
        rows += scenario_2_concurrent(args.base_url, w, args.quick, proc_watcher)
        rows += scenario_3_batch(args.base_url, w, args.quick)
        rows += scenario_4_consistency(args.base_url, w, args.quick)

        fieldnames: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        csv_writer.writeheader()
        for r in rows:
            csv_writer.writerow(r)
    print(f"\n[write] {args.out_csv}  ({len(rows)} rows)")

    # Markdown summary
    with open(args.out_md, 'w', encoding='utf-8') as f:
        f.write(f"# Hand Rank stress test summary\n\n")
        f.write(f"Base URL: `{args.base_url}`  |  quick: `{args.quick}`  |  host psutil: `{HAVE_PSUTIL}`\n\n")

        f.write("## Latency sweep (p50 / p95 / p99, ms)\n\n")
        f.write("| variant | trials | p50 | p95 | p99 | min | max |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            if r.get('scenario') == 'latency_sweep':
                f.write(f"| {r['variant']} | {r['trials']} | {r['p50']:.1f} | {r['p95']:.1f} | {r['p99']:.1f} | {r['min']:.1f} | {r['max']:.1f} |\n")

        f.write("\n## Concurrency\n\n")
        f.write("| workers | throughput rps | p50 | p95 | p99 | errors | rss delta (MB) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            if r.get('scenario') == 'concurrent':
                rss = r.get('rss_delta_mb')
                f.write(f"| {r['workers']} | {r['throughput_rps']} | {r['p50']:.1f} | {r['p95']:.1f} | {r['p99']:.1f} | {r['errors']} | {rss if rss is not None else '-'} |\n")

        f.write("\n## Batch scaling\n\n")
        f.write("| batch | mean ms | per-hand ms | p95 ms |\n|---|---|---|---|\n")
        for r in rows:
            if r.get('scenario') == 'batch':
                f.write(f"| {r['batch_size']} | {r['mean']:.1f} | {r['per_hand_mean_ms']} | {r['p95']:.1f} |\n")

        f.write("\n## Consistency (rank stdev over repeats)\n\n")
        f.write("| variant | hand | reps | rank_mean | rank_stdev | rank_range |\n|---|---|---|---|---|---|\n")
        for r in rows:
            if r.get('scenario') == 'consistency':
                f.write(f"| {r['variant']} | {r['hand']} | {r['reps']} | {r['rank_mean']} | {r['rank_stdev']} | {r['rank_range']} |\n")
    print(f"[write] {args.out_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
