"""
Comprehensive test for PLO4, PLO5, and PLO6 endpoints.
Tests both Python-based (PLO6) and Java-based (PLO4/5) endpoints.
"""
import requests
import time
import itertools
import random

BASE_URL = "http://localhost:5050"

def generate_hands(card_count, num_hands=4):
    """Generate random hands with specified card count."""
    deck = [r + s for r in '23456789TJQKA' for s in 'shdc']
    random.shuffle(deck)
    hands = [''.join(deck[i*card_count:(i+1)*card_count]) for i in range(num_hands)]
    return hands

def test_health():
    """Test health endpoint."""
    print("=" * 60)
    print("TEST: Health Check")
    print("=" * 60)
    t1 = time.time()
    r = requests.get(f"{BASE_URL}/health")
    elapsed = time.time() - t1
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text}")
    print(f"Time: {elapsed:.3f}s")
    return r.status_code == 200

def test_plo6_python():
    """Test PLO6 Python endpoint (uses ProcessPoolExecutor)."""
    print("\n" + "=" * 60)
    print("TEST: PLO6 Python (plo6pythonindividualandpairwise)")
    print("=" * 60)
    
    hands = generate_hands(6, 4)
    print(f"Hands: {hands}")
    
    payload = {
        "cards": hands,
        "trials": 10000,
        "processes": 4
    }
    
    t1 = time.time()
    r = requests.post(f"{BASE_URL}/plo6pythonindividualandpairwise", json=payload)
    elapsed = time.time() - t1
    
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Individual equities: {data.get('individual', 'N/A')}")
        pairwise = data.get('pairwise', {})
        print(f"Pairwise matchups: {len(pairwise)} pairs")
    else:
        print(f"Error: {r.text[:200]}")
    print(f"Time: {elapsed:.3f}s")
    return r.status_code == 200, elapsed

def test_plo5_python():
    """Test PLO5 Python endpoint."""
    print("\n" + "=" * 60)
    print("TEST: PLO5 Python (plo5pythonindividualandpairwise)")
    print("=" * 60)
    
    hands = generate_hands(5, 4)
    print(f"Hands: {hands}")
    
    payload = {
        "cards": hands,
        "trials": 10000,
        "processes": 4
    }
    
    t1 = time.time()
    r = requests.post(f"{BASE_URL}/plo5pythonindividualandpairwise", json=payload)
    elapsed = time.time() - t1
    
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Individual equities: {data.get('individual', 'N/A')}")
        pairwise = data.get('pairwise', {})
        print(f"Pairwise matchups: {len(pairwise)} pairs")
    else:
        print(f"Error: {r.text[:200]}")
    print(f"Time: {elapsed:.3f}s")
    return r.status_code == 200, elapsed

def test_plo4_java():
    """Test PLO4 Java endpoint (uses subprocess)."""
    print("\n" + "=" * 60)
    print("TEST: PLO4 Java (strength endpoint)")
    print("=" * 60)
    
    hands = generate_hands(4, 4)
    print(f"Hands: {hands}")
    
    payload = {
        "cards": hands,
        "board": ""
    }
    
    t1 = time.time()
    r = requests.post(f"{BASE_URL}/strength", json=payload)
    elapsed = time.time() - t1
    
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Result: {data}")
    else:
        print(f"Error: {r.text[:200]}")
    print(f"Time: {elapsed:.3f}s")
    return r.status_code == 200, elapsed

def test_plo5_java():
    """Test PLO5 Java endpoint (uses subprocess)."""
    print("\n" + "=" * 60)
    print("TEST: PLO5 Java (strength endpoint with 5-card hands)")
    print("=" * 60)
    
    hands = generate_hands(5, 4)
    print(f"Hands: {hands}")
    
    payload = {
        "cards": hands,
        "board": ""
    }
    
    t1 = time.time()
    r = requests.post(f"{BASE_URL}/strength", json=payload)
    elapsed = time.time() - t1
    
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Result: {data}")
    else:
        print(f"Error: {r.text[:200]}")
    print(f"Time: {elapsed:.3f}s")
    return r.status_code == 200, elapsed

def test_multiple_plo6_requests():
    """Test multiple PLO6 requests to see warm pool benefit."""
    print("\n" + "=" * 60)
    print("TEST: Multiple PLO6 Requests (showing warm pool benefit)")
    print("=" * 60)
    
    times = []
    for i in range(3):
        hands = generate_hands(6, 4)
        payload = {"cards": hands, "trials": 10000, "processes": 4}
        
        t1 = time.time()
        r = requests.post(f"{BASE_URL}/plo6pythonindividualandpairwise", json=payload)
        elapsed = time.time() - t1
        times.append(elapsed)
        
        status = "OK" if r.status_code == 200 else "FAIL"
        print(f"Request {i+1}: {elapsed:.2f}s [{status}]")
    
    print(f"\nFirst request:  {times[0]:.2f}s")
    print(f"Avg subsequent: {sum(times[1:])/len(times[1:]):.2f}s")
    return times

if __name__ == "__main__":
    print("=" * 60)
    print("COMPREHENSIVE PLO ENDPOINT TEST")
    print(f"Server: {BASE_URL}")
    print("=" * 60)
    
    results = {}
    
    # Health check
    results['health'] = test_health()
    
    # Wait a moment for background warmup
    print("\n[Waiting 3s for background warmup...]")
    time.sleep(3)
    
    # PLO6 Python tests
    success, elapsed = test_plo6_python()
    results['plo6_python'] = (success, elapsed)
    
    # PLO5 Python tests  
    success, elapsed = test_plo5_python()
    results['plo5_python'] = (success, elapsed)
    
    # PLO4 Java tests (may fail without license)
    success, elapsed = test_plo4_java()
    results['plo4_java'] = (success, elapsed)
    
    # PLO5 Java tests (may fail without license)
    success, elapsed = test_plo5_java()
    results['plo5_java'] = (success, elapsed)
    
    # Multiple requests test
    times = test_multiple_plo6_requests()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Health Check:      {'PASS' if results['health'] else 'FAIL'}")
    print(f"PLO6 Python:       {'PASS' if results['plo6_python'][0] else 'FAIL'} ({results['plo6_python'][1]:.2f}s)")
    print(f"PLO5 Python:       {'PASS' if results['plo5_python'][0] else 'FAIL'} ({results['plo5_python'][1]:.2f}s)")
    print(f"PLO4 Java:         {'PASS' if results['plo4_java'][0] else 'FAIL (expected - needs license)'}")
    print(f"PLO5 Java:         {'PASS' if results['plo5_java'][0] else 'FAIL (expected - needs license)'}")
    print(f"\nMultiple PLO6 requests:")
    print(f"  First:  {times[0]:.2f}s")
    print(f"  Second: {times[1]:.2f}s")
    print(f"  Third:  {times[2]:.2f}s")
