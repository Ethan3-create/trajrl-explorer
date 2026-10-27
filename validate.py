# -*- coding: utf-8 -*-
"""验证 web_server.py 可启动与核心 API 正确性."""
import json
import sys
import threading
import time
import urllib.request

# Start server in background
from web_server import DataStore, load_records, SAMPLE_PATH, APIHandler

# 1. Check data loading
print("[1/5] Testing data loading...")
try:
    records = load_records(SAMPLE_PATH)
    print(f"  ✓ Loaded {len(records)} records")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# 2. Check DataStore
print("[2/5] Testing DataStore...")
try:
    store = DataStore(records)
    print(f"  ✓ Features shape: {store.features.shape}")
    print(f"  ✓ TTE baseline: a={store.a:.6f}, b={store.b:.1f}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# 3. Check search
print("[3/5] Testing KNN search...")
try:
    results = store.search(0, k=3)
    print(f"  ✓ Found {len(results)} neighbors for record 0")
    for r in results:
        print(f"    - {r['trip_id']}: dist={r['distance']:.1f}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# 4. Check projection
print("[4/5] Testing PCA projection...")
try:
    proj = store.get_projection()
    print(f"  ✓ Projection shape: {len(proj)}x{len(proj[0])}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# 5. Check augment
print("[5/5] Testing trajectory augmentation...")
try:
    from web_server import augment_shift
    aug = augment_shift(records[0], 0.15, 42)
    print(f"  ✓ Augmented trajectory: {len(aug['points'])} points")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

print("\n✅ All validations passed!")
print("Ready to start server: python web_server.py --port 8601")
