#!/usr/bin/env python
"""
Simple test script to verify _to_pandas coverage.
"""
import sys
sys.path.insert(0, '/c/Users/Caio e Rafah/workspace/github-issues')

import numpy as np
import pandas as pd
import polars as pl
from statsmodels.tools.data import _to_pandas

def test_to_pandas():
    """Test _to_pandas with all input types."""
    tests_passed = 0
    tests_total = 0

    print("\n=== Testing _to_pandas() ===\n")

    # Test 1: None
    tests_total += 1
    result = _to_pandas(None)
    if result is None:
        print("[PASS] Test 1: None passthrough")
        tests_passed += 1
    else:
        print("[FAIL] Test 1: None passthrough")

    # Test 2: numpy array
    tests_total += 1
    arr = np.array([1, 2, 3])
    result = _to_pandas(arr)
    if result is arr:
        print("[PASS] Test 2: numpy array passthrough - PASS")
        tests_passed += 1
    else:
        print("[FAIL] Test 2: numpy array passthrough - FAIL")

    # Test 3: pandas Series
    tests_total += 1
    series = pd.Series([1, 2, 3])
    result = _to_pandas(series)
    if result is series:
        print("[PASS] Test 3: pandas Series passthrough - PASS")
        tests_passed += 1
    else:
        print("[FAIL] Test 3: pandas Series passthrough - FAIL")

    # Test 4: pandas DataFrame
    tests_total += 1
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = _to_pandas(df)
    if result is df:
        print("[PASS] Test 4: pandas DataFrame passthrough - PASS")
        tests_passed += 1
    else:
        print("[FAIL] Test 4: pandas DataFrame passthrough - FAIL")

    # Test 5: list
    tests_total += 1
    lst = [1, 2, 3]
    result = _to_pandas(lst)
    if result is lst:
        print("[PASS] Test 5: list passthrough - PASS")
        tests_passed += 1
    else:
        print("[FAIL] Test 5: list passthrough - FAIL")

    # Test 6: tuple
    tests_total += 1
    tpl = (1, 2, 3)
    result = _to_pandas(tpl)
    if result is tpl:
        print("[PASS] Test 6: tuple passthrough - PASS")
        tests_passed += 1
    else:
        print("[FAIL] Test 6: tuple passthrough - FAIL")

    # Test 7: Polars Series
    tests_total += 1
    polars_series = pl.Series("test", [1, 2, 3])
    result = _to_pandas(polars_series)
    if isinstance(result, pd.Series):
        print("[PASS] Test 7: Polars Series conversion - PASS")
        tests_passed += 1
    else:
        print("[FAIL] Test 7: Polars Series conversion - FAIL")

    # Test 8: Polars DataFrame
    tests_total += 1
    polars_df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = _to_pandas(polars_df)
    if isinstance(result, pd.DataFrame):
        print("[PASS] Test 8: Polars DataFrame conversion - PASS")
        tests_passed += 1
    else:
        print("[FAIL] Test 8: Polars DataFrame conversion - FAIL")

    # Test 9: dict (non-convertible)
    tests_total += 1
    d = {"key": "value"}
    result = _to_pandas(d)
    if result is d:
        print("[PASS] Test 9: dict passthrough - PASS")
        tests_passed += 1
    else:
        print("[FAIL] Test 9: dict passthrough - FAIL")

    return tests_passed, tests_total

if __name__ == "__main__":
    total_passed, total_tests = test_to_pandas()

    print("\n" + "="*50)
    print(f"SUMMARY: {total_passed}/{total_tests} tests passed")
    print("="*50)

    if total_passed == total_tests:
        print("[PASS] All tests passed! Coverage should be 100%")
        sys.exit(0)
    else:
        print(f"[FAIL] {total_tests - total_passed} test(s) failed!")
        sys.exit(1)
