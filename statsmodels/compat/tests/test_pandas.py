from statsmodels.compat.pandas import is_float_index, is_int_index

import warnings

import numpy as np
import pytest

HAS_NUMERIC_INDEX = False
try:
    from pandas import NumericIndex

    HAS_NUMERIC_INDEX = True
except ImportError:
    pass

HAS_LEGACY_INDEX = False
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        from pandas import Float64Index, Int64Index, UInt64Index

    HAS_LEGACY_INDEX = True
except ImportError:
    pass


@pytest.mark.skipif(not HAS_NUMERIC_INDEX, reason="Requires NumericIndex")
@pytest.mark.parametrize("int_type", ["u", "i"])
@pytest.mark.parametrize("int_size", [1, 2, 4, 8])
def test_is_int_index(int_type, int_size):
    index = NumericIndex(np.arange(100), dtype=f"{int_type}{int_size}")
    assert is_int_index(index)
    assert not is_float_index(index)


@pytest.mark.skipif(not HAS_NUMERIC_INDEX, reason="Requires NumericIndex")
@pytest.mark.parametrize("float_size", [4, 8])
def test_is_float_index(float_size):
    index = NumericIndex(np.arange(100), dtype=f"f{float_size}")
    assert is_float_index(index)
    assert not is_int_index(index)


@pytest.mark.skipif(not HAS_LEGACY_INDEX, reason="Requires U/Int64Index")
def test_legacy_int_index():
    index = Int64Index(np.arange(100))
    assert is_int_index(index)
    assert not is_float_index(index)

    index = UInt64Index(np.arange(100))
    assert is_int_index(index)
    assert not is_float_index(index)


@pytest.mark.skipif(not HAS_LEGACY_INDEX, reason="Requires Float64Index")
def test_legacy_float_index():
    index = Float64Index(np.arange(100))
    assert not is_int_index(index)
    assert is_float_index(index)
