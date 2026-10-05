from statsmodels.compat.pandas import (
    deprecate_kwarg,
    is_float_index,
    is_int_index,
)

import warnings

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("int_type", ["u", "i"])
@pytest.mark.parametrize("int_size", [1, 2, 4, 8])
def test_is_int_index(int_type, int_size):
    index = pd.Index(np.arange(100), dtype=f"{int_type}{int_size}")
    assert is_int_index(index)
    assert not is_float_index(index)


@pytest.mark.parametrize("float_size", [4, 8])
def test_is_float_index(float_size):
    index = pd.Index(np.arange(100.0), dtype=f"f{float_size}")
    assert is_float_index(index)
    assert not is_int_index(index)


# Tests for vendored deprecate_kwarg. Adapted directly from:
#     pandas/tests/util/test_deprecate_kwarg.py
# (relocated from test_docstring_helpers.py, which does not test this
# module: deprecate_kwarg lives in statsmodels.compat.pandas, not
# statsmodels.tools.docstring_helpers)


@deprecate_kwarg(old_arg_name="old", new_arg_name="new")
def _f1(new=False):
    return new


_f2_mappings = {"yes": True, "no": False}


@deprecate_kwarg(old_arg_name="old", new_arg_name="new", mapping=_f2_mappings)
def _f2(new=False):
    return new


def _f3_mapping(x):
    return x + 1


@deprecate_kwarg(old_arg_name="old", new_arg_name="new", mapping=_f3_mapping)
def _f3(new=0):
    return new


@deprecate_kwarg(old_arg_name="old", new_arg_name=None)
def _f4(old=True, unchanged=True):
    return old, unchanged


@pytest.mark.parametrize("key,warns", [("old", True), ("new", False)])
def test_deprecate_kwarg(key, warns):
    x = 78
    if warns:
        with pytest.warns(FutureWarning):
            assert _f1(**{key: x}) == x
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert _f1(**{key: x}) == x
        assert len(w) == 0


@pytest.mark.parametrize("key", list(_f2_mappings.keys()))
def test_dict_deprecate_kwarg(key):
    with pytest.warns(FutureWarning):
        assert _f2(old=key) == _f2_mappings[key]


@pytest.mark.parametrize("key", ["bogus", 12345, -1.23])
def test_missing_deprecate_kwarg(key):
    # keys not in mapping are forwarded unchanged
    with pytest.warns(FutureWarning):
        assert _f2(old=key) == key


@pytest.mark.parametrize("x", [1, -1.4, 0])
def test_callable_deprecate_kwarg(x):
    with pytest.warns(FutureWarning):
        assert _f3(old=x) == _f3_mapping(x)


def test_bad_deprecate_kwarg():
    msg = "mapping from old to new argument values must be dict or callable!"
    with pytest.raises(TypeError, match=msg):

        @deprecate_kwarg("old", "new", mapping=0)
        def f4(new=None):
            return new


@pytest.mark.parametrize("key", ["old", "unchanged"])
def test_deprecate_keyword(key):
    x = 9
    if key == "old":
        expected = (x, True)
        with pytest.warns(FutureWarning):
            assert _f4(**{key: x}) == expected
    else:
        expected = (True, x)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert _f4(**{key: x}) == expected
        assert len(w) == 0
