from collections import OrderedDict
from decimal import Decimal
from enum import IntEnum
from fractions import Fraction
from types import MappingProxyType
import warnings

import numpy as np
import pandas as pd
import pytest

from statsmodels.tools.validation import (
    PandasWrapper,
    array_like,
    bool_like,
    dict_like,
    float_like,
    int_like,
    is_float_like,
    is_int_like,
    required_int_like,
    string_like,
)
from statsmodels.tools.validation.validation import _right_squeeze


@pytest.fixture(params=[True, False])
def use_pandas(request):
    return request.param


def gen_data(dim, use_pandas):
    if dim == 1:
        out = np.empty(
            10,
        )
        if use_pandas:
            out = pd.Series(out)
    elif dim == 2:
        # zeros, since uninitialized memory can overflow in dot
        out = np.zeros((20, 10))
        if use_pandas:
            out = pd.DataFrame(out)
    else:
        out = np.empty(np.arange(5, 5 + dim))

    return out


class TestArrayLike:
    def test_1d(self, use_pandas):
        data = gen_data(1, use_pandas)
        a = array_like(data, "a")
        assert a.ndim == 1
        assert a.shape == (10,)
        assert type(a) is np.ndarray

        a = array_like(data, "a", ndim=1)
        assert a.ndim == 1
        a = array_like(data, "a", shape=(10,))
        assert a.shape == (10,)
        a = array_like(data, "a", ndim=1, shape=(None,))
        assert a.ndim == 1
        a = array_like(data, "a", ndim=2, shape=(10, 1))
        assert a.ndim == 2
        assert a.shape == (10, 1)

        with pytest.raises(ValueError, match="a is required to have shape"):
            array_like(data, "a", shape=(5,))

    def test_2d(self, use_pandas):
        data = gen_data(2, use_pandas)
        a = array_like(data, "a", ndim=2)
        assert a.ndim == 2
        assert a.shape == (20, 10)
        assert type(a) is np.ndarray

        a = array_like(data, "a", ndim=2)
        assert a.ndim == 2
        a = array_like(data, "a", ndim=2, shape=(20, None))
        assert a.shape == (20, 10)
        with pytest.raises(
            ValueError,
            match="a must have 1 dimension, but has 2",
        ):
            array_like(data, "a", ndim=2, shape=(20,))
        assert a.shape == (20, 10)
        a = array_like(data, "a", ndim=2, shape=(None, 10))
        assert a.shape == (20, 10)

        a = array_like(data, "a", ndim=2, shape=(None, None))
        assert a.ndim == 2
        a = array_like(data, "a", ndim=3)
        assert a.ndim == 3
        assert a.shape == (20, 10, 1)

        with pytest.raises(ValueError, match="a is required to have shape"):
            array_like(data, "a", ndim=2, shape=(10, 10))
        with pytest.raises(ValueError, match="a is required to have shape"):
            array_like(data, "a", ndim=2, shape=(20, 20))
        with pytest.raises(ValueError, match="a is required to have shape"):
            array_like(data, "a", ndim=2, shape=(None, 20))
        match = "a is required to have ndim 1 but has ndim 2"
        with pytest.raises(ValueError, match=match):
            array_like(data, "a", ndim=1)
        match = "a must have ndim <= 1"
        with pytest.raises(ValueError, match=match):
            array_like(data, "a", maxdim=1)

    def test_3d(self):
        data = gen_data(3, False)
        a = array_like(data, "a", ndim=3)
        assert a.shape == (5, 6, 7)
        assert a.ndim == 3
        assert type(a) is np.ndarray

        a = array_like(data, "a", ndim=3, shape=(5, None, 7))
        assert a.shape == (5, 6, 7)
        a = array_like(data, "a", ndim=3, shape=(None, None, 7))
        assert a.shape == (5, 6, 7)
        a = array_like(data, "a", ndim=5)
        assert a.shape == (5, 6, 7, 1, 1)
        with pytest.raises(
            ValueError,
            match="a must have 1 dimension, but has 3",
        ):
            array_like(data, "a", ndim=3, shape=(10,))
        with pytest.raises(ValueError, match="a is required to have shape"):
            array_like(data, "a", ndim=3, shape=(None, None, 5))
        match = "a is required to have ndim 2 but has ndim 3"
        with pytest.raises(ValueError, match=match):
            array_like(data, "a", ndim=2)
        match = "a must have ndim <= 1"
        with pytest.raises(ValueError, match=match):
            array_like(data, "a", maxdim=1)
        match = "a must have ndim <= 2"
        with pytest.raises(ValueError, match=match):
            array_like(data, "a", maxdim=2)

    def test_right_squeeze_and_pad(self):
        data = np.empty((2, 1, 2))
        a = array_like(data, "a", ndim=3)
        assert a.shape == (2, 1, 2)
        data = np.empty(2)
        a = array_like(data, "a", ndim=3)
        assert a.shape == (2, 1, 1)
        data = np.empty((2, 1))
        a = array_like(data, "a", ndim=3)
        assert a.shape == (2, 1, 1)

        data = np.empty((2, 1, 1, 1))
        a = array_like(data, "a", ndim=3)
        assert a.shape == (2, 1, 1)

        data = np.empty((2, 1, 1, 2, 1, 1))
        with pytest.raises(ValueError):
            array_like(data, "a", ndim=3)

    def test_contiguous(self):
        x = np.arange(10)
        y = x[::2]
        a = array_like(y, "a", contiguous=True)
        assert not y.flags["C_CONTIGUOUS"]
        assert a.flags["C_CONTIGUOUS"]

    def test_dtype(self):
        x = np.arange(10)
        a = array_like(x, "a", dtype=np.float32)
        assert a.dtype == np.float32

        a = array_like(x, "a", dtype=np.uint8)
        assert a.dtype == np.uint8

    @pytest.mark.xfail(reason="Failing for now")
    def test_dot(self, use_pandas):
        data = gen_data(2, use_pandas)
        a = array_like(data, "a")
        assert not isinstance(a.T.dot(data), array_like)
        assert not isinstance(a.T.dot(a), array_like)

    def test_slice(self, use_pandas):
        data = gen_data(2, use_pandas)
        a = array_like(data, "a", ndim=2)
        assert type(a[1:]) is np.ndarray

    def test_none(self):
        assert array_like(None, "a", optional=True) is None
        # None is not array_like, and must not be converted to nan
        with pytest.raises(TypeError, match="a must be array_like, not None"):
            array_like(None, "a")


def test_right_squeeze():
    x = np.empty((10, 1, 10))
    y = _right_squeeze(x)
    assert y.shape == (10, 1, 10)

    x = np.empty((10, 10, 1))
    y = _right_squeeze(x)
    assert y.shape == (10, 10)

    x = np.empty((10, 10, 1, 1, 1, 1, 1))
    y = _right_squeeze(x)
    assert y.shape == (10, 10)

    x = np.empty((10, 1, 10, 1, 1, 1, 1, 1))
    y = _right_squeeze(x)
    assert y.shape == (10, 1, 10)


def test_wrap_pandas(use_pandas):
    a = gen_data(1, use_pandas)
    b = gen_data(1, False)

    wrapped = PandasWrapper(a).wrap(b)
    expected_type = pd.Series if use_pandas else np.ndarray
    assert isinstance(wrapped, expected_type)
    assert not use_pandas or wrapped.name is None

    wrapped = PandasWrapper(a).wrap(b, columns="name")
    assert isinstance(wrapped, expected_type)
    assert not use_pandas or wrapped.name == "name"

    wrapped = PandasWrapper(a).wrap(b, columns=["name"])
    assert isinstance(wrapped, expected_type)
    assert not use_pandas or wrapped.name == "name"

    expected_type = pd.DataFrame if use_pandas else np.ndarray
    wrapped = PandasWrapper(a).wrap(b[:, None])
    assert isinstance(wrapped, expected_type)
    assert not use_pandas or wrapped.columns[0] == 0

    wrapped = PandasWrapper(a).wrap(b[:, None], columns=["name"])
    assert isinstance(wrapped, expected_type)
    assert not use_pandas or wrapped.columns == ["name"]

    if use_pandas:
        match = "Can only wrap 1 or 2-d array_like"
        with pytest.raises(ValueError, match=match):
            PandasWrapper(a).wrap(b[:, None, None])

        match = "obj must have the same number of elements in axis 0 as"
        with pytest.raises(ValueError, match=match):
            PandasWrapper(a).wrap(b[: b.shape[0] // 2])


def test_wrap_pandas_append():
    a = gen_data(1, True)
    a.name = "apple"
    b = gen_data(1, False)
    wrapped = PandasWrapper(a).wrap(b, append="appended")
    expected = "apple_appended"
    assert wrapped.name == expected

    a = gen_data(2, True)
    a.columns = ["apple_" + str(i) for i in range(a.shape[1])]
    b = gen_data(2, False)
    wrapped = PandasWrapper(a).wrap(b, append="appended")
    expected = [c + "_appended" for c in a.columns]
    assert list(wrapped.columns) == expected


def test_wrap_pandas_append_non_string():
    # GH 6826
    a = gen_data(1, True)
    a.name = 7
    b = gen_data(1, False)
    wrapped = PandasWrapper(a).wrap(b, append="appended")
    expected = "7_appended"
    assert wrapped.name == expected

    a = gen_data(2, True)
    a.columns = list(range(a.shape[1]))
    b = gen_data(2, False)
    wrapped = PandasWrapper(a).wrap(b, append="appended")
    expected = [f"{c}_appended" for c in a.columns]
    assert list(wrapped.columns) == expected


class CustomDict(dict):
    pass


@pytest.fixture(params=(dict, OrderedDict, CustomDict, None))
def dict_type(request):
    return request.param


def test_optional_dict_like(dict_type):
    val = dict_type() if dict_type is not None else dict_type
    out = dict_like(val, "value", optional=True)
    assert isinstance(out, type(val))


def test_optional_dict_like_error():
    # strict only accepts a dict, so the message must not offer a Mapping
    match = r"value must be a dict$"
    for value in ([], {"a"}, "a"):
        with pytest.raises(TypeError, match=match):
            dict_like(value, "value", optional=True)
    # without strict any Mapping is accepted, which the message reports
    match = r"value must be a dict or dict_like \(i.e., a Mapping\)"
    for value in ([], {"a"}, "a"):
        with pytest.raises(TypeError, match=match):
            dict_like(value, "value", optional=True, strict=False)


def test_dict_like_strict_mapping():
    proxy = MappingProxyType({"a": 1})
    assert dict_like(proxy, "value", strict=False) is proxy
    with pytest.raises(TypeError, match=r"value must be a dict$"):
        dict_like(proxy, "value")


def test_string():
    out = string_like("apple", "value")
    assert out == "apple"

    out = string_like("apple", "value", options=("apple", "banana", "cherry"))
    assert out == "apple"

    with pytest.raises(TypeError, match="value must be a string"):
        string_like(1, "value")
    with pytest.raises(TypeError, match="value must be a string"):
        string_like(b"4", "value")
    with pytest.raises(
        ValueError,
        match="value must be one of: 'apple', 'banana', 'cherry'",
    ):
        string_like("date", "value", options=("apple", "banana", "cherry"))
    # None is only allowed when optional is True
    with pytest.raises(TypeError, match="value must be a string"):
        string_like(None, "value")
    with pytest.raises(TypeError, match="value must be a string"):
        string_like(None, "value", options=("apple", "banana", "cherry"))


def test_optional_string():
    out = string_like("apple", "value")
    assert out == "apple"

    out = string_like("apple", "value", options=("apple", "banana", "cherry"))
    assert out == "apple"

    out = string_like(None, "value", optional=True)
    assert out is None

    out = string_like(
        None, "value", optional=True, options=("apple", "banana", "cherry")
    )
    assert out is None

    with pytest.raises(TypeError, match="value must be a string"):
        string_like(1, "value", optional=True)
    with pytest.raises(TypeError, match="value must be a string"):
        string_like(b"4", "value", optional=True)


def test_string_deprecated_alias():
    # a value matching a key of `deprecated` is accepted and normalized to
    # its replacement, but warns
    with pytest.warns(
        FutureWarning,
        match=r"value='old' is a deprecated alias for value='new'",
    ):
        out = string_like(
            "old", "value", options=("new", "other"), deprecated={"old": "new"}
        )
    assert out == "new"

    # documented options are unaffected and never warn
    out = string_like(
        "new", "value", options=("new", "other"), deprecated={"old": "new"}
    )
    assert out == "new"

    # a value that is neither a documented option nor a deprecated alias
    # still raises, and the message lists both
    with pytest.raises(
        ValueError,
        match=r"value must be one of: 'new', 'other', 'old'",
    ):
        string_like(
            "bogus", "value", options=("new", "other"), deprecated={"old": "new"}
        )

    # lower= still applies before matching against options and deprecated
    with pytest.warns(FutureWarning, match="is a deprecated alias"):
        out = string_like(
            "OLD",
            "value",
            options=("new", "other"),
            lower=True,
            deprecated={"old": "new"},
        )
    assert out == "new"


def test_string_deprecated_alias_removed_after():
    # without removed_after, the message only promises "a future version"
    with pytest.warns(
        FutureWarning,
        match=r"will be removed in a future version",
    ) as rec:
        string_like(
            "old", "value", options=("new",), deprecated={"old": "new"}
        )
    assert "statsmodels" not in str(rec[0].message)

    # removed_after names a concrete release in the same message
    with pytest.warns(
        FutureWarning,
        match=r"will be removed after statsmodels 0\.16 is released",
    ):
        out = string_like(
            "old",
            "value",
            options=("new",),
            deprecated={"old": "new"},
            removed_after="0.16",
        )
    assert out == "new"

    # removed_after is ignored (no warning at all) for values that are
    # not deprecated aliases, canonical or otherwise
    out = string_like(
        "new",
        "value",
        options=("new",),
        deprecated={"old": "new"},
        removed_after="0.16",
    )
    assert out == "new"


class Color(IntEnum):
    RED = 3


class IndexOnly:
    """An integer scalar of another library: supports only __index__."""

    def __index__(self):
        return 3


# (id, value, its int value) for values that are integers even if strict
STRICT_INTEGERS = [
    ("int", 3, 3),
    ("negative int", -4, -4),
    ("big int", 2**70, 2**70),
    ("IntEnum", Color.RED, 3),
    ("int64", np.int64(3), 3),
    ("int8", np.int8(-3), -3),
    ("uint64 max", np.uint64(2**64 - 1), 2**64 - 1),
    ("0-d int array", np.array(3), 3),
    ("__index__ only", IndexOnly(), 3),
]
# integers only if not strict
INTEGRAL_FLOATS = [
    ("float", 3.0, 3),
    ("negative zero", -0.0, 0),
    ("float64", np.float64(-2.0), -2),
    ("float32", np.float32(3.0), 3),
    ("float16", np.float16(4.0), 4),
    ("0-d float array", np.array(5.0), 5),
]
# never integers, not even under the old rules
NOT_INTEGERS = [
    ("bool", True),
    ("np.bool_", np.True_),
    ("0-d bool array", np.array(True)),
    ("bool Series", pd.Series([True])),
    ("bool array", np.array([False])),
    ("fraction", 3.2),
    ("float32 fraction", np.float32(3.2)),
    ("nearly integral", 1000 ** (1 / 3)),
    ("nan", np.nan),
    ("inf", np.inf),
    ("complex", 3 + 0j),
    ("complex128", np.complex128(3)),
    ("str", "3"),
    ("bytes", b"3"),
    ("timedelta64", np.timedelta64(2, "s")),
    ("0-d timedelta64 array", np.array(np.timedelta64(2, "s"))),
    ("datetime64", np.datetime64("2000-01-01")),
    ("Timedelta", pd.Timedelta(3)),
    ("Timestamp", pd.Timestamp("2000-01-01")),
    ("list", [3]),
    ("tuple", (3,)),
    ("2 element array", np.array([3, 4])),
    ("2 element Series", pd.Series([3, 4])),
    ("NA", pd.NA),
    ("object", object()),
]
# rejected by is_int_like, accepted by int_like with a FutureWarning
DEPRECATED_INTEGERS = [
    ("1 element array", np.array([4]), 4, False),
    ("1 element float array", np.array([4.0]), 4, False),
    ("1x1 array", np.array([[3.0]]), 3, False),
    ("1 element Series", pd.Series([3]), 3, True),
    ("1x1 DataFrame", pd.DataFrame([[3]]), 3, True),
    ("Decimal", Decimal(3), 3, False),
    ("Fraction", Fraction(6, 2), 3, False),
]


def _ids(cases):
    return [case[0] for case in cases]


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize(
    "name, value, expected", STRICT_INTEGERS, ids=_ids(STRICT_INTEGERS)
)
def test_is_int_like_integers(name, value, expected, strict):
    assert is_int_like(value, strict=strict) is True
    converted = int_like(value, "value", strict=strict)
    assert type(converted) is int
    assert converted == expected


@pytest.mark.parametrize(
    "name, value, expected", INTEGRAL_FLOATS, ids=_ids(INTEGRAL_FLOATS)
)
def test_is_int_like_integral_floats(name, value, expected):
    assert is_int_like(value) is True
    converted = int_like(value, "value")
    assert type(converted) is int
    assert converted == expected
    assert is_int_like(value, strict=True) is False
    with pytest.raises(TypeError, match=r"value must be .*not a bool or a float"):
        int_like(value, "value", strict=True)


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("name, value", NOT_INTEGERS, ids=_ids(NOT_INTEGERS))
def test_is_int_like_not_integers(name, value, strict):
    assert is_int_like(value, strict=strict) is False
    with pytest.raises(TypeError, match="value must be integer_like"):
        int_like(value, "value", strict=strict)
    with pytest.raises(TypeError, match=r"value must be integer_like.* or None"):
        int_like(value, "value", optional=True, strict=strict)


@pytest.mark.parametrize(
    "name, value, expected, strict_too",
    DEPRECATED_INTEGERS,
    ids=_ids(DEPRECATED_INTEGERS),
)
def test_int_like_deprecated(name, value, expected, strict_too):
    assert is_int_like(value) is False
    assert is_int_like(value, strict=True) is False
    match = (
        rf"Passing nlags of type {type(value).__name__} is deprecated and will "
        rf"raise a TypeError in statsmodels 0\.17\. Pass nlags={expected} as"
    )
    with pytest.warns(FutureWarning, match=match):
        converted = int_like(value, "nlags")
    assert type(converted) is int
    assert converted == expected
    if strict_too:
        with pytest.warns(FutureWarning, match=match):
            assert int_like(value, "nlags", strict=True) == expected
    else:
        with pytest.raises(TypeError, match="nlags must be integer_like"):
            int_like(value, "nlags", strict=True)


def test_int_like_none():
    assert int_like(None, "value", optional=True) is None
    assert int_like(None, "value", optional=True, strict=True) is None
    with pytest.raises(TypeError, match="value must be integer_like"):
        int_like(None, "value")
    assert is_int_like(None) is False


def test_int_like_error_message():
    match = r"k must be integer_like \(an int, a NumPy integer or a float with"
    with pytest.raises(TypeError, match=match):
        int_like(2.5, "k")
    match = r"k must be integer_like \(an int or a NumPy integer, but not a bool"
    with pytest.raises(TypeError, match=match):
        int_like(2.5, "k", strict=True)


def test_required_int_like():
    assert required_int_like(np.int64(3), "value") == 3
    assert required_int_like(3.0, "value") == 3
    with pytest.raises(TypeError, match="value must be integer_like"):
        required_int_like(3.0, "value", strict=True)
    with pytest.raises(TypeError, match="value must be integer_like"):
        required_int_like(None, "value")
    with pytest.raises(TypeError, match="value must be integer_like"):
        required_int_like(np.True_, "value")


@pytest.fixture(params=[True, False, 1, 1.2, "a", ""])
def boolean(request):
    return request.param


def test_bool_like(boolean):
    assert isinstance(bool_like(boolean, "boolean"), bool)
    assert bool_like(None, "boolean", optional=True) is None
    if isinstance(boolean, bool):
        assert isinstance(bool_like(boolean, "boolean", strict=True), bool)
    else:
        with pytest.raises(TypeError):
            bool_like(boolean, "boolean", strict=True)


def test_not_bool_like():
    with pytest.raises(TypeError):
        bool_like(np.array([True, True]), boolean)


def test_array_like_mindim():
    x = 1
    arr = array_like(x, "x", mindim=1)
    assert arr.shape == (1,)
    arr = array_like(x, "x", mindim=2)
    assert arr.shape == (1, 1)
    arr = array_like(x, "x", mindim=3)
    assert arr.shape == (1, 1, 1)
    y = [1, 2]
    arr = array_like(y, "y", mindim=2)
    assert arr.shape == (2, 1)


# (id, value, its float value) for values that are real numbers even if strict
STRICT_REALS = [
    ("float", 1.5, 1.5),
    ("negative float", -2.25, -2.25),
    ("nan", np.nan, np.nan),
    ("inf", np.inf, np.inf),
    ("float64", np.float64(0.25), 0.25),
    ("float32", np.float32(0.5), 0.5),
    ("float16", np.float16(-0.5), -0.5),
    ("int", 3, 3.0),
    ("int64", np.int64(-3), -3.0),
    ("IntEnum", Color.RED, 3.0),
    ("0-d float array", np.array(0.75), 0.75),
    ("0-d int array", np.array(2), 2.0),
    ("__index__ only", IndexOnly(), 3.0),
]
# real numbers only if not strict
NONSTRICT_REALS = [
    ("Fraction", Fraction(1, 4), 0.25),
]
# never real numbers, not even under the old rules
NOT_REALS = [
    ("bool", True),
    ("np.bool_", np.True_),
    ("0-d bool array", np.array(False)),
    ("bool Series", pd.Series([True])),
    ("complex", 1.2 + 1j),
    ("complex128", np.complex128(1 + 2j)),
    ("complex Series", pd.Series([1 + 2j])),
    ("str", "3.2"),
    ("timedelta64", np.timedelta64(2, "s")),
    ("datetime64", np.datetime64("2000-01-01")),
    ("Timedelta", pd.Timedelta(3)),
    ("Decimal", Decimal("0.5")),
    ("list", [0.5]),
    ("2 element array", np.empty(2)),
    ("2 element Series", pd.Series([0.5, 1.5])),
    ("NA", pd.NA),
    ("object", object()),
]
# rejected by is_float_like, accepted by float_like with a FutureWarning
DEPRECATED_REALS = [
    ("1 element array", np.array([1.2]), 1.2, False),
    ("1x1 int array", np.array([[3]]), 3.0, False),
    ("1 element Series", pd.Series([1.2]), 1.2, True),
    ("1x1 DataFrame", pd.DataFrame([[0.5]]), 0.5, True),
    ("complex, zero imaginary part", 1.2 + 0j, 1.2, False),
    ("complex128, zero imaginary part", np.complex128(2), 2.0, False),
    ("0-d complex array", np.array(0.5 + 0j), 0.5, False),
]


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize(
    "name, value, expected", STRICT_REALS, ids=_ids(STRICT_REALS)
)
def test_is_float_like_reals(name, value, expected, strict):
    assert is_float_like(value, strict=strict) is True
    converted = float_like(value, "value", strict=strict)
    assert type(converted) is float
    np.testing.assert_equal(converted, expected)


@pytest.mark.parametrize(
    "name, value, expected", NONSTRICT_REALS, ids=_ids(NONSTRICT_REALS)
)
def test_is_float_like_nonstrict_reals(name, value, expected):
    assert is_float_like(value) is True
    converted = float_like(value, "value")
    assert type(converted) is float
    assert converted == expected
    assert is_float_like(value, strict=True) is False
    with pytest.raises(TypeError, match="value must be float_like"):
        float_like(value, "value", strict=True)


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("name, value", NOT_REALS, ids=_ids(NOT_REALS))
def test_is_float_like_not_reals(name, value, strict):
    assert is_float_like(value, strict=strict) is False
    with pytest.raises(TypeError, match="value must be float_like"):
        float_like(value, "value", strict=strict)
    with pytest.raises(TypeError, match=r"value must be float_like.* or None"):
        float_like(value, "value", optional=True, strict=strict)


@pytest.mark.parametrize(
    "name, value, expected, strict_too",
    DEPRECATED_REALS,
    ids=_ids(DEPRECATED_REALS),
)
def test_float_like_deprecated(name, value, expected, strict_too):
    assert is_float_like(value) is False
    assert is_float_like(value, strict=True) is False
    match = (
        rf"Passing alpha of type {type(value).__name__} is deprecated and will "
        rf"raise a TypeError in statsmodels 0\.17\. Pass alpha={expected} as"
    )
    with pytest.warns(FutureWarning, match=match):
        converted = float_like(value, "alpha")
    assert type(converted) is float
    assert converted == expected
    if strict_too:
        with pytest.warns(FutureWarning, match=match):
            assert float_like(value, "alpha", strict=True) == expected
    else:
        with pytest.raises(TypeError, match="alpha must be float_like"):
            float_like(value, "alpha", strict=True)


def test_float_like_none():
    assert float_like(None, "value", optional=True) is None
    assert float_like(None, "value", optional=True, strict=True) is None
    with pytest.raises(TypeError, match="value must be float_like"):
        float_like(None, "value")
    assert is_float_like(None) is False


def test_float_like_error_message():
    match = r"a must be float_like \(a real number such as a float, an int or a"
    with pytest.raises(TypeError, match=match):
        float_like("0.5", "a")
    match = r"a must be float_like \(a float, an int or a NumPy real number, but"
    with pytest.raises(TypeError, match=match):
        float_like("0.5", "a", strict=True)


class _Subclass(np.ndarray):
    pass


def _matrix(data):
    # np.matrix warns that it is not recommended, which the test suite raises
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        return np.asmatrix(data)


@pytest.mark.parametrize(
    "make",
    [
        lambda: _matrix([[1.0, 2.0], [3.0, 4.0]]),
        lambda: np.ma.masked_array([[1.0, 2.0], [3.0, 4.0]], mask=[[0, 1], [0, 0]]),
        lambda: np.array([[1.0, 2.0], [3.0, 4.0]]).view(_Subclass),
    ],
    ids=["matrix", "masked", "subclass"],
)
@pytest.mark.parametrize("writeable", [True, False])
def test_array_like_returns_base_ndarray(make, writeable):
    # array_like returned subclasses unchanged, unlike np.asarray: a matrix
    # kept matrix semantics and a masked array kept its mask
    obj = make()
    arr = array_like(obj, "obj", ndim=2, writeable=writeable)
    assert type(arr) is np.ndarray
    np.testing.assert_array_equal(arr, np.asarray(obj))


def test_array_like_matrix_column_squeezed():
    # a matrix is always 2-d, so a column could not be squeezed to 1-d
    arr = array_like(_matrix([[1.0], [2.0], [3.0]]), "x", ndim=1)
    assert type(arr) is np.ndarray
    np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    "obj, shape, expected",
    [
        (3.0, (None, None), "x must have 2 dimensions, but has 0"),
        ([1.0, 2.0], (None, None), "x must have 2 dimensions, but has 1"),
        (np.ones((2, 2, 2)), (None, None), "x must have 2 dimensions, but has 3"),
        (np.ones((2, 2)), (None,), "x must have 1 dimension, but has 2"),
    ],
)
def test_array_like_shape_length_message(obj, shape, expected):
    # the message reports the dimensions of the input, before a scalar is
    # padded to 1-d by mindim
    with pytest.raises(ValueError, match=expected):
        array_like(obj, "x", shape=shape)
