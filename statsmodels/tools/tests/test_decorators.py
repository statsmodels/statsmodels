import ast
from pathlib import Path
import re

from numpy.testing import assert_equal
import pytest

from statsmodels.tools import _decorators
from statsmodels.tools._decorators import (
    ResettableCache,
    cache_readonly,
    cache_writable,
    deprecated_alias,
)


def test_resettable_cache():
    # The items are also the attributes
    cache = ResettableCache({"a": 1}, b=2)
    assert cache.a == 1
    assert cache.b == 2
    cache.c = 3
    cache["d"] = 4
    assert cache.d == 4
    assert cache == {"a": 1, "b": 2, "c": 3, "d": 4}


def test_cache_readonly():

    class Example:
        def __init__(self):
            self._cache = {}
            self.a = 0

        @cache_readonly
        def b(self):
            return 1

    ex = Example()

    # Try accessing/setting a readonly attribute
    assert_equal(ex.__dict__, dict(a=0, _cache={}))

    b = ex.b
    assert_equal(b, 1)
    assert_equal(
        ex.__dict__,
        dict(
            a=0,
            _cache=dict(
                b=1,
            ),
        ),
    )
    # assert_equal(ex.__dict__, dict(a=0, b=1, _cache=dict(b=1)))

    with pytest.raises(AttributeError):
        ex.b = -1

    assert_equal(
        ex._cache,
        dict(
            b=1,
        ),
    )


def test_cache_readonly_class_access():
    calls = []

    class Example:
        @cache_readonly
        def b(self):
            """The b value."""
            calls.append(None)
            return 1

    # On the class, the descriptor is returned, with the docstring and name of
    # the method, so that the documentation tools find them
    assert Example.b is Example.__dict__["b"]
    assert Example.b.__doc__ == "The b value."
    assert Example.b.__name__ == "b"

    # An instance without a cache gets one on the first access
    ex = Example()
    assert not hasattr(ex, "_cache")
    assert ex.b == 1
    assert ex.b == 1
    assert ex._cache == {"b": 1}
    assert len(calls) == 1


@pytest.mark.parametrize("cachename", [None, "_other_cache"])
def test_cache_writable(cachename):
    calls = []

    class Example:
        @cache_writable(cachename=cachename)
        def c(self):
            """The c value."""
            calls.append(None)
            return 2

    store = "_cache" if cachename is None else cachename
    ex = Example()
    assert not hasattr(ex, store)
    # The cache is created on the first access and the value is computed once
    assert ex.c == 2
    assert ex.c == 2
    assert getattr(ex, store) == {"c": 2}
    assert len(calls) == 1
    # Unlike cache_readonly, the cached value can be replaced
    ex.c = 5
    assert ex.c == 5
    assert getattr(ex, store) == {"c": 5}
    assert len(calls) == 1

    # On the class, the undecorated method is returned
    method = Example.c
    assert method.__doc__ == "The c value."
    assert method(ex) == 2
    assert len(calls) == 2


def dummy_factory(msg, remove_version, warning):
    class Dummy:
        y = deprecated_alias(
            "y", "x", remove_version=remove_version, msg=msg, warning=warning
        )

        def __init__(self, y):
            self.x = y

    return Dummy(1)


@pytest.mark.parametrize("warning", [FutureWarning, UserWarning])
@pytest.mark.parametrize("remove_version", [None, "0.11"])
@pytest.mark.parametrize("msg", ["test message", None])
def test_deprecated_alias(msg, remove_version, warning):
    if msg is not None:
        expected = msg
    elif remove_version is not None:
        expected = "y is a deprecated alias for x, will be removed in version 0.11"
    else:
        expected = "y is a deprecated alias for x"
    match = f"^{re.escape(expected)}$"

    dummy_set = dummy_factory(msg, remove_version, warning)
    with pytest.warns(warning, match=match) as w:
        dummy_set.y = 2
    assert dummy_set.x == 2
    assert len(w) == 1
    assert w[0].category is warning
    # The warning points at the line that used the alias
    assert Path(w[0].filename).name == Path(__file__).name

    dummy_get = dummy_factory(msg, remove_version, warning)
    with pytest.warns(warning, match=match) as w:
        x = dummy_get.y
    assert x == 1
    assert len(w) == 1
    assert w[0].category is warning
    assert Path(w[0].filename).name == Path(__file__).name


def test_stub_declares_public_names():
    # The stub replaces the module for the type checkers, which report a name
    # that it does not declare as missing from the module
    stub = Path(_decorators.__file__).with_suffix(".pyi")
    if not stub.exists():
        pytest.skip("The stub is not installed")
    declared = set()
    for node in ast.parse(stub.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            declared.add(node.name)
        elif isinstance(node, ast.Assign):
            declared.update(t.id for t in node.targets if isinstance(t, ast.Name))
    assert set(_decorators.__all__) <= declared
