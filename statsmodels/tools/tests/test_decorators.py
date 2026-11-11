import ast
from pathlib import Path

from numpy.testing import assert_equal
import pytest

from statsmodels.tools import _decorators
from statsmodels.tools._decorators import cache_readonly, deprecated_alias


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
    dummy_set = dummy_factory(msg, remove_version, warning)
    with pytest.warns(warning) as w:
        dummy_set.y = 2
    assert dummy_set.x == 2

    assert warning.__class__ is w[0].category.__class__

    dummy_get = dummy_factory(msg, remove_version, warning)
    with pytest.warns(warning) as w:
        x = dummy_get.y
    assert x == 1

    assert warning.__class__ is w[0].category.__class__
    message = str(w[0].message)
    if not msg:
        if remove_version:
            assert "will be removed" in message
        else:
            assert "will be removed" not in message
    else:
        assert msg in message


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
