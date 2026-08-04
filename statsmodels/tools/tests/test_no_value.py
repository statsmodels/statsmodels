"""
Test functions for tools._no_value
"""
import pickle

import pytest

from statsmodels.tools._no_value import _NoValue, _NoValueType


def test_singleton():
    assert _NoValueType() is _NoValue
    assert _NoValueType() is _NoValueType()


def test_distinct_from_none():
    # the reason the sentinel exists: it must never be mistaken for a
    # legitimate attribute value such as None
    assert _NoValue is not None
    assert _NoValue != None  # noqa: E711


def test_repr():
    assert repr(_NoValue) == "<no value>"


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_pickle_preserves_identity(protocol):
    # results objects holding _NoValue are pickled by save/load, so
    # `is` checks must hold after a round-trip under every protocol
    assert pickle.loads(pickle.dumps(_NoValue, protocol)) is _NoValue
