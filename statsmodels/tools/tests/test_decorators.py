# -*- coding: utf-8 -*-

import pytest
from numpy.testing import assert_equal

from statsmodels.tools.decorators import (
    cache_readonly, CacheWriteWarning)


def test_cache_readonly():

    class Example(object):
        def __init__(self):
            self._cache = {}
            self.a = 0

        @cache_readonly
        def b(self):
            return 1

        @cache_readonly
        def e(self):
            return 4

        @cache_readonly
        def f(self):
            return self.e + 1

    ex = Example()

    # Try accessing/setting a readonly attribute
    assert_equal(ex.__dict__, dict(a=0, _cache={}))

    b = ex.b
    assert_equal(b, 1)
    assert_equal(ex.__dict__, dict(a=0, _cache=dict(b=1,)))
    # assert_equal(ex.__dict__, dict(a=0, b=1, _cache=dict(b=1)))

    with pytest.warns(CacheWriteWarning):
        ex.b = -1

    assert_equal(ex._cache, dict(b=1,))
