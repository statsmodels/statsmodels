# -*- coding: utf-8 -*-
from statsmodels.compat.python import lmap

from itertools import product

import pytest
from numpy.testing import assert_equal

from statsmodels.tools.decorators import (resettable_cache, cache_readonly,
                                          cache_writable, CacheWriteWarning,
                                          deprecated_alias)


def test_resettable_cache():
    # This test was taken from the old __main__ section of decorators.py

    reset = dict(a=('b',), b=('c',))
    cache = resettable_cache(a=0, b=1, c=2, reset=reset)
    assert_equal(cache, dict(a=0, b=1, c=2))

    # Try resetting a
    cache['a'] = 1
    assert_equal(cache, dict(a=1, b=None, c=None))
    cache['c'] = 2
    assert_equal(cache, dict(a=1, b=None, c=2))
    cache['b'] = 0
    assert_equal(cache, dict(a=1, b=0, c=None))

    # Try deleting b
    del cache['a']
    assert_equal(cache, {})


def test_cache_readonly():

    class Example(object):
        def __init__(self):
            self._cache = resettable_cache()
            self.a = 0

        @cache_readonly
        def b(self):
            return 1

        @cache_writable(resetlist='d')
        def c(self):
            return 2

        @cache_writable(resetlist=('e', 'f'))
        def d(self):
            return self.c + 1

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

    # Try accessing/resetting a cachewritable attribute
    c = ex.c
    assert_equal(c, 2)
    assert_equal(ex._cache, dict(b=1, c=2))
    d = ex.d
    assert_equal(d, 3)
    assert_equal(ex._cache, dict(b=1, c=2, d=3))
    ex.c = 0
    assert_equal(ex._cache, dict(b=1, c=0, d=None, e=None, f=None))
    ex.d
    assert_equal(ex._cache, dict(b=1, c=0, d=1, e=None, f=None))
    ex.d = 5
    assert_equal(ex._cache, dict(b=1, c=0, d=5, e=None, f=None))


def dummy_factory(msg, remove_version, warning):
    class Dummy(object):
        y = deprecated_alias('y', 'x',
                             remove_version=remove_version,
                             msg=msg,
                             warning=warning)
        def __init__(self, y):
            self.x = y

    return Dummy(1)


@pytest.mark.parametrize('warning', [FutureWarning, UserWarning])
@pytest.mark.parametrize('remove_version', [None, '0.11'])
@pytest.mark.parametrize('msg', ['test message', None])
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
            assert 'will be removed' in message
        else:
            assert 'will be removed' not in message
    else:
        assert msg in message
