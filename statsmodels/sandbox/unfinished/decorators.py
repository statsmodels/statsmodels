if __name__ == "__main__":
    # Tests resettable_cache --------------------------------------------
    reset = dict(a=('b',), b=('c',))
    cache = resettable_cache(a=0, b=1, c=2, reset=reset)
    assert_equal(cache, dict(a=0, b=1, c=2))
    #
    print("Try resetting a")
    cache['a'] = 1
    assert_equal(cache, dict(a=1, b=None, c=None))
    cache['c'] = 2
    assert_equal(cache, dict(a=1, b=None, c=2))
    cache['b'] = 0
    assert_equal(cache, dict(a=1, b=0, c=None))
    #
    print("Try deleting b")
    del(cache['a'])
    assert_equal(cache, {})
    # --------------------------------------------------------------------

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
    print("(attrs  : %s)" % str(ex.__dict__))
    print("(cached : %s)" % str(ex._cache))
    print("Try a   :", ex.a)
    print("Try accessing/setting a readonly attribute")
    assert_equal(ex.__dict__, dict(a=0, _cache={}))
    print("Try b #1:", ex.b)
    b = ex.b
    assert_equal(b, 1)
    assert_equal(ex.__dict__, dict(a=0, _cache=dict(b=1,)))
    # assert_equal(ex.__dict__, dict(a=0, b=1, _cache=dict(b=1)))
    ex.b = -1
    print("Try dict", ex.__dict__)
    assert_equal(ex._cache, dict(b=1,))
    #
    print("Try accessing/resetting a cachewritable attribute")
    c = ex.c
    assert_equal(c, 2)
    assert_equal(ex._cache, dict(b=1, c=2))
    d = ex.d
    assert_equal(d, 3)
    assert_equal(ex._cache, dict(b=1, c=2, d=3))
    ex.c = 0
    assert_equal(ex._cache, dict(b=1, c=0, d=None, e=None, f=None))
    d = ex.d
    assert_equal(ex._cache, dict(b=1, c=0, d=1, e=None, f=None))
    ex.d = 5
    assert_equal(ex._cache, dict(b=1, c=0, d=5, e=None, f=None))
