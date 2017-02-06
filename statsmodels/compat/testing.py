"""
Compatibility shims to facilitate the transition from nose to pytest.  Can be removed once pytest
is the only way to run tests.
"""
import functools

try:
    from unittest.case import SkipTest
except ImportError:
    try:
        from nose import SkipTest
    except ImportError:
        try:
            from unittest2 import SkipTest
        except ImportError:
            raise ImportError('Unable to locate SkipTest.  unittest, unittest2 or nose required.')

def skip(reason):
    """
    Unconditionally skip a test.
    """

    def decorator(test_item):
        if not isinstance(test_item, type):
            @functools.wraps(test_item)
            def skip_wrapper(*args, **kwargs):
                raise SkipTest(reason)

            test_item = skip_wrapper

        test_item.__unittest_skip__ = True
        test_item.__unittest_skip_why__ = reason
        return test_item

    return decorator


def _id(obj):
    return obj


def skipIf(condition, reason):
    """
    Skip a test if the condition is true.
    """
    if condition:
        return skip(reason)
    return _id

skipif = skipIf

__all__ = ['skip', 'skipif', 'SkipTest']