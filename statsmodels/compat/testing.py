"""
Compatibility shims to facilitate the transition from nose to pytest.  Can be removed once pytest
is the only way to run tests.
"""
from statsmodels.compat.python import PY3
import functools

has_pytest = True
try:
    import pytest
except ImportError:
    has_pytest = False

if PY3:
    SKIP_TYPES = type
else:
    import types

    SKIP_TYPES = (type, types.ClassType)

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
    def decorator(test_item):
        if not hasattr(test_item, '__name__') and has_pytest:
            return pytest.mark.skip(reason)
        if not isinstance(test_item, SKIP_TYPES):
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


def example(t):
    """
    Label a test as an example.

    Parameters
    ----------
    t : callable
        The test to label as slow.

    Returns
    -------
    t : callable
        The decorated test `t`.

    Examples
    --------
    The `statsmodels.compat.testing` module includes ``example``.
    A test can be decorated as slow like this::

      from statsmodels.compat.testing import example

      @example
      def test_example(self):
          print('Running an example')
    """

    t.example = True
    return t


__all__ = ['skip', 'skipif', 'SkipTest', 'example']
