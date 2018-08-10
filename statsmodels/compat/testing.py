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


def _id(obj):
    return obj


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


__all__ = ['example']
