from __future__ import print_function
from .compat import PY3

from warnings import simplefilter

from .tools._testing import PytestTester
from .tools.sm_exceptions import (ConvergenceWarning, CacheWriteWarning,
                                  IterationLimitWarning, InvalidTestWarning)
from ._version import get_versions

__docformat__ = 'restructuredtext'

simplefilter("always", (ConvergenceWarning, CacheWriteWarning,
                        IterationLimitWarning, InvalidTestWarning))

debug_warnings = False

if debug_warnings:
    import warnings

    warnings.simplefilter("default")
    # use the following to raise an exception for debugging specific warnings
    # warnings.filterwarnings("error", message=".*integer.*")
    if PY3:
        # ResourceWarning doesn't exist in python 2
        # we have currently many ResourceWarnings in the datasets on python 3.4
        warnings.simplefilter("ignore", ResourceWarning)  # noqa:F821

test = PytestTester()

__version__ = get_versions()['version']
del get_versions
