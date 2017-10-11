from __future__ import print_function

__docformat__ = 'restructuredtext'

from distutils.version import LooseVersion
import os
import sys

from warnings import simplefilter
from ._version import get_versions
from statsmodels.tools.sm_exceptions import (ConvergenceWarning, CacheWriteWarning,
                                             IterationLimitWarning, InvalidTestWarning)

simplefilter("always", (ConvergenceWarning, CacheWriteWarning,
                        IterationLimitWarning, InvalidTestWarning))

debug_warnings = False

if debug_warnings:
    import warnings

    warnings.simplefilter("default")
    # use the following to raise an exception for debugging specific warnings
    # warnings.filterwarnings("error", message=".*integer.*")
    if (sys.version_info[0] >= 3):
        # ResourceWarning doesn't exist in python 2
        # we have currently many ResourceWarnings in the datasets on python 3.4
        warnings.simplefilter("ignore", ResourceWarning)


class PytestTester(object):
    def __init__(self):
        f = sys._getframe(1)
        package_path = f.f_locals.get('__file__', None)
        if package_path is None:
            raise ValueError('Unable to determine path')
        self.package_path = os.path.dirname(package_path)
        self.package_name = f.f_locals.get('__name__', None)

    def __call__(self, extra_args=None, exit=False):
        try:
            import pytest
            if not LooseVersion(pytest.__version__) >= LooseVersion('3.0'):
                raise ImportError
            extra_args = ['--tb=short','--disable-pytest-warnings'] if extra_args is None else extra_args
            cmd = [self.package_path] + extra_args
            print('Running pytest ' + ' '.join(cmd))
            status = pytest.main(cmd)
            if exit:
                sys.exit(status)
        except ImportError:
            raise ImportError('pytest>=3 required to run the test')


test = PytestTester()

__version__ = get_versions()['version']
del get_versions
