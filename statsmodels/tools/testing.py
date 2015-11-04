"""assert functions from numpy and pandas testing

"""

import re
from distutils.version import StrictVersion, LooseVersion

import numpy as np
import numpy.testing as npt
import pandas
import pandas.util.testing as pdt

# for pandas version check
def strip_rc(version):
    return re.sub(r"rc\d+$", "", version)


def is_pandas_min_version(min_version):
    '''check whether pandas is at least min_version
    '''
    from pandas import __version__ as pversion
    return LooseVersion(pversion) >= min_version


# local copies, all unchanged
from numpy.testing import (assert_allclose, assert_almost_equal,
     assert_approx_equal, assert_array_almost_equal,
     assert_array_almost_equal_nulp, assert_array_equal, assert_array_less,
     assert_array_max_ulp, assert_raises, assert_string_equal, assert_warns)


# adjusted functions

if not is_pandas_min_version('0.14.1'):
    def assert_equal(actual, desired, err_msg='', verbose=True, **kwds):
        npt.assert_equal(actual, desired, err_msg='', verbose=True)
else:
    def assert_equal(actual, desired, err_msg='', verbose=True, **kwds):
        if isinstance(desired, pandas.Index):
            pdt.assert_index_equal(actual, desired)
        elif isinstance(desired, pandas.Series):
            pdt.assert_series_equal(actual, desired, **kwds)
        elif isinstance(desired, pandas.DataFrame):
            pdt.assert_frame_equal(actual, desired, **kwds)
        else:
            npt.assert_equal(actual, desired, err_msg='', verbose=True)
