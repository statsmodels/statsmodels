"""assert functions from numpy and pandas testing

"""

import re

import numpy.testing as npt
import pandas
import pandas.util.testing as pdt

# for pandas version check
def strip_rc(version):
    return re.sub(r"rc\d+$", "", version)


# adjusted functions

def assert_equal(actual, desired, err_msg='', verbose=True, **kwds):
    if isinstance(desired, pandas.Index):
        pdt.assert_index_equal(actual, desired)
    elif isinstance(desired, pandas.Series):
        pdt.assert_series_equal(actual, desired, **kwds)
    elif isinstance(desired, pandas.DataFrame):
        pdt.assert_frame_equal(actual, desired, **kwds)
    else:
        npt.assert_equal(actual, desired, err_msg='', verbose=True)
