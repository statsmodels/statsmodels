"""
assert functions from numpy and pandas testing

"""
from statsmodels.compat.pandas import testing as pdt

import numpy.testing as npt
import pandas as pd

from statsmodels.tools.tools import Bunch

# Standard list for parsing tables
PARAM_LIST = ["params", "bse", "tvalues", "pvalues"]


def bunch_factory(attribute, columns):
    """
    Generates a special purpose Bunch class

    Parameters
    ----------
    attribute : str
        Attribute to access when splitting
    columns : List[str]
        List of names to use when splitting the columns of attribute

    Notes
    -----
    After the class is initialized as a Bunch, the columns of attribute
    are split so that Bunch has the keys in columns and
    bunch[column[i]] = bunch[attribute][:, i]

    """
    class FactoryBunch(Bunch):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if not hasattr(self, attribute):
                raise AttributeError("{} is required and must be passed to "
                                     "the constructor".format(attribute))
            for i, att in enumerate(columns):
                self[att] = getattr(self, attribute)[:, i]

    return FactoryBunch


ParamsTableTestBunch = bunch_factory("params_table", PARAM_LIST)

MarginTableTestBunch = bunch_factory("margins_table", PARAM_LIST)


class Holder:
    """Test-focused class to simplify accessing values by attribute"""

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __str__(self):
        ss = "\n".join(str(k) + " = " + str(v).replace("\n", "\n    ")
                       for k, v in vars(self).items())
        return ss

    def __repr__(self):
        # use repr for values including nested cases as in tost
        ss = "\n".join(str(k) + " = " + repr(v).replace("\n", "\n    ")
                       for k, v in vars(self).items())
        ss = str(self.__class__) + "\n" + ss
        return ss


# adjusted functions

def assert_equal(actual, desired, err_msg="", verbose=True, **kwds):
    """
    Dispatch to the appropriate pandas or numpy testing assert function

    Parameters
    ----------
    actual : array_like
        The value to test.
    desired : array_like
        The expected value. If a pandas Index, Series, or DataFrame, the
        matching pandas assert function is used.
    err_msg : str, optional
        The error message to use on failure when neither `desired` is a
        pandas Index, Series, or DataFrame.
    verbose : bool, optional
        Whether to include the actual and desired values in the error
        message when neither `desired` is a pandas Index, Series, or
        DataFrame.
    **kwds
        Additional keyword arguments passed to the pandas assert function
        when `desired` is a pandas Index, Series, or DataFrame.
    """
    if isinstance(desired, pd.Index):
        pdt.assert_index_equal(actual, desired)
    elif isinstance(desired, pd.Series):
        pdt.assert_series_equal(actual, desired, **kwds)
    elif isinstance(desired, pd.DataFrame):
        pdt.assert_frame_equal(actual, desired, **kwds)
    else:
        npt.assert_equal(actual, desired, err_msg="", verbose=True)
