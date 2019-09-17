"""assert functions from numpy and pandas testing

"""
from statsmodels.compat.pandas import testing as pdt

import numpy.testing as npt
import pandas

from statsmodels.tools.tools import Bunch

# Standard list for parsing tables
PARAM_LIST = ['params', 'bse', 'tvalues', 'pvalues']


def bunch_factory(attribute, columns):
    """
    Generates a special purpose Bunch class

    Parameters
    ----------
    attribute: str
        Attribute to access when splitting
    columns: List[str]
        List of names to use when splitting the columns of attribute

    Notes
    -----
    After the class is initialized as a Bunch, the columne of attribute
    are split so that Bunch has the keys in columns and
    bunch[column[i]] = bunch[attribute][:, i]
    """
    class FactoryBunch(Bunch):
        def __init__(self, *args, **kwargs):
            super(FactoryBunch, self).__init__(*args, **kwargs)
            if not hasattr(self, attribute):
                raise AttributeError('{0} is required and must be passed to '
                                     'the constructor'.format(attribute))
            for i, att in enumerate(columns):
                self[att] = getattr(self, attribute)[:, i]

    return FactoryBunch


ParamsTableTestBunch = bunch_factory('params_table', PARAM_LIST)

MarginTableTestBunch = bunch_factory('margins_table', PARAM_LIST)


class Holder(object):
    """
    Test-focused class to simplify accessing values by attribute
    """
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


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
