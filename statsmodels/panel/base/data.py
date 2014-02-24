import numpy as np

from statsmodels.base.data import ModelData, PatsyData, PandasData
from statsmodels.tools.grouputils import Grouping
import statsmodels.tools.data as data_util

#TODO: just move this all to base/data.py?

def _is_balanced(data, dropna=False):
    """
    Checks if len(time) x len(panel) == len(data)
    """
    if dropna:
        dropped = data.dropna()
        return np.multiply.reduce(dropped.index.levshape) == len(dropped)
    else:
        return np.multiply.reduce(data.index.levshape) == len(data)

def _is_balanced_index(index):
        return np.multiply.reduce(index.levshape) == len(index)

def _check_panel_time(kwargs):
    """
    Checks that panel and time are in the dict kwargs, pops them off, and
    returns them. This will modify kwargs in place.
    """
    panel = kwargs.pop('panel', None)
    if panel is None:
        raise ValueError("panel cannot be None")
    time = kwargs.pop('time', None)
    if time is None:
        raise ValueError('time cannot be None')

    return panel, time


class _CommonPanelMethods(object):
    def _initialize(self):
        self.exog, idx = self.groupings.sort(self.exog)
        self.endog, idx = self.groupings.sort(self.endog)
        self.groupings.reindex(idx) # sorting may have altered index
        self.is_balanced = _is_balanced_index(idx)
        self.n_panel, self.n_time = self.groupings.index.levshape


class PanelData(ModelData, _CommonPanelMethods):
    def __init__(self, endog, exog, missing='none', hasconst=None, **kwargs):
        panel, time = _check_panel_time(kwargs)
        self.groupings = Grouping(index_list=[panel, time])
        super(PanelData, self).__init__(endog, exog, missing, hasconst,
                                      **kwargs)
        self._initialize()


class PatsyPanelData(PatsyData, _CommonPanelMethods):
    def __init__(self, endog, exog, missing='none', hasconst=None, **kwargs):
        panel, time = _check_panel_time(kwargs)
        self.groupings = Grouping(index_list=[panel, time])
        super(PanelData, self).__init__(endog, exog, missing, hasconst,
                                      **kwargs)
        self._initialize()


class PandasPanelData(PandasData, _CommonPanelMethods):
    def __init__(self, endog, exog, missing='none', hasconst=None, **kwargs):
        if not endog.index.equals(exog.index):
            raise ValueError("endog and exog index do not equal each other")
            #TODO: could probably relax this and just take the MultiIndex one

        from pandas import MultiIndex
        if not isinstance(endog.index, MultiIndex):
            raise ValueError("index is not a MultiIndex")

        self.groupings = Grouping(index=endog.index)

        super(PandasPanelData, self).__init__(endog, exog, missing, hasconst,
                                              **kwargs)
        self._initialize()


def handle_panel_data(endog, exog, missing='none', hasconst=None, **kwargs):
    if isinstance(endog, (list, tuple)):
        endog = np.asarray(endog)
    if isinstance(exog, (list, tuple)):
        exog = np.asarray(exog)

    if data_util._is_using_ndarray_type(endog, exog):
        klass = PanelData
    elif data_util._is_using_pandas(endog, exog):
        klass = PandasPanelData
    elif data_util._is_using_patsy(endog, exog):
        klass = PatsyPanelData
    # keep this check last
    elif data_util._is_using_ndarray(endog, exog):
        klass = PanelData
    else:
        raise ValueError('unrecognized data structures: %s / %s' %
                         (type(endog), type(exog)))

    return klass(endog, exog=exog, missing=missing, hasconst=hasconst, **kwargs)

### Tests

def test_is_balanced():
    from statsmodels.datasets import grunfeld
    balanced_panel = grunfeld.load_pandas().data

    drop_idx = ((balanced_panel.firm == 'General Motors') &
                 balanced_panel.year.isin([1935, 1938]))
    drop_idx = drop_idx | ((balanced_panel.firm == 'Westinghouse') &
                           balanced_panel.year.isin([1936, 1941]))
    unbalanced_panel = balanced_panel.ix[~drop_idx]

    balanced_panel.set_index(['firm', 'year'], inplace=True)
    unbalanced_panel.set_index(['firm', 'year'], inplace=True)

    np.testing.assert_(_is_balanced(balanced_panel))
    np.testing.assert_(not _is_balanced(unbalanced_panel))
