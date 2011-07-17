import numpy as np

from pandas import DataFrame, Series
from pandas.core.generic import PandasGeneric

class ModelData(object):
    """

    """
    def __init__(self, endog, exog=None):
        # for consistent outputs if endog is (n,1)
        yarr = np.asarray(endog).squeeze()

        xarr = None
        xnames = None
        if not exog is None:
            xarr = np.asarray(exog)
            if xarr.ndim == 1:
                xarr = xarr[:, None]
            if xarr.ndim != 2:
                raise ValueError("exog is not 1d or 2d")
            xnames = _get_names(exog)
            if not xnames:
                xnames = _make_exog_names(xarr)

        self.row_labels = _get_row_labels(exog)
        self.xnames = xnames
        self.ynames = _make_endog_names(endog)
        self.endog = yarr
        self.exog = xarr

        self._check_integrity()

    def _check_integrity(self):
        if self.exog is not None:
            if len(self.exog) != len(self.endog):
                raise ValueError("endog and exog matrices are not aligned.")

    def wrap_output(self, obj, how='columns'):
        if how == 'columns':
            return self.attach_columns(obj)
        elif how == 'rows':
            return self.attach_rows(obj)
        elif how == 'cov':
            return self.attach_cov(obj)
        else:
            return obj

    def attach_columns(self, result):
        return result

    def attach_cov(self, result):
        return result

    def attach_rows(self, result):
        return result

class PandasData(ModelData):

    def attach_columns(self, result):
        return Series(result, index=self.xnames)

    def attach_cov(self, result):
        return DataFrame(result, index=self.xnames, columns=self.xnames)

    def attach_rows(self, result):
        return Series(result, index=self.row_labels)

def _get_row_labels(exog):
    try:
        return exog.index
    except AttributeError:
        return None

def _get_names(data):
    try:
        return data.dtype.names
    except AttributeError:
        pass

    if isinstance(data, DataFrame):
        return list(data.columns)

    return None

def _is_structured_array(data):
    return isinstance(data, np.ndarray) and data.dtype.names is not None

def _make_endog_names(endog):
    if endog.ndim == 1 or endog.shape[1] == 1:
        ynames = ['y']
    else: # for VAR
        ynames = ['y%d' % (i+1) for i in range(endog.shape[1])]

    return ynames

def _make_exog_names(exog):
    exog_var = exog.var(0)
    if (exog_var == 0).any():
        # assumes one constant in first or last position
        # avoid exception if more than one constant
        const_idx = exog_var.argmin()
        if const_idx == exog.shape[1] - 1:
            exog_names = ['x%d' % i for i in range(1,exog.shape[1])]
            exog_names += ['const']
        else:
            exog_names = ['x%d' % i for i in range(exog.shape[1])]
            exog_names[const_idx] = 'const'
    else:
        exog_names = ['x%d' % i for i in range(exog.shape[1])]

    return exog_names

def handle_data(endog, exog):
    klass = ModelData
    if isinstance(endog, PandasGeneric) or isinstance(exog, PandasGeneric):
        klass = PandasData

    return klass(endog, exog=exog)
