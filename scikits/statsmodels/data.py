"""
Base tools for handling various kinds of data structures, attaching metadata to
results, and doing data cleaning
"""

import numpy as np
from pandas import DataFrame, Series

class ModelData(object):
    """

    """
    def __init__(self, endog, exog=None):
        self._orig_endog = endog
        self._orig_exog = exog
        (self.endog, self.exog, self.ynames,
         self.xnames, self.row_labels) = self._convert_endog_exog()
        self._check_integrity()

    def _convert_endog_exog(self):
        endog = self._orig_endog
        exog = self._orig_exog

        # for consistent outputs if endog is (n,1)
        yarr = self._get_yarr(endog)
        xarr = None
        if exog is not None:
            xarr = self._get_xarr(exog)
            if xarr.ndim == 1:
                xarr = xarr[:, None]
            if xarr.ndim != 2:
                raise ValueError("exog is not 1d or 2d")

        if exog is not None:
            xnames = self._get_names(exog)
            if not xnames:
                xnames = _make_exog_names(xarr)

        ynames = _make_endog_names(endog)

        if exog is not None:
            row_labels = _get_row_labels(exog)
        else:
            row_labels = _get_row_labels(endog)

        return yarr, xarr, ynames, xnames, row_labels

    def _get_names(self, exog):
        try:
            return exog.dtype.names
        except AttributeError:
            pass

        if isinstance(exog, DataFrame):
            return list(exog.columns)

        return None

    def _get_yarr(self, endog):
        return np.asarray(endog).squeeze()

    def _get_xarr(self, exog):
        return np.asarray(exog)

    def _check_integrity(self):
        if self.exog is not None:
            if len(self.exog) != len(self.endog):
                raise ValueError("endog and exog matrices are different sizes")

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
    """
    Data handling class which knows how to reattach pandas metadata to model
    results
    """

    def attach_columns(self, result):
        return Series(result, index=self.xnames)

    def attach_cov(self, result):
        return DataFrame(result, index=self.xnames, columns=self.xnames)

    def attach_rows(self, result):
        return Series(result, index=self.row_labels)

class LarryData(ModelData):
    """
    Data handling class which knows how to reattach pandas metadata to model
    results
    """

    def _get_yarr(self, endog):
        try:
            return endog.x
        except AttributeError:
            return np.asarray(endog).squeeze()

    def _get_xarr(self, exog):
        try:
            return exog.x
        except AttributeError:
            return np.asarray(exog)

    def _get_names(self, exog):
        try:
            return exog.label[1]
        except Exception:
            pass

        return None

    def attach_columns(self, result):
        import la
        return la.larry(result, [self.xnames])

    def attach_cov(self, result):
        import la
        return la.larry(result, [self.xnames, self.xnames])

    def attach_rows(self, result):
        import la
        return la.larry(result, [self.row_labels])

def _get_row_labels(exog):
    try:
        return exog.index
    except AttributeError:
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
    """
    Given inputs
    """
    if _is_using_pandas(endog, exog):
        klass = PandasData
    elif _is_using_ndarray(endog, exog):
        klass = ModelData
    elif _is_using_larry(endog, exog):
        klass = LarryData
    else:
        raise ValueError('unrecognized data structures: %s / %s' %
                         (type(endog), type(exog)))

    return klass(endog, exog=exog)

def _is_using_ndarray(endog, exog):
    return (isinstance(endog, np.ndarray) and
            (isinstance(exog, np.ndarray) or exog is None))

def _is_using_pandas(endog, exog):
    from pandas.core.generic import PandasGeneric
    return (isinstance(endog, PandasGeneric) or
            isinstance(exog, PandasGeneric))

def _is_using_larry(endog, exog):
    import la
    return isinstance(endog, la.larry) or isinstance(exog, la.larry)
