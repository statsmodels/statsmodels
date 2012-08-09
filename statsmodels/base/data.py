"""
Base tools for handling various kinds of data structures, attaching metadata to
results, and doing data cleaning
"""

import numpy as np
from pandas import DataFrame, Series, TimeSeries
from statsmodels.tools.decorators import (resettable_cache,
                cache_readonly, cache_writable)
import statsmodels.tools.data as data_util

class MissingDataError(Exception):
    pass

class ModelData(object):
    """
    Class responsible for handling input data and extracting metadata into the
    appropriate form
    """
    def __init__(self, endog, exog=None, missing=None, **kwds):
        self._orig_endog = endog
        self._orig_exog = exog
        if missing is not None:
            endog, exog = self._handle_missing(endog, exog, missing)
        self.endog, self.exog = self._convert_endog_exog(endog, exog)
        self._check_integrity()
        self._cache = resettable_cache()

    def _handle_missing(self, endog, exog, missing):
        if endog.ndim == 1:
            endog = endog[:,None]
        if exog is not None:
            combined = np.c_[endog, exog]
        else:
            combined = endog

        if missing == 'raise' and np.any(np.isnan(combined)):
            raise MissingDataError("NaNs were encountered in the data")

        elif missing == 'drop':
            endog_idx = endog.shape[1]
            combined = combined[~np.isnan(combined).any(axis=1)]
            endog = combined[:,:endog_idx].squeeze()
            if exog is not None:
                exog = combined[:,endog_idx:]
            return endog, exog
        else:
            raise ValueError("missing option %s not understood" % missing)

    def _convert_endog_exog(self, endog, exog):

        # for consistent outputs if endog is (n,1)
        yarr = self._get_yarr(endog)
        xarr = None
        if exog is not None:
            xarr = self._get_xarr(exog)
            if xarr.ndim == 1:
                xarr = xarr[:, None]
            if xarr.ndim != 2:
                raise ValueError("exog is not 1d or 2d")

        return yarr, xarr

    @cache_writable()
    def ynames(self):
        endog = self._orig_endog
        ynames = self._get_names(endog)
        if not ynames:
            ynames = _make_endog_names(self.endog)

        if len(ynames) == 1:
            return ynames[0]
        else:
            return list(ynames)

    @cache_writable()
    def xnames(self):
        exog = self._orig_exog
        if exog is not None:
            xnames = self._get_names(exog)
            if not xnames:
                xnames = _make_exog_names(self.exog)
            return list(xnames)
        return None

    @cache_readonly
    def row_labels(self):
        exog = self._orig_exog
        if exog is not None:
            row_labels = self._get_row_labels(exog)
        else:
            endog = self._orig_endog
            row_labels = self._get_row_labels(endog)
        return row_labels

    def _get_row_labels(self, arr):
        return None

    def _get_names(self, arr):
        if isinstance(arr, DataFrame):
            return list(arr.columns)
        elif isinstance(arr, Series):
            if arr.name:
                return [arr.name]
            else:
                return
        else:
            try:
                return arr.dtype.names
            except AttributeError:
                pass

        return None

    def _get_yarr(self, endog):
        if data_util._is_structured_ndarray(endog):
            endog = data_util.struct_to_ndarray(endog)
        return np.asarray(endog).squeeze()

    def _get_xarr(self, exog):
        if data_util._is_structured_ndarray(exog):
            exog = data_util.struct_to_ndarray(exog)
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
        elif how == 'dates':
            return self.attach_dates(obj)
        elif how == 'columns_eq':
            return self.attach_columns_eq(obj)
        elif how == 'cov_eq':
            return self.attach_cov_eq(obj)
        else:
            return obj

    def attach_columns(self, result):
        return result

    def attach_columns_eq(self, result):
        return result

    def attach_cov(self, result):
        return result

    def attach_cov_eq(self, result):
        return result

    def attach_rows(self, result):
        return result

    def attach_dates(self, result):
        return result

class PatsyData(ModelData):
    def _get_names(self, arr):
        return arr.column_info.column_names

class PandasData(ModelData):
    """
    Data handling class which knows how to reattach pandas metadata to model
    results
    """
    def _handle_missing(self, endog, exog, missing):
        if missing == 'raise':
            from pandas import isnull
            if np.any(isnull(endog)) or (exog is not None and
                                         np.any(isnull(exog))):
                raise MissingDataError("NaNs were encountered in the data")

        elif missing == 'drop':
            y_index = endog.dropna().index
            if exog is not None:
                x_index = exog.dropna().index
            else:
                x_index = y_index # just union with itself

            index = y_index.intersection(x_index)
            if exog is not None:
                exog = exog.ix[index]
            return endog.ix[index], exog
        else:
            raise ValueError("missing option %s not understood" % missing)

    def _check_integrity(self):
        try:
            endog, exog = self._orig_endog, self._orig_exog
            # exog can be None and we could be upcasting one or the other
            if exog is not None and (hasattr(endog, 'index') and
                    hasattr(exog, 'index')):
                assert self._orig_endog.index.equals(self._orig_exog.index)
        except AssertionError:
            raise ValueError("The indices for endog and exog are not aligned")
        super(PandasData, self)._check_integrity()

    def _get_row_labels(self, arr):
        try:
            return arr.index
        except AttributeError, err:
            # if we've gotten here it's because endog is pandas and
            # exog is not, so just return the row labels from endog
            return self._orig_endog.index

    def attach_columns(self, result):
        if result.squeeze().ndim <= 1:
            return Series(result, index=self.xnames)
        else: # for e.g., confidence intervals
            return DataFrame(result, index=self.xnames)

    def attach_columns_eq(self, result):
        return DataFrame(result, index=self.xnames, columns=self.ynames)

    def attach_cov(self, result):
        return DataFrame(result, index=self.xnames, columns=self.xnames)

    def attach_cov_eq(self, result):
        return DataFrame(result, index=self.ynames, columns=self.ynames)

    def attach_rows(self, result):
        # assumes if len(row_labels) > len(result) it's bc it was truncated
        # at the front, for AR lags, for example
        if result.squeeze().ndim == 1:
            return Series(result, index=self.row_labels[-len(result):])
        else: # this is for VAR results, may not be general enough
            return DataFrame(result, index=self.row_labels[-len(result):],
                                columns=self.ynames)

    def attach_dates(self, result):
        return TimeSeries(result, index=self.predict_dates)

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
        exog_names = ['x%d' % i for i in range(1,exog.shape[1])]
        exog_names.insert(const_idx, 'const')
    else:
        exog_names = ['x%d' % i for i in range(1,exog.shape[1]+1)]

    return exog_names

def handle_data(endog, exog, missing=None):
    """
    Given inputs
    """
    # deal with lists and tuples up-front
    if isinstance(endog, (list, tuple)):
        endog = np.asarray(endog)
    if isinstance(exog, (list, tuple)):
        exog = np.asarray(exog)

    if data_util._is_using_ndarray_type(endog, exog):
        klass = ModelData
    elif data_util._is_using_pandas(endog, exog):
        klass = PandasData
    elif data_util._is_using_patsy(endog, exog):
        klass = PatsyData
    # keep this check last
    elif data_util._is_using_ndarray(endog, exog):
        klass = ModelData
    else:
        raise ValueError('unrecognized data structures: %s / %s' %
                         (type(endog), type(exog)))

    return klass(endog, exog=exog, missing=missing)
