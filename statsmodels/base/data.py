"""
Base tools for handling various kinds of data structures, attaching metadata to
results, and doing data cleaning
"""

import numpy as np
from pandas import DataFrame, Series, TimeSeries, isnull
from statsmodels.tools.decorators import (resettable_cache,
                cache_readonly, cache_writable)
import statsmodels.tools.data as data_util

try:
    reduce
    pass
except NameError:
    #python 3.2
    from functools import reduce

class MissingDataError(Exception):
    pass

def _asarray_2dcolumns(x):
    if np.asarray(x).ndim > 1 and np.asarray(x).squeeze().ndim == 1:
        return

def _asarray_2d_null_rows(x):
    """
    Makes sure input is an array and is 2d. Makes sure output is 2d. True
    indicates a null in the rows of 2d x.
    """
    #Have to have the asarrays because isnull doesn't account for array-like
    #input
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:,None]
    return np.any(isnull(x), axis=1)[:,None]


def _nan_rows(*arrs):
    """
    Returns a boolean array which is True where any of the rows in any
    of the _2d_ arrays in arrs are NaNs. Inputs can be any mixture of Series,
    DataFrames or array-like.
    """
    if len(arrs) == 1:
        arrs += ([[False]],)
    def _nan_row_maybe_two_inputs(x, y):
        # check for dtype bc dataframe has dtypes
        x_is_boolean_array = hasattr(x, 'dtype') and x.dtype == bool and x
        return np.logical_or(_asarray_2d_null_rows(x),
                             (x_is_boolean_array | _asarray_2d_null_rows(y)))
    return reduce(_nan_row_maybe_two_inputs, arrs).squeeze()

class ModelData(object):
    """
    Class responsible for handling input data and extracting metadata into the
    appropriate form
    """
    def __init__(self, endog, exog=None, missing='none', hasconst=None,
                       **kwargs):
        if missing != 'none':
            arrays, nan_idx = self._handle_missing(endog, exog, missing,
                                                       **kwargs)
            self.missing_row_idx = nan_idx
            self.__dict__.update(arrays) # attach all the data arrays
            self.orig_endog = self.endog
            self.orig_exog = self.exog
            self.endog, self.exog = self._convert_endog_exog(self.endog,
                    self.exog)
        else:
            self.__dict__.update(kwargs) # attach the extra arrays anyway
            self.orig_endog = endog
            self.orig_exog = exog
            self.endog, self.exog = self._convert_endog_exog(endog, exog)

        # this has side-effects, attaches k_constant and const_idx
        self._handle_constant(hasconst)
        self._check_integrity()
        self._cache = resettable_cache()

    def _handle_constant(self, hasconst):
        if hasconst is not None:
            if hasconst:
                self.k_constant = 1
                self.const_idx = None
            else:
                self.k_constant = 0
                self.const_idx = None
        else:
            try: # to detect where the constant is
                const_idx = np.where(self.exog.var(axis = 0) == 0)[0].squeeze()
                self.k_constant = const_idx.size
                if self.k_constant > 1:
                    raise ValueError("More than one constant detected.")
                else:
                    self.const_idx = const_idx
            except: # should be an index error but who knows, means no const
                self.const_idx = None
                self.k_constant = 0

    def _drop_nans(self, x, nan_mask):
        return x[nan_mask]

    def _drop_nans_2d(self, x, nan_mask):
        return x[nan_mask][:, nan_mask]

    def _handle_missing(self, endog, exog, missing, **kwargs):
        """
        This returns a dictionary with keys endog, exog and the keys of
        kwargs. It preserves Nones.
        """
        none_array_names = []

        if exog is not None:
            combined = (endog, exog)
            combined_names = ['endog', 'exog']
        else:
            combined = (endog,)
            combined_names = ['endog']
            none_array_names += ['exog']

        # deal with other arrays
        combined_2d = ()
        combined_2d_names = []
        if len(kwargs):
            for key, value_array in kwargs.iteritems():
                if value_array is None or value_array.ndim == 0:
                    none_array_names += [key]
                    continue
                # grab 1d arrays
                if value_array.ndim == 1:
                    combined += (value_array,)
                    combined_names += [key]
                elif value_array.squeeze().ndim == 1:
                    combined += (value_array,)
                    combined_names += [key]

                # grab 2d arrays that are _assumed_ to be symmetric
                elif value_array.ndim == 2:
                    combined_2d += (value_array,)
                    combined_2d_names += [key]
                else:
                    raise ValueError("Arrays with more than 2 dimensions "
                            "aren't yet handled")

        nan_mask = _nan_rows(*combined)
        if combined_2d:
            nan_mask = _nan_rows(*(nan_mask[:,None],) + combined_2d)

        if missing == 'raise' and np.any(nan_mask):
            raise MissingDataError("NaNs were encountered in the data")

        elif missing == 'drop':
            nan_mask = ~nan_mask
            drop_nans = lambda x : self._drop_nans(x, nan_mask)
            drop_nans_2d = lambda x : self._drop_nans_2d(x, nan_mask)
            combined = dict(zip(combined_names, map(drop_nans, combined)))
            if combined_2d:
                combined.update(dict(zip(combined_2d_names,
                                         map(drop_nans_2d, combined_2d))))
            if none_array_names:
                combined.update(dict(zip(none_array_names,
                                         [None]*len(none_array_names)
                                         )))
            return combined, np.where(~nan_mask)[0].tolist()
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
        endog = self.orig_endog
        ynames = self._get_names(endog)
        if not ynames:
            ynames = _make_endog_names(self.endog)

        if len(ynames) == 1:
            return ynames[0]
        else:
            return list(ynames)

    @cache_writable()
    def xnames(self):
        exog = self.orig_exog
        if exog is not None:
            xnames = self._get_names(exog)
            if not xnames:
                xnames = _make_exog_names(self.exog)
            return list(xnames)
        return None

    @cache_readonly
    def row_labels(self):
        exog = self.orig_exog
        if exog is not None:
            row_labels = self._get_row_labels(exog)
        else:
            endog = self.orig_endog
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
        return arr.design_info.column_names

class PandasData(ModelData):
    """
    Data handling class which knows how to reattach pandas metadata to model
    results
    """
    def _drop_nans(self, x, nan_mask):
        if hasattr(x, 'ix'):
            return x.ix[nan_mask]
        else: # extra arguments could be plain ndarrays
            return super(PandasData, self)._drop_nans(x, nan_mask)

    def _drop_nans_2d(self, x, nan_mask):
        if hasattr(x, 'ix'):
            return x.ix[nan_mask].ix[:, nan_mask]
        else:  # extra arguments could be plain ndarrays
            return super(PandasData, self)._drop_nans_2d(x, nan_mask)

    def _check_integrity(self):
        try:
            endog, exog = self.orig_endog, self.orig_exog
            # exog can be None and we could be upcasting one or the other
            if exog is not None and (hasattr(endog, 'index') and
                    hasattr(exog, 'index')):
                assert self.orig_endog.index.equals(self.orig_exog.index)
        except AssertionError:
            raise ValueError("The indices for endog and exog are not aligned")
        super(PandasData, self)._check_integrity()

    def _get_row_labels(self, arr):
        try:
            return arr.index
        except AttributeError:
            # if we've gotten here it's because endog is pandas and
            # exog is not, so just return the row labels from endog
            return self.orig_endog.index

    def attach_columns(self, result):
        # this can either be a 1d array or a scalar
        # don't squeeze because it might be a 2d row array
        # if it needs a squeeze, the bug is elsewhere
        if result.ndim <= 1:
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

def handle_data(endog, exog, missing='none', hasconst=None, **kwargs):
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

    return klass(endog, exog=exog, missing=missing, hasconst=hasconst, **kwargs)
