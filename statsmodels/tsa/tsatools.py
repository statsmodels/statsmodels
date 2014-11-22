from statsmodels.compat.python import range, lrange, lzip
import numpy as np
from statsmodels.compat import string_types
import numpy as np
import pandas as pd

from statsmodels.compat import string_types

import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from warnings import warn


__all__ = ['lagmat', 'lagmat2ds', 'add_trend', 'duplication_matrix',
           'elimination_matrix', 'commutation_matrix',
           'vec', 'vech', 'unvec', 'unvech', 'reintegrate', 'coint',
           'adfuller']


def adfuller(*args, **kwargs):
    """See statsmodels.tsa.tsatools.adfuller"""
    import unitroot

    func = np.deprecate(unitroot.adfuller,
                        old_name='statsmodels.tsa.tsatools.adfuller',
                        new_name='statsmodels.tsa.unitroot.ADF')
    return func(*args, **kwargs)


def coint(*args, **kwargs):
    """See statsmodels.tsa.cointegration.coint"""
    import cointegration

    func = np.deprecate(cointegration.coint,
                        old_name='statsmodels.tsa.tsatools.coint',
                        new_name='statsmodels.tsa.cointegration.coint')
    return func(*args, **kwargs)


class ColumnNameConflict(Warning):
    pass


column_name_conflict_doc = """
Some of the column named being added were not unique and have been renamed.

             {0}
"""


def _enforce_unique_col_name(existing, new):
    converted_names = []
    unique_names = list(new[:])
    for i, n in enumerate(new):
        if n in existing:
            original_name = n
            fixed_name = n
            duplicate_count = 0
            while fixed_name in existing:
                fixed_name = n + '_' + str(duplicate_count)
                duplicate_count += 1
            unique_names[i] = fixed_name
            converted_names.append(
                '{0}   ->   {1}'.format(original_name, fixed_name))
    if converted_names:
        import warnings

        ws = column_name_conflict_doc.format('\n    '.join(converted_names))
        warnings.warn(ws, ColumnNameConflict)

    return unique_names


def add_trend(x=None, trend="c", prepend=False, nobs=None, has_constant='skip'):
    """
    Adds a trend and/or constant to an array.

    Parameters
    ----------
    x : array-like or None
        Original array of data. If None, then nobs must be a positive integer
    trend : str {"c","t","ct","ctt"}
        "c" add constant only
        "t" add trend only
        "ct" add constant and linear trend
        "ctt" add constant and linear and quadratic trend.
    prepend : bool
        If True, prepends the new data to the columns of x.
    n : int, positive
        Positive integer containing the length of the trend series.  Only used
        if x is none.
    has_constant : str {'raise', 'add', 'skip'}
        Controls what happens when trend is 'c' and a constant already
        exists in X. 'raise' will raise an error. 'add' will duplicate a
        constant. 'skip' will return the data without change. 'skip' is the
        default.

    Notes
    -----
    Returns columns as ["ctt","ct","t","c"] whenever applicable. There is currently
    no checking for an existing trend.

    See also
    --------
    statsmodels.tools.tools.add_constant
    """
    #TODO: could be generalized for trend of aribitrary order
    trend = trend.lower()
    if trend == "c":  # handles structured arrays
        trend_order = 0
    elif trend == "ct" or trend == "t":
        trend_order = 1
    elif trend == "ctt":
        trend_order = 2
    else:
        raise ValueError("trend %s not understood" % trend)
    if x is not None:
        nobs = len(np.asanyarray(x))
    elif nobs <= 0:
        raise ValueError("nobs must be a positive integer if x is None")
    trend_array = np.vander(np.arange(1, nobs + 1, dtype=np.float64),
                            trend_order + 1)
    # put in order ctt
    trend_array = np.fliplr(trend_array)
    if trend == "t":
        trend_array = trend_array[:, 1:]
        # check for constant
    if x is None:
        return trend_array
    x_array = np.asarray(x)
    if "c" in trend and \
            np.any(np.logical_and(np.ptp(x_array, axis=0) == 0,
                                  np.all(x_array != 0, axis=0))):
        if has_constant == 'raise':
            raise ValueError("x already contains a constant")
        elif has_constant == 'add':
            pass
        elif has_constant == 'skip' and trend == "ct":
            trend_array = trend_array[:, 1]
    if isinstance(x, pd.DataFrame):
        columns = ('const', 'trend', 'quadratic_trend')
        if trend == 't':
            columns = (columns[1],)
        else:
            columns = columns[0:trend_order + 1]
        columns = _enforce_unique_col_name(x.columns, columns)
        trend_array = pd.DataFrame(trend_array, index=x.index, columns=columns)
        if prepend:
            x = trend_array.join(x)
        else:
            x = x.join(trend_array)
    else:
        if prepend:
            x = np.column_stack((trend_array, x))
        else:
            x = np.column_stack((x, trend_array))

    return x


def add_lag(x, col=None, lags=1, drop=False, insert=True):
    """
    Returns an array with lags included given an array.

    Parameters
    ----------
    x : array
        An array or NumPy ndarray subclass. Can be either a 1d or 2d array with
        observations in columns.
    col : 'string', int, or None
        If data is a pandas DataFrame, `col` can be a string that is the name
        of the column containing the variable. Or `col` can be an int of the
        zero-based column index. If it's a 1d array `col` can be None.
    lags : int
        The number of lags desired.
    drop : bool
        Whether to keep the contemporaneous variable for the data.
    insert : bool or int
        If True, inserts the lagged values after `col`. If False, appends
        the data. If int inserts the lags at int.

    Returns
    -------
    array : ndarray
        Array with lags

    Examples
    --------

    >>> import statsmodels.api as sm
    >>> data = sm.datasets.macrodata.load()
    >>> data = data.data[['year','quarter','realgdp','cpi']]
    >>> data = sm.tsa.add_lag(data, 'realgdp', lags=2)

    Notes
    -----
    Trims the array both forward and backward, so that the array returned
    so that the length of the returned array is len(`X`) - lags. The lags are
    returned in increasing order, ie., t-1,t-2,...,t-lags
    """
    df = None
    is_rec_array = False
    try:
        if x.dtype.fields:
            is_rec_array = True
            x = pd.DataFrame(x)
    except:
        pass

    if isinstance(x, pd.DataFrame):
        df = x
        if isinstance(col, string_types):
            col = list(df.columns).index(col)

    x = np.asarray(x)
    if x.ndim == 1:  # make 2d if 1d
        x = x[:, None]
    if col is None:
        col = 0
    # handle negative index
    if col < 0:
        col = x.shape[1] + col
    contemporary = x[:, col]

    if insert is True:
        ins_idx = col + 1
    elif insert is False:
        ins_idx = x.shape[1]
    else:
        if insert < 0:  # handle negative index
            insert = x.shape[1] + insert + 1
        if insert > x.shape[1]:
            raise ValueError("insert greater than the number of variables")
        ins_idx = insert

    ndlags = lagmat(contemporary, lags, trim='Both')
    first_cols = lrange(ins_idx)
    last_cols = lrange(ins_idx, x.shape[1])
    if drop:
        if col in first_cols:
            first_cols.pop(first_cols.index(col))
        else:
            last_cols.pop(last_cols.index(col))
    out = np.column_stack((x[lags:, first_cols], ndlags, x[lags:, last_cols]))
    if df is not None:
        columns = list(df.columns)
        index = df.index
        # Create new column labels
        lag_columns = [str(columns[col]) + '_L_' + str(i + 1) for i in
                       range(lags)]
        out_columns = [columns[col_idx] for col_idx in first_cols]
        out_columns.extend(lag_columns)
        for col_idx in last_cols:
            out_columns.append(columns[col_idx])
        # Alter index for correct length
        index = index[lags:]
        lag_columns = _enforce_unique_col_name(df.columns, lag_columns)
        ndlags = pd.DataFrame(ndlags, columns=lag_columns, index=index)
        df = df.join(ndlags)
        out = df[out_columns].reindex(index)
    if is_rec_array:
        out = out.to_records(index=False)

    return out


def detrend(x, order=1, axis=0):
    """
    Detrend an array with a trend of given order along axis 0 or 1

    Parameters
    ----------
    x : array_like, 1d or 2d
        data, if 2d, then each row or column is independently detrended with the
        same trend order, but independent trend estimates
    order : int
        specifies the polynomial order of the trend, zero is constant, one is
        linear trend, two is quadratic trend
    axis : int
        for detrending with order > 0, axis can be either 0 observations by
        rows, or 1, observations by columns

    Returns
    -------
    y : array
        The detrended series is the residual of the linear regression of the
        data on the trend of given order.
    """
    x = np.asarray(x)
    ndim = x.ndim
    if order == 0:
        return x - np.expand_dims(x.mean(axis), axis)
    if ndim > 2:
        raise NotImplementedError('x must be 1d or 2d')
    if ndim == 2 and axis == 1:
        x = x.T
    elif ndim == 1:
        x = x[:, None]
    nobs = x.shape[0]
    trends = np.vander(np.arange(nobs).astype(np.float64), N=order + 1)
    beta = np.linalg.pinv(trends).dot(x)
    resid = x - np.dot(trends, beta)
    if x.ndim == 2 and axis == 1:
        resid = resid.T
    elif ndim == 1:
        resid = resid.ravel()

    return resid


def lagmat(x, maxlag, trim='forward', original='ex', fill_value=np.nan):
    """create 2d array of lags

    Parameters
    ----------
    x : array_like, 1d or 2d
        data; if 2d, observation in rows and variables in columns
    maxlag : int or sequence of ints
        all lags from zero to maxlag are included
    trim : str {'forward', 'backward', 'both', 'none'} or None
        * 'forward' : trim invalid observations in front
        * 'backward' : trim invalid initial observations
        * 'both' : trim invalid observations on both sides
        * 'none', None : no trimming of observations
    original : str {'ex','sep','in'}
        * 'ex' : drops the original array returning only the lagged values.
        * 'in' : returns the original array and the lagged values as a single
          array.
        * 'sep' : returns a tuple (original array, lagged values). The original
                  array is truncated to have the same number of rows as
                  the returned lagmat.
    fill_value : float
        the value to use for filling missing values.  Default is np.nan

    Returns
    -------
    lagmat : 2d array
        array with lagged observations
    y : 2d array, optional
        Only returned if original == 'sep'

    Examples
    --------
    >>> from statsmodels.tsa.tsatools import lagmat
    >>> import numpy as np
    >>> X = np.arange(1,7).reshape(-1,2)
    >>> lagmat(X, maxlag=2, trim="forward", original='in')
    array([[ 1.,  2.,  0.,  0.,  0.,  0.],
       [ 3.,  4.,  1.,  2.,  0.,  0.],
       [ 5.,  6.,  3.,  4.,  1.,  2.]])

    >>> lagmat(X, maxlag=2, trim="backward", original='in')
    array([[ 5.,  6.,  3.,  4.,  1.,  2.],
       [ 0.,  0.,  5.,  6.,  3.,  4.],
       [ 0.,  0.,  0.,  0.,  5.,  6.]])

    >>> lagmat(X, maxlag=2, trim="both", original='in')
    array([[ 5.,  6.,  3.,  4.,  1.,  2.]])

    >>> lagmat(X, maxlag=2, trim="none", original='in')
    array([[ 1.,  2.,  0.,  0.,  0.,  0.],
       [ 3.,  4.,  1.,  2.,  0.,  0.],
       [ 5.,  6.,  3.,  4.,  1.,  2.],
       [ 0.,  0.,  5.,  6.,  3.,  4.],
       [ 0.,  0.,  0.,  0.,  5.,  6.]])

    Notes
    -----
    If trim is not 'both', new values are 0-filled
    """
    # TODO: allow list of lags additional to maxlag
    # TODO: create varnames for columns
    x = np.asarray(x)
    dropidx = 0
    if x.ndim == 1:
        x = x[:, None]
    nobs, nvar = x.shape
    if original in ['ex', 'sep']:
        dropidx = nvar
    if maxlag >= nobs:
        raise ValueError("maxlag should be < nobs")
    lm = np.empty((nobs + maxlag, nvar * (maxlag + 1)))
    lm.fill(fill_value)
    for k in range(0, int(maxlag + 1)):
        lm[maxlag - k:nobs + maxlag - k,
        nvar * (maxlag - k):nvar * (maxlag - k + 1)] = x
    if trim:
        trimlower = trim.lower()
    else:
        trimlower = trim
    if trimlower == 'none' or not trimlower:
        startobs = 0
        stopobs = len(lm)
    elif trimlower == 'forward':
        startobs = 0
        stopobs = nobs + maxlag - k
    elif trimlower == 'both':
        startobs = maxlag
        stopobs = nobs + maxlag - k
    elif trimlower == 'backward':
        startobs = maxlag
        stopobs = len(lm)

    else:
        raise ValueError('trim option not valid')
    if original == 'sep':
        return lm[startobs:stopobs, dropidx:], x[startobs:stopobs]
    else:
        return lm[startobs:stopobs, dropidx:]


def lagmat2ds(x, maxlag0, maxlagex=None, dropex=0, trim='forward'):
    """Generate lagmatrix for 2d array, columns arranged by variables

    Parameters
    ----------
    x : array_like, 2d
        2d data, observation in rows and variables in columns
    maxlag0 : int
        for first variable all lags from zero to maxlag are included
    maxlagex : None or int
        max lag for all other variables; all lags from zero to maxlag are included
    dropex : int (default is 0)
        exclude first dropex lags from other variables
        for all variables, except the first, lags from dropex to maxlagex are
        included
    trim : string
        * 'forward' : trim invalid observations in front
        * 'backward' : trim invalid initial observations
        * 'both' : trim invalid observations on both sides
        * 'none' : no trimming of observations

    Returns
    -------
    lagmat : 2d array
        array with lagged observations, columns ordered by variable

    Notes
    -----
    very inefficient for unequal lags, just done for convenience
    """
    if maxlagex is None:
        maxlagex = maxlag0
    maxlag = max(maxlag0, maxlagex)
    nobs, nvar = x.shape
    lagsli = [
        lagmat(x[:, 0], maxlag, trim=trim, original='in')[:, :maxlag0 + 1]]
    for k in range(1, nvar):
        lagsli.append(lagmat(x[:, k], maxlag, trim=trim, original='in')[:,
                      dropex:maxlagex + 1])
    return np.column_stack(lagsli)


def vec(mat):
    return mat.ravel('F')


def vech(mat):
    # Gets Fortran-order
    return mat.T.take(_triu_indices(len(mat)))


# tril/triu/diag, suitable for ndarray.take
def _tril_indices(n):
    rows, cols = np.tril_indices(n)
    return rows * n + cols


def _triu_indices(n):
    rows, cols = np.triu_indices(n)
    return rows * n + cols


def _diag_indices(n):
    rows, cols = np.diag_indices(n)
    return rows * n + cols


def unvec(v):
    k = int(np.sqrt(len(v)))
    assert (k * k == len(v))
    return v.reshape((k, k), order='F')


def unvech(v):
    # quadratic formula, correct fp error
    rows = .5 * (-1 + np.sqrt(1 + 8 * len(v)))
    rows = int(np.round(rows))

    result = np.zeros((rows, rows))
    result[np.triu_indices(rows)] = v
    result = result + result.T

    # divide diagonal elements by 2
    result[np.diag_indices(rows)] /= 2

    return result


def duplication_matrix(n):
    """
    Create duplication matrix D_n which satisfies vec(S) = D_n vech(S) for
    symmetric matrix S

    Parameters
    ----------
    n : int
        The length of vech(S)

    Returns
    -------
    D_n : array
        The duplication array
    """
    tmp = np.eye(n * (n + 1) / 2)
    return np.array([unvech(x).ravel() for x in tmp]).T


def elimination_matrix(n):
    """
    Create the elimination matrix L_n which satisfies vech(M) = L_n vec(M) for
    any matrix M

    Parameters
    ----------
    n : int
        The length of vec(M)

    Returns
    -------
    L : array
        The commutation matrix

    """
    vech_indices = vec(np.tril(np.ones((n, n))))
    return np.eye(n * n)[vech_indices != 0]


def commutation_matrix(p, q):
    """
    Create the commutation matrix K_{p,q} satisfying vec(A') = K_{p,q} vec(A)

    Parameters
    ----------
    p, q : int

    Returns
    -------
    K : ndarray, (pq, pq)
    """
    K = np.eye(p * q)
    indices = np.arange(p * q).reshape((p, q), order='F')
    return K.take(indices.ravel(), axis=0)


def _ar_transparams(params):
    """
    Transforms params to induce stationarity/invertability.

    Parameters
    ----------
    params : array
        The AR coefficients

    Reference
    ---------
    Jones(1980)
    """
    trans_params = ((1 - np.exp(-params)) / (1 + np.exp(-params)))
    tmp = trans_params.copy()
    for j in range(1, len(params)):
        a = trans_params[j]
        for k in range(j):
            tmp[k] -= a * trans_params[j - k - 1]
        trans_params[:j] = tmp[:j]
    return trans_params


def _ar_invtransparams(params):
    """
    Inverse of the Jones reparameterization

    Parameters
    ----------
    params : array
        The transformed AR coefficients
    """
    # AR coefficients
    tmp = params.copy()
    for j in range(len(params) - 1, 0, -1):
        a = params[j]
        for k in range(j):
            tmp[k] = (params[k] + a * params[j - k - 1]) / \
                     (1 - a ** 2)
        params[:j] = tmp[:j]
    inv_ar_coefs = -np.log((1 - params) / (1 + params))
    return inv_ar_coefs


def _ma_transparams(params):
    """
    Transforms params to induce stationarity/invertability.

    Parameters
    ----------
    params : array
        The MA coeffecients of an (AR)MA model.

    Reference
    ---------
    Jones(1980)
    """
    trans_params = ((1 - np.exp(-params)) / (1 + np.exp(-params))).copy()
    tmp = trans_params.copy()

    # levinson-durbin to get macf
    for j in range(1, len(params)):
        b = trans_params[j]
        for k in range(j):
            tmp[k] += b * trans_params[j - k - 1]
        trans_params[:j] = tmp[:j]
    return trans_params


def _ma_invtransparams(macoefs):
    """
    Inverse of the Jones reparameterization

    Parameters
    ----------
    params : array
        The transformed MA coefficients
    """
    tmp = macoefs.copy()
    for j in range(len(macoefs) - 1, 0, -1):
        b = macoefs[j]
        scale = (1 - b ** 2)
        for k in range(j):
            tmp[k] = (macoefs[k] - b * macoefs[j - k - 1]) / scale
        macoefs[:j] = tmp[:j]
    inv_ma_coefs = -np.log((1 - macoefs) / (1 + macoefs))
    return inv_ma_coefs


def reintegrate_levels(x, d):
    """
    Returns the successive differences needed to reintegrate the series.

    Parameters
    ----------
    x : array-like
        The original series
    d : int
        The number of differences of the differenced series.

    Returns
    -------
    y : array-like
        The increasing differences from 0 to d-1 of the first d elements
        of x.

    See Also
    --------
    reintegrate
    """
    x = x[:d]
    return np.asarray([np.diff(x, d - i)[0] for i in range(d, 0, -1)])

def reintegrate(x, levels):
    """
    After taking n-differences of a series, return the original series

    Parameters
    ----------
    x : array-like
        The n-th differenced series
    levels : list
        A list of the initial-value in each differenced series, for
        [first-difference, second-difference, ..., n-th difference]

    Returns
    -------
    y : array-like
        The original series, reintegrated

    See Also
    --------
    reintegrate_levels

    Examples
    --------
    >>> x = np.array([1, 3, 9., 19, 8.])
    >>> levels = [x[0], np.diff(x, 1)[0]]
    >>> reintegrate(np.diff(x, 2), levels)
    array([  1.,   3.,   9.,  19.,   8.])
    """
    levels = list(levels[:])  # copy
    if len(levels) > 1:
        x0 = levels.pop(-1)
        return reintegrate(np.cumsum(np.r_[x0, x]), levels)
    x0 = levels[0]
    return np.cumsum(np.r_[x0, x])


def freq_to_period(freq):
    """
    Convert a pandas frequency to a periodicity

    Parameters
    ----------
    freq : str or offset
        Frequency to convert

    Returns
    -------
    period : int
        Periodicity of freq

    Notes
    -----
    Annual maps to 1, quarterly maps to 4, monthly to 12, weekly to 52.
    """
    if not isinstance(freq, offsets.DateOffset):
        freq = to_offset(freq)  # go ahead and standardize
    freq = freq.rule_code.upper()

    if freq == 'A' or freq.startswith(('A-', 'AS-')):
        return 1
    elif freq == 'Q' or freq.startswith(('Q-', 'QS-')):
        return 4
    elif freq == 'M' or freq.startswith(('M-', 'MS')):
        return 12
    elif freq == 'B' or freq == 'W' or freq.startswith('W-'):
        return 52
    else:  # pragma : no cover
        raise ValueError("freq {} not understood. Please report if you "
                         "think this in error.".format(freq))


if __name__ == '__main__':
    pass
