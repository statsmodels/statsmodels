
from statsmodels.compat.python import range, lrange, lzip, long, PY3
from statsmodels.compat.numpy import recarray_select

import numpy as np
import numpy.lib.recfunctions as nprf
import pandas as pd
from pandas import DataFrame
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.data import _is_using_pandas, _is_recarray


def add_trend(x, trend="c", prepend=False, has_constant='skip'):
    """
    Adds a trend and/or constant to an array.

    Parameters
    ----------
    X : array-like
        Original array of data.
    trend : str {"c","t","ct","ctt"}
        "c" add constant only
        "t" add trend only
        "ct" add constant and linear trend
        "ctt" add constant and linear and quadratic trend.
    prepend : bool
        If True, prepends the new data to the columns of X.
    has_constant : str {'raise', 'add', 'skip'}
        Controls what happens when trend is 'c' and a constant already
        exists in X. 'raise' will raise an error. 'add' will duplicate a
        constant. 'skip' will return the data without change. 'skip' is the
        default.

    Returns
    -------
    y : array, recarray or DataFrame
        The original data with the additional trend columns.  If x is a
        recarray or pandas Series or DataFrame, then the trend column names
        are 'const', 'trend' and 'trend_squared'.

    Notes
    -----
    Returns columns as ["ctt","ct","c"] whenever applicable. There is currently
    no checking for an existing trend.

    See also
    --------
    statsmodels.tools.tools.add_constant
    """
    # TODO: could be generalized for trend of aribitrary order
    trend = trend.lower()
    columns = ['const', 'trend', 'trend_squared']
    if trend == "c":  # handles structured arrays
        columns = columns[:1]
        trendorder = 0
    elif trend == "ct" or trend == "t":
        columns = columns[:2]
        if trend == "t":
            columns = columns[1:2]
        trendorder = 1
    elif trend == "ctt":
        trendorder = 2
    else:
        raise ValueError("trend %s not understood" % trend)

    is_recarray = _is_recarray(x)
    is_pandas = _is_using_pandas(x, None) or is_recarray
    if is_pandas or is_recarray:
        if is_recarray:
            descr = x.dtype.descr
            x = pd.DataFrame.from_records(x)
        elif isinstance(x, pd.Series):
            x = pd.DataFrame(x)
        else:
            x = x.copy()
    else:
        x = np.asanyarray(x)

    nobs = len(x)
    trendarr = np.vander(np.arange(1, nobs + 1, dtype=np.float64), trendorder + 1)
    # put in order ctt
    trendarr = np.fliplr(trendarr)
    if trend == "t":
        trendarr = trendarr[:, 1]

    if "c" in trend:
        if is_pandas or is_recarray:
            # Mixed type protection
            def safe_is_const(s):
                try:
                    return np.ptp(s) == 0.0 and np.any(s != 0.0)
                except:
                    return False
            col_const = x.apply(safe_is_const, 0)
        else:
            col_const = np.logical_and(np.any(np.ptp(np.asanyarray(x), axis=0) == 0, axis=0),
                                       np.all(x != 0.0, axis=0))
        if np.any(col_const):
            if has_constant == 'raise':
                raise ValueError("x already contains a constant")
            elif has_constant == 'skip':
                columns = columns[1:]
                trendarr = trendarr[:, 1:]

    order = 1 if prepend else -1
    if is_recarray or is_pandas:
        trendarr = pd.DataFrame(trendarr, index=x.index, columns=columns)
        x = [trendarr, x]
        x = pd.concat(x[::order], 1)
    else:
        x = [trendarr, x]
        x = np.column_stack(x[::order])

    if is_recarray:
        x = x.to_records(index=False, convert_datetime64=False)
        new_descr = x.dtype.descr
        extra_col = len(new_descr) - len(descr)
        if prepend:
            descr = new_descr[:extra_col] + descr
        else:
            descr = descr + new_descr[-extra_col:]

        if not PY3:
            # See 3658
            names = [entry[0] for entry in descr]
            dtypes = [entry[1] for entry in descr]
            names = [bytes(name) for name in names]
            # Fail loudly if there is a non-ascii name
            descr = list(zip(names, dtypes))

        x = x.astype(np.dtype(descr))

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
        If data is a structured array or a recarray, `col` can be a string
        that is the name of the column containing the variable. Or `col` can
        be an int of the zero-based column index. If it's a 1d array `col`
        can be None.
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
    if x.dtype.names:
        names = x.dtype.names
        if not col and np.squeeze(x).ndim > 1:
            raise IndexError("col is None and the input array is not 1d")
        elif len(names) == 1:
            col = names[0]
        if isinstance(col, (int, long)):
            col = x.dtype.names[col]
        if not PY3:
            # TODO: Get rid of this kludge.  See GH # 3658
            names = [bytes(name) if isinstance(name, unicode) else name for name in names]
            # Fail loudly if there is a non-ascii name.
            x.dtype.names = names
            if isinstance(col, unicode):
                col = bytes(col)

        contemp = x[col]

        # make names for lags
        tmp_names = [col + '_'+'L(%i)' % i for i in range(1, lags+1)]
        ndlags = lagmat(contemp, maxlag=lags, trim='Both')

        # get index for return
        if insert is True:
            ins_idx = list(names).index(col) + 1
        elif insert is False:
            ins_idx = len(names) + 1
        else: # insert is an int
            if insert > len(names):
                import warnings
                warnings.warn("insert > number of variables, inserting at the"
                              " last position", ValueWarning)
            ins_idx = insert

        first_names = list(names[:ins_idx])
        last_names = list(names[ins_idx:])

        if drop:
            if col in first_names:
                first_names.pop(first_names.index(col))
            else:
                last_names.pop(last_names.index(col))

        if first_names: # only do this if x isn't "empty"
            # Workaround to avoid NumPy FutureWarning
            _x = recarray_select(x, first_names)
            first_arr = nprf.append_fields(_x[lags:], tmp_names, ndlags.T,
                                           usemask=False)

        else:
            first_arr = np.zeros(len(x)-lags, dtype=lzip(tmp_names,
                (x[col].dtype,)*lags))
            for i,name in enumerate(tmp_names):
                first_arr[name] = ndlags[:,i]
        if last_names:
            return nprf.append_fields(first_arr, last_names,
                    [x[name][lags:] for name in last_names], usemask=False)
        else: # lags for last variable
            return first_arr

    else: # we have an ndarray

        if x.ndim == 1: # make 2d if 1d
            x = x[:,None]
        if col is None:
            col = 0

        # handle negative index
        if col < 0:
            col = x.shape[1] + col

        contemp = x[:,col]

        if insert is True:
            ins_idx = col + 1
        elif insert is False:
            ins_idx = x.shape[1]
        else:
            if insert < 0: # handle negative index
                insert = x.shape[1] + insert + 1
            if insert > x.shape[1]:
                insert = x.shape[1]
                import warnings
                warnings.warn("insert > number of variables, inserting at the"
                              " last position", ValueWarning)
            ins_idx = insert

        ndlags = lagmat(contemp, lags, trim='Both')
        first_cols = lrange(ins_idx)
        last_cols = lrange(ins_idx,x.shape[1])
        if drop:
            if col in first_cols:
                first_cols.pop(first_cols.index(col))
            else:
                last_cols.pop(last_cols.index(col))
        return np.column_stack((x[lags:,first_cols],ndlags,
                    x[lags:,last_cols]))


def detrend(x, order=1, axis=0):
    """
    Detrend an array with a trend of given order along axis 0 or 1

    Parameters
    ----------
    x : array_like, 1d or 2d
        data, if 2d, then each row or column is independently detrended with the
        same trendorder, but independent trend estimates
    order : int
        specifies the polynomial order of the trend, zero is constant, one is
        linear trend, two is quadratic trend
    axis : int
        axis can be either 0, observations by rows,
        or 1, observations by columns

    Returns
    -------
    detrended data series : ndarray
        The detrended series is the residual of the linear regression of the
        data on the trend of given order.
    """
    if x.ndim == 2 and int(axis) == 1:
        x = x.T
    elif x.ndim > 2:
        raise NotImplementedError('x.ndim > 2 is not implemented until it is needed')

    nobs = x.shape[0]
    if order == 0:
        # Special case demean
        resid = x - x.mean(axis=0)
    else:
        trends = np.vander(np.arange(float(nobs)), N=order + 1)
        beta = np.linalg.pinv(trends).dot(x)
        resid = x - np.dot(trends, beta)

    if x.ndim == 2 and int(axis) == 1:
        resid = resid.T

    return resid


def lagmat(x, maxlag, trim='forward', original='ex', use_pandas=False):
    """
    Create 2d array of lags

    Parameters
    ----------
    x : array_like, 1d or 2d
        data; if 2d, observation in rows and variables in columns
    maxlag : int
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
    use_pandas : bool, optional
        If true, returns a DataFrame when the input is a pandas
        Series or DataFrame.  If false, return numpy ndarrays.

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
    When using a pandas DataFrame or Series with use_pandas=True, trim can only
    be 'forward' or 'both' since it is not possible to consistently extend index
    values.
    """
    # TODO:  allow list of lags additional to maxlag
    is_pandas = _is_using_pandas(x, None) and use_pandas
    trim = 'none' if trim is None else trim
    trim = trim.lower()
    if is_pandas and trim in ('none', 'backward'):
        raise ValueError("trim cannot be 'none' or 'forward' when used on "
                         "Series or DataFrames")

    xa = np.asarray(x)
    dropidx = 0
    if xa.ndim == 1:
        xa = xa[:, None]
    nobs, nvar = xa.shape
    if original in ['ex', 'sep']:
        dropidx = nvar
    if maxlag >= nobs:
        raise ValueError("maxlag should be < nobs")
    lm = np.zeros((nobs + maxlag, nvar * (maxlag + 1)))
    for k in range(0, int(maxlag + 1)):
        lm[maxlag - k:nobs + maxlag - k,
        nvar * (maxlag - k):nvar * (maxlag - k + 1)] = xa

    if trim in ('none', 'forward'):
        startobs = 0
    elif trim in ('backward', 'both'):
        startobs = maxlag
    else:
        raise ValueError('trim option not valid')

    if trim in ('none', 'backward'):
        stopobs = len(lm)
    else:
        stopobs = nobs

    if is_pandas:
        x_columns = x.columns if isinstance(x, DataFrame) else [x.name]
        columns = [str(col) for col in x_columns]
        for lag in range(maxlag):
            lag_str = str(lag + 1)
            columns.extend([str(col) + '.L.' + lag_str for col in x_columns])
        lm = DataFrame(lm[:stopobs], index=x.index, columns=columns)
        lags = lm.iloc[startobs:]
        if original in ('sep', 'ex'):
            leads = lags[x_columns]
            lags = lags.drop(x_columns, 1)
    else:
        lags = lm[startobs:stopobs, dropidx:]
        if original == 'sep':
            leads = lm[startobs:stopobs, :dropidx]

    if original == 'sep':
        return lags, leads
    else:
        return lags


def lagmat2ds(x, maxlag0, maxlagex=None, dropex=0, trim='forward',
              use_pandas=False):
    """
    Generate lagmatrix for 2d array, columns arranged by variables

    Parameters
    ----------
    x : array_like, 2d
        2d data, observation in rows and variables in columns
    maxlag0 : int
        for first variable all lags from zero to maxlag are included
    maxlagex : None or int
        max lag for all other variables all lags from zero to maxlag are included
    dropex : int (default is 0)
        exclude first dropex lags from other variables
        for all variables, except the first, lags from dropex to maxlagex are
        included
    trim : string
        * 'forward' : trim invalid observations in front
        * 'backward' : trim invalid initial observations
        * 'both' : trim invalid observations on both sides
        * 'none' : no trimming of observations
    use_pandas : bool, optional
        If true, returns a DataFrame when the input is a pandas
        Series or DataFrame.  If false, return numpy ndarrays.

    Returns
    -------
    lagmat : 2d array
        array with lagged observations, columns ordered by variable

    Notes
    -----
    Inefficient implementation for unequal lags, implemented for convenience
    """

    if maxlagex is None:
        maxlagex = maxlag0
    maxlag = max(maxlag0, maxlagex)
    is_pandas = _is_using_pandas(x, None)

    if x.ndim == 1:
        if is_pandas:
            x = pd.DataFrame(x)
        else:
            x = x[:, None]
    elif x.ndim == 0 or x.ndim > 2:
        raise TypeError('Only supports 1 and 2-dimensional data.')

    nobs, nvar = x.shape

    if is_pandas and use_pandas:
        lags = lagmat(x.iloc[:, 0], maxlag, trim=trim,
                      original='in', use_pandas=True)
        lagsli = [lags.iloc[:, :maxlag0 + 1]]
        for k in range(1, nvar):
            lags = lagmat(x.iloc[:, k], maxlag, trim=trim,
                          original='in', use_pandas=True)
            lagsli.append(lags.iloc[:, dropex:maxlagex + 1])
        return pd.concat(lagsli, axis=1)
    elif is_pandas:
        x = np.asanyarray(x)

    lagsli = [lagmat(x[:, 0], maxlag, trim=trim, original='in')[:, :maxlag0 + 1]]
    for k in range(1, nvar):
        lagsli.append(lagmat(x[:, k], maxlag, trim=trim, original='in')[:, dropex:maxlagex + 1])
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
    assert(k * k == len(v))
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

    Returns
    -------
    D_n : ndarray
    """
    tmp = np.eye(n * (n + 1) // 2)
    return np.array([unvech(x).ravel() for x in tmp]).T

def elimination_matrix(n):
    """
    Create the elimination matrix L_n which satisfies vech(M) = L_n vec(M) for
    any matrix M

    Parameters
    ----------

    Returns
    -------

    """
    vech_indices = vec(np.tril(np.ones((n, n))))
    return np.eye(n * n)[vech_indices != 0]

def commutation_matrix(p, q):
    """
    Create the commutation matrix K_{p,q} satisfying vec(A') = K_{p,q} vec(A)

    Parameters
    ----------
    p : int
    q : int

    Returns
    -------
    K : ndarray (pq x pq)
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
    newparams = ((1-np.exp(-params))/
                (1+np.exp(-params))).copy()
    tmp = ((1-np.exp(-params))/
               (1+np.exp(-params))).copy()
    for j in range(1,len(params)):
        a = newparams[j]
        for kiter in range(j):
            tmp[kiter] -= a * newparams[j-kiter-1]
        newparams[:j] = tmp[:j]
    return newparams

def _ar_invtransparams(params):
    """
    Inverse of the Jones reparameterization

    Parameters
    ----------
    params : array
        The transformed AR coefficients
    """
    # AR coeffs
    tmp = params.copy()
    for j in range(len(params)-1,0,-1):
        a = params[j]
        for kiter in range(j):
            tmp[kiter] = (params[kiter] + a * params[j-kiter-1])/\
                    (1-a**2)
        params[:j] = tmp[:j]
    invarcoefs = -np.log((1-params)/(1+params))
    return invarcoefs

def _ma_transparams(params):
    """
    Transforms params to induce stationarity/invertability.

    Parameters
    ----------
    params : array
        The ma coeffecients of an (AR)MA model.

    Reference
    ---------
    Jones(1980)
    """
    newparams = ((1-np.exp(-params))/(1+np.exp(-params))).copy()
    tmp = ((1-np.exp(-params))/(1+np.exp(-params))).copy()

    # levinson-durbin to get macf
    for j in range(1,len(params)):
        b = newparams[j]
        for kiter in range(j):
            tmp[kiter] += b * newparams[j-kiter-1]
        newparams[:j] = tmp[:j]
    return newparams

def _ma_invtransparams(macoefs):
    """
    Inverse of the Jones reparameterization

    Parameters
    ----------
    params : array
        The transformed MA coefficients
    """
    tmp = macoefs.copy()
    for j in range(len(macoefs)-1,0,-1):
        b = macoefs[j]
        for kiter in range(j):
            tmp[kiter] = (macoefs[kiter]-b *macoefs[j-kiter-1])/(1-b**2)
        macoefs[:j] = tmp[:j]
    invmacoefs = -np.log((1-macoefs)/(1+macoefs))
    return invmacoefs


def unintegrate_levels(x, d):
    """
    Returns the successive differences needed to unintegrate the series.

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
    unintegrate
    """
    x = x[:d]
    return np.asarray([np.diff(x, d - i)[0] for i in range(d, 0, -1)])


def unintegrate(x, levels):
    """
    After taking n-differences of a series, return the original series

    Parameters
    ----------
    x : array-like
        The n-th differenced series
    levels : list
        A list of the first-value in each differenced series, for
        [first-difference, second-difference, ..., n-th difference]

    Returns
    -------
    y : array-like
        The original series de-differenced

    Examples
    --------
    >>> x = np.array([1, 3, 9., 19, 8.])
    >>> levels = unintegrate_levels(x, 2)
    >>> levels
    array([ 1.,  2.])
    >>> unintegrate(np.diff(x, 2), levels)
    array([  1.,   3.,   9.,  19.,   8.])
    """
    levels = list(levels)[:] # copy
    if len(levels) > 1:
        x0 = levels.pop(-1)
        return unintegrate(np.cumsum(np.r_[x0, x]), levels)
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
    elif freq == 'W' or freq.startswith('W-'):
        return 52
    elif freq == 'D':
        return 7
    elif freq == 'B':
        return 5
    elif freq == 'H':
        return 24
    else:  # pragma : no cover
        raise ValueError("freq {} not understood. Please report if you "
                         "think this is in error.".format(freq))


__all__ = ['lagmat', 'lagmat2ds','add_trend', 'duplication_matrix',
           'elimination_matrix', 'commutation_matrix',
           'vec', 'vech', 'unvec', 'unvech']

