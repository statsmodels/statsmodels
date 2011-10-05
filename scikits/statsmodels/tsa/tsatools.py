import numpy as np
import numpy.lib.recfunctions as nprf
from scikits.statsmodels.tools.tools import add_constant

def add_trend(X, trend="c", prepend=False):
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

    Notes
    -----
    Returns columns as ["ctt","ct","c"] whenever applicable.  There is currently
    no checking for an existing constant or trend.

    See also
    --------
    scikits.statsmodels.add_constant
    """
    #TODO: could be generalized for trend of aribitrary order
    trend = trend.lower()
    if trend == "c":    # handles structured arrays
        return add_constant(X, prepend=prepend)
    elif trend == "ct" or trend == "t":
        trendorder = 1
    elif trend == "ctt":
        trendorder = 2
    else:
        raise ValueError("trend %s not understood" % trend)
    X = np.asanyarray(X)
    nobs = len(X)
    trendarr = np.vander(np.arange(1,nobs+1, dtype=float), trendorder+1)
    # put in order ctt
    trendarr = np.fliplr(trendarr)
    if trend == "t":
        trendarr = trendarr[:,1]
    if not X.dtype.names:
        if not prepend:
            X = np.column_stack((X, trendarr))
        else:
            X = np.column_stack((trendarr, X))
    else:
        return_rec = data.__clas__ is np.recarray
        if trendorder == 1:
            if trend == "ct":
                dt = [('const',float),('trend',float)]
            else:
                dt = [('trend', float)]
        elif trendorder == 2:
            dt = [('const',float),('trend',float),('trend_squared', float)]
        trendarr = trendarr.view(dt)
        if prepend:
            X = nprf.append_fields(trendarr, X.dtype.names, [X[i] for i
                in data.dtype.names], usemask=False, asrecarray=return_rec)
        else:
            X = nprf.append_fields(X, trendarr.dtype.names, [trendarr[i] for i
                in trendarr.dtype.names], usemask=false, asrecarray=return_rec)
    return X

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

    >>> import scikits.statsmodels.api as sm
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
            raise IndexError, "col is None and the input array is not 1d"
        elif len(names) == 1:
            col = names[0]
        if isinstance(col, int):
            col = x.dtype.names[col]
        contemp = x[col]

        # make names for lags
        tmp_names = [col + '_'+'L(%i)' % i for i in range(1,lags+1)]
        ndlags = lagmat(contemp, maxlag=lags, trim='Both')

        # get index for return
        if insert is True:
            ins_idx = list(names).index(col) + 1
        elif insert is False:
            ins_idx = len(names) + 1
        else: # insert is an int
            if insert > len(names):
                raise Warning("insert > number of variables, inserting at the"+
                              " last position")
            ins_idx = insert

        first_names = list(names[:ins_idx])
        last_names = list(names[ins_idx:])

        if drop:
            if col in first_names:
                first_names.pop(first_names.index(col))
            else:
                last_names.pop(last_names.index(col))

        if first_names: # only do this if x isn't "empty"
            first_arr = nprf.append_fields(x[first_names][lags:],tmp_names,
                        ndlags.T, usemask=False)
        else:
            first_arr = np.zeros(len(x)-lags, dtype=zip(tmp_names,
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
                raise Warning("insert > number of variables, inserting at the"+
                              " last position")
            ins_idx = insert

        ndlags = lagmat(contemp, lags, trim='Both')
        first_cols = range(ins_idx)
        last_cols = range(ins_idx,x.shape[1])
        if drop:
            if col in first_cols:
                first_cols.pop(first_cols.index(col))
            else:
                last_cols.pop(last_cols.index(col))
        return np.column_stack((x[lags:,first_cols],ndlags,
                    x[lags:,last_cols]))

def detrend(x, order=1, axis=0):
    '''detrend an array with a trend of given order along axis 0 or 1

    Parameters
    ----------
    x : array_like, 1d or 2d
        data, if 2d, then each row or column is independently detrended with the
        same trendorder, but independent trend estimates
    order : int
        specifies the polynomial order of the trend, zero is constant, one is
        linear trend, two is quadratic trend
    axis : int
        for detrending with order > 0, axis can be either 0 observations by rows,
        or 1, observations by columns

    Returns
    -------
    detrended data series : ndarray
        The detrended series is the residual of the linear regression of the
        data on the trend of given order.


    '''
    x = np.asarray(x)
    nobs = x.shape[0]
    if order == 0:
        return x - np.expand_dims(x.mean(ax), x)
    else:
        if x.ndim == 2 and range(2)[axis]==1:
            x = x.T
        elif x.ndim > 2:
            raise NotImplementedError('x.ndim>2 is not implemented until it is needed')
        #could use a polynomial, but this should work also with 2d x, but maybe not yet
        trends = np.vander(np.arange(nobs).astype(float), N=order+1)
        beta = np.linalg.lstsq(trends, x)[0]
        resid = x - np.dot(trends, beta)
        if x.ndim == 2 and range(2)[axis]==1:
            resid = resid.T
        return resid


def lagmat(x, maxlag, trim='forward', original='ex'):
    '''create 2d array of lags

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

    Returns
    -------
    lagmat : 2d array
        array with lagged observations
    y : 2d array, optional
        Only returned if original == 'sep'

    Examples
    --------
    >>> from scikits.statsmodels.tsa.tsatools import lagmat
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
    TODO:
    * allow list of lags additional to maxlag
    * create varnames for columns
    '''
    x = np.asarray(x)
    dropidx = 0
    if x.ndim == 1:
        x = x[:,None]
    nobs, nvar = x.shape
    if original in ['ex','sep']:
        dropidx = nvar
    if maxlag >= nobs:
        raise ValueError("maxlag should be < nobs")
    lm = np.zeros((nobs+maxlag, nvar*(maxlag+1)))
    for k in range(0, int(maxlag+1)):
        lm[maxlag-k:nobs+maxlag-k, nvar*(maxlag-k):nvar*(maxlag-k+1)] = x
    if trim:
        trimlower = trim.lower()
    else:
        trimlower = trim
    if trimlower == 'none' or not trimlower:
        startobs = 0
        stopobs = len(lm)
    elif trimlower == 'forward':
        startobs = 0
        stopobs = nobs+maxlag-k
    elif trimlower == 'both':
        startobs = maxlag
        stopobs = nobs+maxlag-k
    elif trimlower == 'backward':
        startobs = maxlag
        stopobs = len(lm)

    else:
        raise ValueError('trim option not valid')
    if original == 'sep':
        return lm[startobs:stopobs,dropidx:], x[startobs:stopobs]
    else:
        return lm[startobs:stopobs,dropidx:]

def lagmat2ds(x, maxlag0, maxlagex=None, dropex=0, trim='forward'):
    '''generate lagmatrix for 2d array, columns arranged by variables

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

    Returns
    -------
    lagmat : 2d array
        array with lagged observations, columns ordered by variable

    Notes
    -----
    very inefficient for unequal lags, just done for convenience
    '''
    if maxlagex is None:
        maxlagex = maxlag0
    maxlag = max(maxlag0, maxlagex)
    nobs, nvar = x.shape
    lagsli = [lagmat(x[:,0], maxlag, trim=trim, original='in')[:,:maxlag0+1]]
    for k in range(1,nvar):
        lagsli.append(lagmat(x[:,k], maxlag, trim=trim, original='in')[:,dropex:maxlagex+1])
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
    tmp = np.eye(n * (n + 1) / 2)
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


__all__ = ['lagmat', 'lagmat2ds','add_trend', 'duplication_matrix',
           'elimination_matrix', 'commutation_matrix',
           'vec', 'vech', 'unvec', 'unvech']

if __name__ == '__main__':
    # sanity check, mainly for imports
    x = np.random.normal(size=(100,2))
    tmp = lagmat(x,2)
    tmp = lagmat2ds(x,2)
#    grangercausalitytests(x, 2)
