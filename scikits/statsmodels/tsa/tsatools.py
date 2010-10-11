import numpy as np
import numpy.lib.recfunctions as nprf
from scikits.statsmodels.tools import add_constant

def add_trend(X, trend="c", prepend=False):
    """
    Adds a trend and/or constant to an array.

    Parameters
    ----------
    X : array-like
        Original array of data.
    trend : str {"c","ct","ctt"}
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
    if trend == "t":
        trendarr = trendarr[:,0]
    if not X.dtype.names:
        if not prepend:
            X = np.column_stack((X, trendarr))
        else:
            X = np.column_stack((trendarr, X))
    else:
        return_rec = data.__clas__ is np.recarray
        if trendorder == 1:
            if trend == "ct":
                dt = [('trend',float),('const',float)]
            else:
                dt = [('trend', float)]
        elif trendorder == 2:
            dt = [('trend_squared', float),('trend',float),('const',float)]
        trendarr = trendarr.view(dt)
        if prepend:
            X = nprf.append_fields(trendarr, X.dtype.names, [X[i] for i
                in data.dtype.names], usemask=False, asrecarray=return_rec)
        else:
            X = nprf.append_fields(X, trendarr.dtype.names, [trendarr[i] for i
                in trendarr.dtype.names], usemask=false, asrecarray=return_rec)
    return X

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
    >>> from scikits.statsmodels.sandbox.tsa.tsatools import lagmat
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
        lm = lm[:,dropidx:]
    elif trimlower == 'forward':
        lm = lm[:nobs+maxlag-k,dropidx:]
    elif trimlower == 'both':
        lm = lm[maxlag:nobs+maxlag-k,dropidx:]
    elif trimlower == 'backward':
        lm = lm[maxlag:,dropidx:]
    else:
        raise ValueError, 'trim option not valid'
    if original == 'sep':
        return lm, x[maxlag:]
    else:
        return lm

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
    lagsli = [lagmat(x[:,0], maxlag, trim=trim)[:,:maxlag0]]
    for k in range(1,nvar):
        lagsli.append(lagmat(x[:,k], maxlag, trim=trim)[:,dropex:maxlagex])
    return np.column_stack(lagsli)


__all__ = ['lagmat', 'lagmat2ds','add_trend']

if __name__ == '__main__':
    # sanity check, mainly for imports
    x = np.random.normal(size=(100,2))
    tmp = lagmat(x,2)
    tmp = lagmat2ds(x,2)
#    grangercausalitytests(x, 2)
