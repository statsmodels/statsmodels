import numpy as np

def lagmat(x, maxlag, trim='forward'):
    '''create 2d array of lags

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

    Returns
    -------
    lagmat : 2d array
        array with lagged observations

    Examples
    --------
    >>> from scikits.statsmodels.sandbox.tsa.tsatools import lagmat
    >>> import numpy as np
    >>> X = np.arange(1,7).reshape(-1,2)
    >>> lagmat(X, maxlag=2, trim="forward")
    array([[ 1.,  2.,  0.,  0.,  0.,  0.],
       [ 3.,  4.,  1.,  2.,  0.,  0.],
       [ 5.,  6.,  3.,  4.,  1.,  2.]])

    >>> lagmat(X, maxlag=2, trim="backward")
    array([[ 5.,  6.,  3.,  4.,  1.,  2.],
       [ 0.,  0.,  5.,  6.,  3.,  4.],
       [ 0.,  0.,  0.,  0.,  5.,  6.]])

    >>> lagmat(X, maxlag=2, trim="both"
    array([[ 5.,  6.,  3.,  4.,  1.,  2.]])

    >>> lagmat(X, maxlag=2, trim="none")
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
    if x.ndim == 1:
        x = x[:,None]
    nobs, nvar = x.shape
    if maxlag >= nobs:
        raise ValueError("maxlag should be < nobs")
    lm = np.zeros((nobs+maxlag, nvar*(maxlag+1)))
    for k in range(0, int(maxlag+1)):
        #print k, maxlag-k,nobs-k, nvar*k,nvar*(k+1), x.shape, lm.shape
        lm[maxlag-k:nobs+maxlag-k, nvar*(maxlag-k):nvar*(maxlag-k+1)] = x
    if trim:
        trimlower = trim.lower()
    else:
        trimlower = trim
    if trimlower == 'none' or not trimlower:
        return lm
    elif trimlower == 'forward':
        return lm[:nobs+maxlag-k,:]
    elif trimlower == 'both':
        return lm[maxlag:nobs+maxlag-k,:]
    elif trimlower == 'backward':
        return lm[maxlag:,:]
    else:
        raise ValueError, 'trim option not valid'

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
        for all variables, except the first, lags from dropex to maxlagex are included
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


__all__ = ['lagmat', 'lagmat2ds']

if __name__ == '__main__':
    # sanity check, mainly for imports
    x = np.random.normal(size=(100,2))
    tmp = lagmat(x,2)
    tmp = lagmat2ds(x,2)
#    grangercausalitytests(x, 2)
