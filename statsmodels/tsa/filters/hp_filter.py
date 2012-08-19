from scipy import sparse
from scipy.sparse import dia_matrix, eye as speye
from scipy.sparse.linalg import spsolve
import numpy as np

def hpfilter(X, lamb=1600):
    """
    Hodrick-Prescott filter

    Parameters
    ----------
    X : array-like
        The 1d ndarray timeseries to filter of length (nobs,) or (nobs,1)
    lamb : float
        The Hodrick-Prescott smoothing parameter. A value of 1600 is
        suggested for quarterly data. Ravn and Uhlig suggest using a value
        of 6.25 (1600/4**4) for annual data and 129600 (1600*3**4) for monthly
        data.

    Returns
    -------
    cycle : array
        The estimated cycle in the data given lamb.
    trend : array
        The estimated trend in the data given lamb.

    Examples
    ---------
    >>> import statsmodels.api as sm
    >>> dta = sm.datasets.macrodata.load()
    >>> X = dta.data['realgdp']
    >>> cycle, trend = sm.tsa.filters.hpfilter(X,1600)

    Notes
    -----
    The HP filter removes a smooth trend, `T`, from the data `X`. by solving

    min sum((X[t] - T[t])**2 + lamb*((T[t+1] - T[t]) - (T[t] - T[t-1]))**2)
     T   t

    Here we implemented the HP filter as a ridge-regression rule using
    scipy.sparse. In this sense, the solution can be written as

    T = inv(I - lamb*K'K)X

    where I is a nobs x nobs identity matrix, and K is a (nobs-2) x nobs matrix
    such that

    K[i,j] = 1 if i == j or i == j + 2
    K[i,j] = -2 if i == j + 1
    K[i,j] = 0 otherwise

    References
    ----------
    Hodrick, R.J, and E. C. Prescott. 1980. "Postwar U.S. Business Cycles: An
        Empricial Investigation." `Carnegie Mellon University discussion
        paper no. 451`.
    Ravn, M.O and H. Uhlig. 2002. "Notes On Adjusted the Hodrick-Prescott
        Filter for the Frequency of Observations." `The Review of Economics and
        Statistics`, 84(2), 371-80.
    """
    X = np.asarray(X, float)
    if X.ndim > 1:
        X = X.squeeze()
    nobs = len(X)
    I = speye(nobs,nobs)
    offsets = np.array([0,1,2])
    data = np.repeat([[1.],[-2.],[1.]], nobs, axis=1)
    K = dia_matrix((data, offsets), shape=(nobs-2,nobs))

    import scipy
    if (X.dtype != np.dtype('<f8') and
            int(scipy.__version__[:3].split('.')[1]) < 11):
        #scipy umfpack bug on Big Endian machines, will be fixed in 0.11
        use_umfpack = False
    else:
        use_umfpack = True

    if scipy.__version__[:3] == '0.7':
        #doesn't have use_umfpack option
        #will be broken on big-endian machines with scipy 0.7 and umfpack
        trend = spsolve(I+lamb*K.T.dot(K), X)
    else:
        trend = spsolve(I+lamb*K.T.dot(K), X, use_umfpack=use_umfpack)
    cycle = X-trend
    return cycle, trend
