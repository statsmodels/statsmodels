"""
Miscellaneous utility code for VAR estimation
"""
from statsmodels.compat.python import range, string_types, asbytes, long
import numpy as np
import scipy.stats as stats
import scipy.linalg.decomp as decomp

import statsmodels.tsa.tsatools as tsa

#------------------------------------------------------------------------------
# Auxiliary functions for estimation

def get_forecast_X(nobs, trend, lags, steps):
    """
    Returns X for forecasting steps forecasts from nobs.

    Parameters
    ----------
    nobs : int
        Starting value for forecasts
    trend : str ('nc', 'c', 'ct', 'ctt')
        Trend
    steps : int
        Number of forecasts produced

    Examples
    --------
    >>> get_forecast_X(200, 'ctt', 4, 4)
    array([[     1.,    201.,  40000.],
           [     1.,    202.,  40401.],
           [     1.,    203.,  40804.],
           [     1.,    204.,  41616.]])
    """
    trendorder = get_trendorder(trend)
    if trendorder == 0:
        return
    return np.fliplr(np.vander(np.arange(nobs + 1,
                                         nobs + steps + 1,
                                         dtype=float), trendorder))


def get_lagged_y(y, lags, trend='c', has_constant='skip'):
    """
    Make predictor matrix for VAR(p) process

    X := (X_0, ..., X_T).T (T x Kp)
    X_t = [1 y_t y_{t-1} ... y_{t - p + 1}] (Kp x 1)

    Ref: Lutkepohl p.70 (transposed)

    has_constant can be 'raise', 'add', or 'skip'. See add_constant.
    """
    nobs = len(y)
    # pad with zeros so trend is correct
    y = np.r_[np.zeros((lags, y.shape[1])), y]
    # Ravel C order, need to put in descending order
    X = np.array([y[t-lags : t][::-1].ravel() for t in range(lags,
                                                             nobs + lags)])

    # Add constant, trend, etc.
    if trend != 'nc':
        X = tsa.add_trend(X, prepend=True, trend=trend,
                          has_constant=has_constant)


    return X[lags:]

def get_trendorder(trend='c'):
    # Handle constant, etc.
    if trend == 'c':
        trendorder = 1
    elif trend == 'nc':
        trendorder = 0
    elif trend == 'ct':
        trendorder = 2
    elif trend == 'ctt':
        trendorder = 3
    return trendorder

def make_lag_names(names, lag_order, trendorder=1):
    """
    Produce list of lag-variable names. Constant / trends go at the beginning

    Examples
    --------
    >>> make_lag_names(['foo', 'bar'], 2, 1)
    ['const', 'L1.foo', 'L1.bar', 'L2.foo', 'L2.bar']

    """
    lag_names = []
    if isinstance(names, string_types):
        names = [names]

    # take care of lagged endogenous names
    for i in range(1, lag_order + 1):
        for name in names:
            if not isinstance(name, string_types):
                name = str(name) # will need consistent unicode handling
            lag_names.append('L'+str(i)+'.'+name)

    # handle the constant name
    if trendorder > 2:
        lag_names.insert(0, 'trend**2')
    if trendorder > 1:
        lag_names.insert(0, 'trend')
    if trendorder != 0:
        lag_names.insert(0, 'const')

    return lag_names

def comp_matrix(coefs):
    """
    Return compansion matrix for the VAR(1) representation for a VAR(p) process
    (companion form)

    A = [A_1 A_2 ... A_p-1 A_p
         I_K 0       0     0
         0   I_K ... 0     0
         0 ...       I_K   0]
    """
    p, k, k2 = coefs.shape
    assert(k == k2)

    kp = k * p

    result = np.zeros((kp, kp))
    result[:k] = np.concatenate(coefs, axis=1)

    # Set I_K matrices
    if p > 1:
        result[np.arange(k, kp), np.arange(kp-k)] = 1

    return result

#-------------------------------------------------------------------------------
# Miscellaneous stuff

def parse_lutkepohl_data(path): # pragma: no cover
    """
    Parse data files from Lutkepohl (2005) book

    Source for data files: www.jmulti.de
    """

    from collections import deque
    from datetime import datetime
    import pandas
    import pandas.core.datetools as dt
    import re

    regex = re.compile(asbytes('<(.*) (\w)([\d]+)>.*'))
    with open(path, 'rb') as f:
        lines = deque(f)

    to_skip = 0
    while asbytes('*/') not in lines.popleft():
        #while '*/' not in lines.popleft():
        to_skip += 1

    while True:
        to_skip += 1
        line = lines.popleft()
        m = regex.match(line)
        if m:
            year, freq, start_point = m.groups()
            break

    data = np.genfromtxt(path, names=True, skip_header=to_skip+1)

    n = len(data)

    # generate the corresponding date range (using pandas for now)
    start_point = int(start_point)
    year = int(year)

    offsets = {
        asbytes('Q') : dt.BQuarterEnd(),
        asbytes('M') : dt.BMonthEnd(),
        asbytes('A') : dt.BYearEnd()
    }

    # create an instance
    offset = offsets[freq]

    inc = offset * (start_point - 1)
    start_date = offset.rollforward(datetime(year, 1, 1)) + inc

    offset = offsets[freq]
    from pandas import DatetimeIndex   # pylint: disable=E0611
    date_range = DatetimeIndex(start=start_date, freq=offset, periods=n)

    return data, date_range


def get_logdet(m):
    from statsmodels.tools.linalg import logdet_symm
    return logdet_symm(m)


get_logdet = np.deprecate(get_logdet,
                          "statsmodels.tsa.vector_ar.util.get_logdet",
                          "statsmodels.tools.linalg.logdet_symm",
                          "get_logdet is deprecated and will be removed in "
                          "0.8.0")


def norm_signif_level(alpha=0.05):
    return stats.norm.ppf(1 - alpha / 2)


def acf_to_acorr(acf):
    diag = np.diag(acf[0])
    # numpy broadcasting sufficient
    return acf / np.sqrt(np.outer(diag, diag))


def varsim(coefs, intercept, sig_u, steps=100, initvalues=None, seed=None):
    """
    Simulate simple VAR(p) process with known coefficients, intercept, white
    noise covariance, etc.
    """
    if seed is not None:
        np.random.seed(seed=seed)
    from numpy.random import multivariate_normal as rmvnorm
    p, k, k = coefs.shape
    ugen = rmvnorm(np.zeros(len(sig_u)), sig_u, steps)
    result = np.zeros((steps, k))
    result[p:] = intercept + ugen[p:]

    # add in AR terms
    for t in range(p, steps):
        ygen = result[t]
        for j in range(p):
            ygen += np.dot(coefs[j], result[t-j-1])

    return result

def get_index(lst, name):
    try:
        result = lst.index(name)

    except ValueError:
        if not isinstance(name, (int, long)):
            raise ValueError("No variable named %s in endog_names" %name)
        result = name
    return result
    #method used repeatedly in Sims-Zha error bands
def eigval_decomp(sym_array):
    """
    Returns
    -------
    W: array of eigenvectors
    eigva: list of eigenvalues
    k: largest eigenvector
    """
    #check if symmetric, do not include shock period
    eigva, W = decomp.eig(sym_array, left=True, right=False)
    k = np.argmax(eigva)
    return W, eigva, k

def vech(A):
    """
    Simple vech operator
    Returns
    -------
    vechvec: vector of all elements on and below diagonal
    """

    length=A.shape[1]
    vechvec=[]
    for i in range(length):
        b=i
        while b < length:
            vechvec.append(A[b,i])
            b=b+1
    vechvec=np.asarray(vechvec)
    return vechvec

