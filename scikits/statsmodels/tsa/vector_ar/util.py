"""
Miscellaneous utility code for VAR estimation
"""

import numpy as np
import scipy.stats as stats
import scipy.linalg as la

import scikits.statsmodels.tsa.tsatools as tsa
from scipy.linalg import cholesky

#-------------------------------------------------------------------------------
# Auxiliary functions for estimation
def get_var_endog(y, lags, trend='c'):
    """
    Make predictor matrix for VAR(p) process

    Z := (Z_0, ..., Z_T).T (T x Kp)
    Z_t = [1 y_t y_{t-1} ... y_{t - p + 1}] (Kp x 1)

    Ref: Lutkepohl p.70 (transposed)
    """
    nobs = len(y)
    # Ravel C order, need to put in descending order
    Z = np.array([y[t-lags : t][::-1].ravel() for t in xrange(lags, nobs)])

    # Add constant, trend, etc.
    if trend != 'nc':
        Z = tsa.add_trend(Z, prepend=True, trend=trend)

    return Z

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

    Example
    -------
    >>> make_lag_names(['foo', 'bar'], 2, 1)
    ['const', 'L1.foo', 'L1.bar', 'L2.foo', 'L2.bar']

    """
    lag_names = []

    # take care of lagged endogenous names
    for i in range(1, lag_order + 1):
        for name in names:
            lag_names.append('L'+str(i)+'.'+name)

    # handle the constant name
    if trendorder != 0:
        lag_names.insert(0, 'const')
    if trendorder > 1:
        lag_names.insert(0, 'trend')
    if trendorder > 2:
        lag_names.insert(0, 'trend**2')

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

    regex = re.compile('<(.*) (\w)([\d]+)>.*')
    lines = deque(open(path))

    to_skip = 0
    while '*/' not in lines.popleft():
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
        'Q' : dt.BQuarterEnd(),
        'M' : dt.BMonthEnd(),
        'A' : dt.BYearEnd()
    }

    # create an instance
    offset = offsets[freq]

    inc = offset * (start_point - 1)
    start_date = offset.rollforward(datetime(year, 1, 1)) + inc

    offset = offsets[freq]
    date_range = pandas.DateRange(start_date, offset=offset, periods=n)

    return data, date_range

def get_logdet(m):
    from scikits.statsmodels.tools.compatibility import np_slogdet
    logdet = np_slogdet(m)

    if logdet[0] == -1: # pragma: no cover
        raise ValueError("Matrix is not positive definite")
    elif logdet[0] == 0: # pragma: no cover
        raise ValueError("Matrix is singular")
    else:
        logdet = logdet[1]

    return logdet

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
    for t in xrange(p, steps):
        ygen = result[t]
        for j in xrange(p):
            ygen += np.dot(coefs[j], result[t-j-1])

    return result

def get_index(lst, name):
    try:
        result = lst.index(name)
    except Exception:
        if not isinstance(name, int):
            raise
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
    #check if symmetric
    eigva, W = la.eig(sym_array)
    k = np.argmax(eigva)
    return W, eigva, k
