"""
Miscellaneous utility code for VAR estimation
"""

import numpy as np
import scipy.stats as stats

def interpret_data(data, names):
    """
    Convert passed data structure to form required by VAR estimation classes

    Parameters
    ----------

    Returns
    -------

    """
    if isinstance(data, np.ndarray):
        provided_names = data.dtype.names

        # structured array type
        if provided_names:
            if names is None:
                names = provided_names
            else:
                assert(len(names) == len(provided_names))

            Y = struct_to_ndarray(data)
        else:
            Y = data
    else:
        raise Exception('cannot handle other input types at the moment')

    return Y, names

def struct_to_ndarray(arr):
    return arr.view((float, len(arr.dtype.names)))

def parse_data(path):
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

def get_logdet(m):
    from scikits.statsmodels.tools.compatibility import np_slogdet
    logdet = np_slogdet(m)

    if logdet[0] == -1:
        raise ValueError("Matrix is not positive definite")
    elif logdet[0] == 0:
        raise ValueError("Matrix is singluar")
    else:
        logdet = logdet[1]

    return logdet

def norm_signif_level(alpha=0.05):
    return stats.norm.ppf(1 - alpha / 2)


def acf_to_acorr(acf):
    diag = np.diag(acf[0])
    # numpy broadcasting sufficient
    return acf / np.sqrt(np.outer(diag, diag))


def varsim(coefs, intercept, sig_u, steps=100, initvalues=None):
    """
    Simulate simple VAR(p) process with known coefficients, intercept, white
    noise covariance, etc.
    """
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
