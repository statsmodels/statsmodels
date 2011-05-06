import numpy as np
from scipy import percentile

## Univariate Rule of Thumb Bandwidths ##
def bw_scott(x, axis=0):
    """
    Scott's Rule of Thumb

    Parameter
    ---------
    x : array-like
        Array for which to get the bandwidth

    Returns
    -------
    bw : float
        The estimate of the bandwidth

    Notes
    -----
    Returns 1.059 * A * n ** (-1/5.)

    A = min(std(x, ddof=1), IQR/1.34)
    IQR = np.subtract.reduce(scipy.percentile(x, [75,25]))

    References
    ---------- ::

    Scott, D.W. (1992) `Multivariate Density Estimation: Theory, Practice, and
        Visualization.`
    """
    n = len(x)
    IQR = np.subtract.reduce(percentile(x, [75,25], axis=axis), axis=axis)/1.34
    A = np.minimum(np.std(x, axis=axis, ddof=1), IQR)
    return 1.059 * A * n ** -.2

def bw_silverman(x, axis=0):
    """
    Silverman's Rule of Thumb

    Parameter
    ---------
    x : array-like
        Array for which to get the bandwidth

    Returns
    -------
    bw : float
        The estimate of the bandwidth

    Notes
    -----
    Returns .9 * A * n ** (-1/5.)

    A = min(std(x, ddof=1), IQR/1.34)
    IQR = np.subtract.reduce(scipy.percentile(x, [75,25]))

    References
    ---------- ::

    Silverman, B.W. (1986) `Density Estimation.`
    """
    n = len(x)
    IQR = np.subtract.reduce(percentile(x, [75,25], axis=axis), axis=axis)/1.34
    A = np.minimum(np.std(x, axis=axis, ddof=1), IQR)
    return .9 * A * n ** -.2

## Plug-In Methods ##

## Least Squares Cross-Validation ##
