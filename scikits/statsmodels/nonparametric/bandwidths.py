import numpy as np
from scipy import percentile
#from scipy.stats import norm

def _select_sigma(X, axis=0):
    """
    Returns the smaller of std(X, ddof=1) or normalized IQR(X)

    References
    ----------
    Silverman (1986) p.47
    """
#    normalize = norm.ppf(.75) - norm.ppf(.25)
    normalize = 1.349
    IQR = np.subtract.reduce(percentile(X, [75,25],
                             axis=axis), axis=axis)/normalize
    return np.minimum(np.std(X, axis=axis, ddof=1), IQR)


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

    A = min(std(x, ddof=1), IQR/1.349)
    IQR = np.subtract.reduce(scipy.percentile(x, [75,25]))

    References
    ---------- ::

    Scott, D.W. (1992) `Multivariate Density Estimation: Theory, Practice, and
        Visualization.`
    """
    A = _select_sigma(x, axis)
    n = len(x)
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

    A = min(std(x, ddof=1), IQR/1.349)
    IQR = np.subtract.reduce(scipy.percentile(x, [75,25]))

    References
    ---------- ::

    Silverman, B.W. (1986) `Density Estimation.`
    """
    A = _select_sigma(x, axis)
    n = len(x)
    return .9 * A * n ** -.2

## Plug-In Methods ##

## Least Squares Cross-Validation ##
