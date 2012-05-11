import numpy as np
from scipy.stats import scoreatpercentile as sap

#from scipy.stats import norm

def _select_sigma(X):
    """
    Returns the smaller of std(X, ddof=1) or normalized IQR(X) over axis 0.

    References
    ----------
    Silverman (1986) p.47
    """
#    normalize = norm.ppf(.75) - norm.ppf(.25)
    normalize = 1.349
#    IQR = np.subtract.reduce(percentile(X, [75,25],
#                             axis=axis), axis=axis)/normalize
    IQR = (sap(X, 75) - sap(X, 25))/normalize
    return np.minimum(np.std(X, axis=0, ddof=1), IQR)


## Univariate Rule of Thumb Bandwidths ##
def bw_scott(x):
    """
    Scott's Rule of Thumb

    Parameters
    ----------
    x : array-like
        Array for which to get the bandwidth

    Returns
    -------
    bw : float
        The estimate of the bandwidth

    Notes
    -----
    Returns 1.059 * A * n ** (-1/5.) where ::

       A = min(std(x, ddof=1), IQR/1.349)
       IQR = np.subtract.reduce(np.percentile(x, [75,25]))

    References
    ----------

    Scott, D.W. (1992) Multivariate Density Estimation: Theory, Practice, and
        Visualization.
    """
    A = _select_sigma(x)
    n = len(x)
    return 1.059 * A * n ** -.2

def bw_silverman(x):
    """
    Silverman's Rule of Thumb

    Parameters
    ----------
    x : array-like
        Array for which to get the bandwidth

    Returns
    -------
    bw : float
        The estimate of the bandwidth

    Notes
    -----
    Returns .9 * A * n ** (-1/5.) where ::

       A = min(std(x, ddof=1), IQR/1.349)
       IQR = np.subtract.reduce(np.percentile(x, [75,25]))

    References
    ----------

    Silverman, B.W. (1986) `Density Estimation.`
    """
    A = _select_sigma(x)
    n = len(x)
    return .9 * A * n ** -.2

## Plug-In Methods ##

## Least Squares Cross-Validation ##

## Helper Functions ##

bandwidth_funcs = dict(scott=bw_scott,silverman=bw_silverman)

def select_bandwidth(x, bw, kernel):
    """
    Selects bandwidth for a selection rule bw

    this is a wrapper around existing bandwidth selection rules

    Parameters
    ----------
    x : array-like
        Array for which to get the bandwidth
    bw : string
        name of bandwidth selection rule, currently "scott" and "silverman"
        are supported
    kernel : not used yet

    Returns
    -------
    bw : float
        The estimate of the bandwidth

    """
    bw = bw.lower()
    if bw not in ["scott","silverman"]:
        raise ValueError("Bandwidth %s not understood" % bw)
#TODO: uncomment checks when we have non-rule of thumb bandwidths for diff. kernels
#    if kernel == "gauss":
    return bandwidth_funcs[bw](x)
#    else:
#        raise ValueError("Only Gaussian Kernels are currently supported")

