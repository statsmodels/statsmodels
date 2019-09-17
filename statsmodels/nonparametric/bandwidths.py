import numpy as np
from scipy.stats import scoreatpercentile as sap

from statsmodels.compat.pandas import Substitution
from statsmodels.sandbox.nonparametric import kernels

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
def bw_scott(x, kernel=None):
    """
    Scott's Rule of Thumb

    Parameters
    ----------
    x : array_like
        Array for which to get the bandwidth
    kernel : CustomKernel object
        Unused

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
    return 1.059 * A * n ** (-0.2)

def bw_silverman(x, kernel=None):
    """
    Silverman's Rule of Thumb

    Parameters
    ----------
    x : array_like
        Array for which to get the bandwidth
    kernel : CustomKernel object
        Unused

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
    return .9 * A * n ** (-0.2)


def bw_normal_reference(x, kernel=kernels.Gaussian):
    """
    Plug-in bandwidth with kernel specific constant based on normal reference.

    This bandwidth minimizes the mean integrated square error if the true
    distribution is the normal. This choice is an appropriate bandwidth for
    single peaked distributions that are similar to the normal distribution.

    Parameters
    ----------
    x : array_like
        Array for which to get the bandwidth
    kernel : CustomKernel object
        Used to calculate the constant for the plug-in bandwidth.

    Returns
    -------
    bw : float
        The estimate of the bandwidth

    Notes
    -----
    Returns C * A * n ** (-1/5.) where ::

       A = min(std(x, ddof=1), IQR/1.349)
       IQR = np.subtract.reduce(np.percentile(x, [75,25]))
       C = constant from Hansen (2009)

    When using a Gaussian kernel this is equivalent to the 'scott' bandwidth up
    to two decimal places. This is the accuracy to which the 'scott' constant is
    specified.

    References
    ----------

    Silverman, B.W. (1986) `Density Estimation.`
    Hansen, B.E. (2009) `Lecture Notes on Nonparametrics.`
    """
    C = kernel.normal_reference_constant
    A = _select_sigma(x)
    n = len(x)
    return C * A * n ** (-0.2)

## Plug-In Methods ##

## Least Squares Cross-Validation ##

## Helper Functions ##

bandwidth_funcs = {
    "scott": bw_scott,
    "silverman": bw_silverman,
    "normal_reference": bw_normal_reference,
}


@Substitution(", ".join(sorted(bandwidth_funcs.keys())))
def select_bandwidth(x, bw, kernel):
    """
    Selects bandwidth for a selection rule bw

    this is a wrapper around existing bandwidth selection rules

    Parameters
    ----------
    x : array_like
        Array for which to get the bandwidth
    bw : str
        name of bandwidth selection rule, currently supported are:
        %s
    kernel : not used yet

    Returns
    -------
    bw : float
        The estimate of the bandwidth

    """
    bw = bw.lower()
    if bw not in bandwidth_funcs:
        raise ValueError("Bandwidth %s not understood" % bw)
#TODO: uncomment checks when we have non-rule of thumb bandwidths for diff. kernels
#    if kernel == "gauss":
    return bandwidth_funcs[bw](x, kernel)
#    else:
#        raise ValueError("Only Gaussian Kernels are currently supported")
