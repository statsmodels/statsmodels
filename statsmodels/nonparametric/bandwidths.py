from __future__ import division

import numpy as np
from scipy.stats import scoreatpercentile as sap
from statsmodels.sandbox.nonparametric import kernels

import KernelFunctions as kf

kernel_func=dict(ordered=kf.WangRyzin, unordered=kf.AitchisonAitken,
                 continuous=kf.Epanechnikov)
# NOTE: Here we don't care so much about the kernel as we do about the
#   type of the variable (ordered, unordered, continuous). This is due to
#   Racine who argues that:
#   "It turns out that a range of kernel functions
#   result in estimators having similar relative efficiencies so one could
#   choose the kernel based on computational considerations...Unlike choosing
#   a kernel function, however, choosing an appropriate bandwidth is a
#   curcial aspect of sound nonparametric analysis."
#   See Racine, J. Nonparametric Econometrics: A primer (2008).
#   Foundations and Trends in Econometrics Vol.3, No.1, 1-88


#   
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

class LeaveOneOut(object):
    # Written by Skipper
    """
    Generator to give leave one out views on X

    Parameters
    ----------
    X : array-like
        1d array

    Examples
    --------
    >>> X = np.arange(10)
    >>> loo = LeaveOneOut(X)
    >>> for x in loo:
    ...    print x

    Notes
    -----
    A little lighter weight than sklearn LOO. We don't need test index.
    Also passes views on X, not the index.
    """
    def __init__(self, X):
        self.X = np.asarray(X)

    def __iter__(self):
        X = self.X
        n = len(X)
        for i in xrange(n):
            index = np.ones(n, dtype=np.bool)
            index[i] = False
            yield X[index]


def likelihood_cv(h,X,var_type):
    """
    Returns the leave one out log likelihood

    Parameters
    ----------
    h : float
        The bandwdith paremteter value (smoothing parameter)
    x : arraylike
        The training data
    var_type : str
        Defines the type of variable. Can take continuous, ordered or unordered

    Returns
    -------
    L : float
        The (negative of the) log likelihood value

    References
    ---------
    Nonparametric econometrics : theory and practice / Qi Li and Jeffrey Scott Racine.
     (p.16)
    """

    #TODO: Extend this to handle the categorical kernels
    var_type=var_type.lower()
    mykern=kernel_func[var_type]

    n=len(X)
    LOO=LeaveOneOut(X)
    L=0
    i=0
    for X_j in LOO:
        f_i = sum(mykern(h,-X_j,-X[i]))*1/((n-1)*h)    
        i += 1
        L += np.log(f_i)
    return -L

from scipy.optimize import fmin
def bw_likelihood_cv(x,var_type):
    """
    Returns the bandwidth parameter which maximizes the leave one out likelihood
    """
    
    h0 = 1.0
    bw = fmin (likelihood_cv, h0,(x,var_type))
    return bw


## Univariate Rule of Thumb Bandwidths ##
def bw_scott(x, kernel=None):
    """
    Scott's Rule of Thumb

    Parameters
    ----------
    x : array-like
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
    x : array-like
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
    x : array-like
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


def select_bandwidth(x, bw, kernel):
    """
    Selects bandwidth for a selection rule bw

    this is a wrapper around existing bandwidth selection rules

    Parameters
    ----------
    x : array-like
        Array for which to get the bandwidth
    bw : string
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

# Interpolate docstring to plugin supported bandwidths
select_bandwidth.__doc__ %=  (", ".join(sorted(bandwidth_funcs.keys())),)
