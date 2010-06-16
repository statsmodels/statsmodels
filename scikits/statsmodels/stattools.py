"""
Statistical tests to be used in conjunction with the models

Notes
-----
These functions haven't been formally tested.
"""

from scipy import stats
import numpy as np
from numpy.testing.decorators import setastest # doesn't work for some reason
from numpy.testing import dec

#TODO: these are pretty straightforward but they should be tested
def durbin_watson(resids):
    """
    Calculates the Durbin-Watson statistic

    Parameters
    -----------
    resids : array-like

    Returns
        --------
    Durbin Watson statistic.  This is defined as
    sum_(t=2)^(T)((e_t - e_(t-1))^(2))/sum_(t=1)^(T)e_t^(2)
    """
    diff_resids = np.diff(resids,1)
    dw = np.dot(diff_resids,diff_resids) / \
        np.dot(resids,resids);
    return dw

def omni_normtest(resids, axis=0):
    """
    Omnibus test for normality

    Parameters
    -----------
    resid : array-like
    axis : int, optional
        Default is 0

    Returns
    -------
    Chi^2 score, two-tail probability
    """
    return stats.normaltest(resids, axis=0)

def jarque_bera(resids):
    """
    Calculate residual skewness, kurtosis, and do the JB test for normality

    Parameters
    -----------
    resids : array-like

    Returns
    -------
    JB, JBpv, skew, kurtosis

    JB = n/6*(S^2 + (K-3)^2/4)

    JBpv is the Chi^2 two-tail probability value

    skew is the measure of skewness

    kurtosis is the measure of kurtosis

    """
    resids = np.asarray(resids)
    # Calculate residual skewness and kurtosis
    skew = stats.skew(resids)
    kurtosis = 3 + stats.kurtosis(resids)

    # Calculate the Jarque-Bera test for normality
    JB = (resids.shape[0]/6) * (skew**2 + (1/4)*(kurtosis-3)**2)
    JBpv = stats.chi2.sf(JB,2);

    return JB, JBpv, skew, kurtosis

def conditionnum(exog):
    """
    Returns the condition number of an exogenous design array.

    The given array is first normalized (except for a constant term assumed
    to be in the last column) so that each column is a unit length vector
    then the condition number of dot(norm_exog.T,norm_exog) is returned.
    The condition number is defined as the square root of the ratio of the
    largest eigenvalue to the smallest eigenvalue.

    Parameters
    ----------
    exog : array-like
        An exogenous design matrix with the final column assumed to be a
        the constant

    Returns
    -------
    Condition numbers.
    """
    exog = np.array(exog)
    if exog.ndim is 1:
        exog = exog[:,None]
    numvar = exog.shape[1]
    norm_exog = np.ones_like(exog)
    for i in range(numvar):
        norm_exog[:,i] = exog[:,i]/np.linalg.norm(exog[:,i])
    xtx = np.dot(norm_exog.T,norm_exog)
    eigs = np.linalg.eigvals(xtx)
    return np.sqrt(eigs.max()/eigs.min())
