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
    #TODO: change to exception in summary branch and catch in summary()
    #behavior changed between scipy 0.9 and 0.10
    resids = np.asarray(resids)
    n = resids.shape[axis]
    if n < 8:
        return np.nan, np.nan
        raise ValueError(
            "skewtest is not valid with less than 8 observations; %i samples"
            " were given." % int(n))

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
    JB = (resids.shape[0]/6.) * (skew**2 + (1/4.)*(kurtosis-3)**2)
    JBpv = stats.chi2.sf(JB,2);

    return JB, JBpv, skew, kurtosis

