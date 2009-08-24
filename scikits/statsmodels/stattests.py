"""
Statistical tests to be used in conjuction with the smodels
"""

from scipy import stats
import numpy as np
from numpy.testing.decorators import setastest # doesn't work for some reason
from numpy.testing import dec
from nose.tools import nottest  # can't get this to work either

#TODO: these are pretty straightforward but they should be tested
def durbin_watson(resids):
    """
    Calculates the Durbin-Waston statistic

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

#@setastest(False)
#@nottest(omni_norm_test)
#@nottest   # should pass func?
# neither of these seems to exclude this from the tests?
@dec.skipif(True, "This is not a test!")
def omni_norm_test(resids, axis=0):
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
