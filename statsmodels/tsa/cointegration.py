"""
Estimation and testing of cointegrated time series
"""
from __future__ import division

import numpy as np

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.unitroot import mackinnonp, mackinnoncrit


def coint(y1, y2, regression="c"):
    """
    This is a simple cointegration test. Uses unit-root test on residuals to
    test for cointegrated relationship

    See Hamilton (1994) 19.2

    Parameters
    ----------
    y1 : array_like, 1d
        first element in cointegrating vector
    y2 : array_like
        remaining elements in cointegrating vector
    c : str {'c'}
        Included in regression
        * 'c' : Constant

    Returns
    -------
    coint_t : float
        t-statistic of unit-root test on residuals
    pvalue : float
        MacKinnon's approximate p-value based on MacKinnon (1994)
    crit_value : dict
        Critical values for the test statistic at the 1 %, 5 %, and 10 %
        levels.

    Notes
    -----
    The Null hypothesis is that there is no cointegration, the alternative
    hypothesis is that there is cointegrating relationship. If the pvalue is
    small, below a critical size, then we can reject the hypothesis that there
    is no cointegrating relationship.

    P-values are obtained through regression surface approximation from
    MacKinnon 1994.

    References
    ----------
    MacKinnon, J.G. 1994.  "Approximate asymptotic distribution functions for
        unit-root and cointegration tests.  `Journal of Business and Economic
        Statistics` 12, 167-76.

    """
    regression = regression.lower()
    if regression not in ['c', 'nc', 'ct', 'ctt']:
        raise ValueError("regression option %s not understood") % regression
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    if regression == 'c':
        y2 = add_constant(y2, prepend=False)
    st1_resid = OLS(y1, y2).fit().resid  # stage one residuals
    lgresid_cons = add_constant(st1_resid[0:-1], prepend=False)
    uroot_reg = OLS(st1_resid[1:], lgresid_cons).fit()
    coint_t = (uroot_reg.params[0] - 1) / uroot_reg.bse[0]
    pvalue = mackinnonp(coint_t, regression="c", num_unit_roots=2)
    crit_value = mackinnoncrit(num_unit_roots=1, regression="c", nobs=len(y1))
    return coint_t, pvalue, crit_value
