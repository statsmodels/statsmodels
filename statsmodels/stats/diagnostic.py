# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats

# collect some imports of verified (at least one example) functions
from statsmodels.sandbox.stats.diagnostic import (
    acorr_ljungbox, breaks_cusumolsresid, breaks_hansen, CompareCox, CompareJ,
    compare_cox, compare_j, het_breuschpagan, HetGoldfeldQuandt,
    het_goldfeldquandt, het_arch,
    recursive_olsresiduals, acorr_breusch_godfrey,
    linear_harvey_collier, linear_rainbow, linear_lm,
    spec_white, unitroot_adf)

from ._lilliefors import (  # noqa:F401
    kstest_fit, lilliefors, kstest_normal, kstest_exponential)
from ._adnorm import normal_ad  # noqa:F401


# -----------------------------------------------------------------
# Misc Helpers

class ResultsStore(object):
    # TODO: implement in something like tools.tools?
    def __str__(self):
        return self._str


# -----------------------------------------------------------------
# Tests for Heteroscedasticity

def het_white(resid, exog):
    """
    White's Lagrange Multiplier Test for Heteroscedasticity

    Parameters
    ----------
    resid : array_like
        residuals, square of it is used as endogenous variable
    exog : array_like
        possible explanatory variables for variance, squares and interaction
        terms are included in the auxilliary regression.

    Returns
    -------
    lm : float
        lagrange multiplier statistic
    lm_pvalue :float
        p-value of lagrange multiplier test
    fvalue : float
        f-statistic of the hypothesis that the error variance does not depend
        on x. This is an alternative test variant not the original LM test.
    f_pvalue : float
        p-value for the f-statistic

    Notes
    -----
    Assumes x contains constant (for counting degrees of freedom)

    TODO: does f-statistic make sense? constant ?

    References
    ----------
    Greene section 11.4.1 5th edition p. 222
        now test statistic reproduces Greene 5th, example 11.3
    """
    x = np.asarray(exog)
    y = np.asarray(resid)
    if x.ndim == 1:
        raise ValueError('x should have constant and at least one '
                         'more variable')

    nobs, nvars0 = x.shape
    i0, i1 = np.triu_indices(nvars0)
    exog = x[:, i0] * x[:, i1]
    nobs, nvars = exog.shape
    assert nvars == nvars0 * (nvars0 - 1) / 2. + nvars0

    from statsmodels.api import OLS
    resols = OLS(y**2, exog).fit()
    fval = resols.fvalue
    fpval = resols.f_pvalue
    lm = nobs * resols.rsquared
    # Note: degrees of freedom for LM test is nvars minus constant
    #   degrees of freedom take possible reduced rank in exog into account
    #   df_model checks the rank to determine df
    lmpval = stats.chi2.sf(lm, resols.df_model)
    return lm, lmpval, fval, fpval


# -----------------------------------------------------------------
# Tests for Autocorrelation


# -----------------------------------------------------------------
# Tests for Structural Breaks


# -----------------------------------------------------------------
# Tests for Functional Form
