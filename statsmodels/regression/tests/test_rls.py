"""
Tests for recursive least squares models

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
import os

import warnings
from statsmodels.datasets import macrodata
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.rls import RecursiveLS
from statsmodels.stats.diagnostic import recursive_olsresiduals
from statsmodels.tools import add_constant
from numpy.testing import assert_equal, assert_almost_equal, assert_raises, assert_allclose
from nose.exc import SkipTest

try:
    import matplotlib.pyplot as plt
    have_matplotlib = True
except ImportError:
    have_matplotlib = False


current_path = os.path.dirname(os.path.abspath(__file__))

results_R_path = 'results' + os.sep + 'results_rls_R.csv'
results_R = pd.read_csv(current_path + os.sep + results_R_path)

results_stata_path = 'results' + os.sep + 'results_rls_stata.csv'
results_stata = pd.read_csv(current_path + os.sep + results_stata_path)

dta = macrodata.load_pandas().data
dta.index = pd.date_range(start='1959-01-01', end='2009-07-01', freq='QS')

endog = dta['cpi']
exog = add_constant(dta['m1'])

def test_estimates():
    mod = RecursiveLS(endog, exog)
    res = mod.fit()

    # Test the RLS coefficient estimates against those from R (quantreg)
    # Due to initialization issues, we get more agreement as we get
    # farther from the initial values.
    assert_allclose(res.recursive_coefficients.filtered[:, 2:10].T,
                    results_R.ix[:7, ['beta1', 'beta2']], atol=1e-2, rtol=1e-3)
    assert_allclose(res.recursive_coefficients.filtered[:, 9:20].T,
                    results_R.ix[7:17, ['beta1', 'beta2']], atol=1e-3, rtol=1e-4)
    assert_allclose(res.recursive_coefficients.filtered[:, 19:].T,
                    results_R.ix[17:, ['beta1', 'beta2']], atol=1e-4, rtol=1e-4)

    # Test the RLS estimates against OLS estimates
    mod_ols = OLS(endog, exog)
    res_ols = mod_ols.fit()
    assert_allclose(res.params, res_ols.params)


def test_recursive_resid():
    mod = RecursiveLS(endog, exog)
    res = mod.fit()

    # Test the recursive residuals against those from R (strucchange)
    # Due to initialization issues, we get more agreement as we get
    # farther from the initial values.
    assert_allclose(res.recursive_resid[2:10].T,
                    results_R.ix[:7, 'rec_resid'], atol=1e-2, rtol=1e-3)
    assert_allclose(res.recursive_resid[9:20].T,
                    results_R.ix[7:17, 'rec_resid'], atol=1e-3, rtol=1e-4)
    assert_allclose(res.recursive_resid[19:].T,
                    results_R.ix[17:, 'rec_resid'], atol=1e-4, rtol=1e-4)

    # Test the RLS estimates against those from Stata (cusum6)
    assert_allclose(res.recursive_resid[3:],
                    results_stata.ix[3:, 'rr'], atol=1e-3)

    # Test the RLS estimates against statsmodels estimates
    mod_ols = OLS(endog, exog)
    res_ols = mod_ols.fit()
    desired_recursive_resid = recursive_olsresiduals(res_ols)[4][2:]
    assert_allclose(res.recursive_resid[2:], desired_recursive_resid,
                    atol=1e-4, rtol=1e-4)


def test_cusum():
    mod = RecursiveLS(endog, exog)
    res = mod.fit()

    # Test the cusum statistics against those from R (strucchange)
    # These values are not even close to ours, to Statas, or to the alternate
    # statsmodels values
    # assert_allclose(res.cusum, results_R['cusum'])

    # Test the cusum statistics against Stata (cusum6)
    # Note: cusum6 excludes the first 3 elements due to OLS initialization
    # whereas we exclude only the first 2. Also there are initialization
    # differences (as seen above in the recursive residuals).
    # Here we explicitly reverse engineer our cusum to match their to show the
    # equivalence
    llb = res.loglikelihood_burn
    cusum = res.cusum * np.std(res.recursive_resid[llb:], ddof=1)
    cusum -= res.recursive_resid[llb]
    cusum /= np.std(res.recursive_resid[llb+1:], ddof=1)
    cusum = cusum[1:]
    assert_allclose(cusum, results_stata.ix[3:, 'cusum'], atol=1e-3, rtol=1e-3)

    # Test the cusum statistics against statsmodels estimates
    mod_ols = OLS(endog, exog)
    res_ols = mod_ols.fit()
    desired_cusum = recursive_olsresiduals(res_ols)[-2][1:]
    assert_allclose(res.cusum, desired_cusum, atol=1e-4, rtol=1e-4)

    # Test the cusum bounds against Stata (cusum6)
    # Again note that cusum6 excludes the first 3 elements, so we need to
    # change the ddof and points.
    actual_bounds = res._cusum_significance_bounds(
        alpha=0.05, ddof=1, points=np.arange(llb+1, res.nobs))
    desired_bounds = results_stata.ix[3:, ['lw', 'uw']].T
    assert_allclose(actual_bounds, desired_bounds, atol=1e-4)

    # Test the cusum bounds against statsmodels
    actual_bounds = res._cusum_significance_bounds(
        alpha=0.05, ddof=0, points=np.arange(llb, res.nobs))
    desired_bounds = recursive_olsresiduals(res_ols)[-1]
    assert_allclose(actual_bounds, desired_bounds)


def test_stata():
    # Test the cusum and cusumsq statistics against Stata (cusum6)
    # Note that here we change the loglikelihood_burn variable to explicitly
    # excude the first 3 elements as in Stata, so we can compare directly
    mod = RecursiveLS(endog, exog, loglikelihood_burn=3)
    res = mod.fit()
    llb = res.loglikelihood_burn

    assert_allclose(res.recursive_resid[3:], results_stata.ix[3:, 'rr'],
                    atol=1e-4, rtol=1e-4)
    assert_allclose(res.cusum, results_stata.ix[3:, 'cusum'], atol=1e-4)
    assert_allclose(res.cusum_squares, results_stata.ix[3:, 'cusum2'],
                    atol=1e-4)

    actual_bounds = res._cusum_significance_bounds(
        alpha=0.05, ddof=0, points=np.arange(llb+1, res.nobs+1))
    desired_bounds = results_stata.ix[3:, ['lw', 'uw']].T
    assert_allclose(actual_bounds, desired_bounds, atol=1e-4)

    # Note: Stata uses a set of tabulated critical values whereas we use an
    # approximation formula, so this test is quite imprecise
    actual_bounds = res._cusum_squares_significance_bounds(
        alpha=0.05, points=np.arange(llb+1, res.nobs+1))
    desired_bounds = results_stata.ix[3:, ['lww', 'uww']].T
    assert_allclose(actual_bounds, desired_bounds, atol=1e-2)
