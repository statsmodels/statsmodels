"""
Tests for the generic MLEModel

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
import os

import warnings
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.statespace.mlemodel import MLEModel
from numpy.testing import assert_almost_equal, assert_equal, assert_raises
from nose.exc import SkipTest

current_path = os.path.dirname(os.path.abspath(__file__))

def get_dummy_mod():
    # This tests time-varying parameters regression when in fact the parameters
    # are not time-varying, and in fact the regression fit is perfect
    endog = np.arange(100)*1.0
    exog = 2*endog

    mod = sarimax.SARIMAX(endog, exog=exog, order=(0,0,0), time_varying_regression=True, mle_regression=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Regression test to make sure bfgs_tune works
        res = mod.fit(disp=-1)
    
    return mod, res

def test_fit_misc():
    mod, res = get_dummy_mod()

    # Test bfgs_tune=True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Regression test to make sure bfgs_tune works
        res = mod.fit(disp=-1, bfgs_tune=True)
    assert_almost_equal(res.params, [0,0], 7)

    # Test return_params=True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Regression test to make sure bfgs_tune works
        res_params = mod.fit(disp=-1, return_params=True)

    assert_almost_equal(res_params, [0,0], 7)

def test_loglike_misc():
    mod, res = get_dummy_mod()

    # Test average_loglike=False
    loglike = mod.loglike()
    average_loglike = mod.loglike(average_loglike=True)

    assert_equal(loglike/mod.nobs, average_loglike)

def test_score_misc():
    mod, res = get_dummy_mod()

    # Test that the score function works
    mod.score(mod.params)
    mod.score(mod.params, initial_state=res.initial_state, initial_state_cov=res.initial_state_cov)

def test_from_formula():
    assert_raises(NotImplementedError, lambda: MLEModel.from_formula(1,2,3))

def test_cov_params():
    mod, res = get_dummy_mod()

    assert_almost_equal(res.cov_params(), res.cov_params_delta)

def test_results():
    mod, res = get_dummy_mod()

    # Test fitted values
    assert_almost_equal(res.fittedvalues()[2:], mod.endog[2:])

    # Test residuals
    assert_almost_equal(res.resid()[0,2:], np.zeros(mod.nobs-2))

    