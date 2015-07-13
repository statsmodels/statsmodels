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
from numpy.testing import assert_almost_equal, assert_equal, assert_allclose, assert_raises
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

def test_score_misc():
    mod, res = get_dummy_mod()

    # Test that the score function works
    mod.score(res.params)

def test_from_formula():
    assert_raises(NotImplementedError, lambda: MLEModel.from_formula(1,2,3))

def test_cov_params():
    mod, res = get_dummy_mod()

    assert_almost_equal(res.cov_params(), res.cov_params_delta)

def test_results():
    mod, res = get_dummy_mod()

    # Test fitted values
    assert_almost_equal(res.fittedvalues()[2:], mod.ssm.endog[2:])

    # Test residuals
    assert_almost_equal(res.resid()[0,2:], np.zeros(mod.nobs-2))

def check_endog(endog, nobs=2, k_endog=1, **kwargs):
    # create the model
    mod = MLEModel(endog, **kwargs)
    # the data directly available in the model is the Statsmodels version of
    # the data; it should be 2-dim, C-contiguous, long-shaped:
    # (nobs, k_endog) == (2, 1)
    assert(mod.endog.ndim == 2)
    assert(mod.endog.flags['C_CONTIGUOUS'] == True)
    assert(mod.endog.shape == (nobs, k_endog))
    # the data in the `ssm` object is the state space version of the data; it
    # should be 2-dim, F-contiguous, wide-shaped (k_endog, nobs) == (1, 2)
    # and it should share data with mod.endog
    assert(mod.ssm.endog.ndim == 2)
    assert(mod.ssm.endog.flags['F_CONTIGUOUS'] == True)
    assert(mod.ssm.endog.shape == (k_endog, nobs))
    assert(mod.ssm.endog.base is mod.endog)

    return mod

def test_basic_endog():
    # Test various types of basic python endog inputs (e.g. lists, scalars...)

    # Setup state space matrices / options to initialize test models with
    kwargs = {
        'k_states': 1, 'design': [[1]], 'transition': [[1]],
        'selection': [[1]], 'state_cov': [[1]],
        'initialization': 'approximate_diffuse'
    }

    # Check cannot call with non-array-like
    # fails due to checks in Statsmodels base classes
    assert_raises(ValueError, MLEModel, endog=1, k_states=1)
    assert_raises(ValueError, MLEModel, endog='a', k_states=1)
    assert_raises(ValueError, MLEModel, endog=True, k_states=1)

    # Check behavior with different types
    mod = MLEModel([1], **kwargs)
    res = mod.filter([])
    assert_equal(res.filter_results.endog, [[1]])

    mod = MLEModel([1.], **kwargs)
    res = mod.filter([])
    assert_equal(res.filter_results.endog, [[1]])

    mod = MLEModel([True], **kwargs)
    res = mod.filter([])
    assert_equal(res.filter_results.endog, [[1]])

    mod = MLEModel(['a'], **kwargs)
    # raises error due to inability coerce string to numeric
    assert_raises(ValueError, mod.filter, [])

    # Check that a different iterable tpyes give the expected result
    endog = [1.,2.]
    mod = check_endog(endog, **kwargs)
    mod.filter([])

    endog = [[1.],[2.]]
    mod = check_endog(endog, **kwargs)
    mod.filter([])

    endog = (1.,2.)
    mod = check_endog(endog, **kwargs)
    mod.filter([])

def test_numpy_endog():
    # Test various types of numpy endog inputs

    # Setup state space matrices / options to initialize test models with
    kwargs = {
        'k_states': 1, 'design': [[1]], 'transition': [[1]],
        'selection': [[1]], 'state_cov': [[1]],
        'initialization': 'approximate_diffuse'
    }

    # Check behavior of the link maintained between passed `endog` and
    # `mod.endog` arrays
    endog = np.array([1., 2.])
    mod = MLEModel(endog, **kwargs)
    assert(mod.endog.base is not mod.data.orig_endog)
    assert(mod.endog.base is not endog)
    assert(mod.data.orig_endog.base is not endog)
    endog[0] = 2
    # there is no link to mod.endog
    assert_equal(mod.endog, np.r_[1, 2].reshape(2,1))
    # there remains a link to mod.data.orig_endog
    assert_equal(mod.data.orig_endog, endog)

    # Check behavior with different memory layouts / shapes

    # Example  (failure): 0-dim array
    endog = np.array(1.)
    # raises error due to len(endog) failing in Statsmodels base classes
    assert_raises(TypeError, check_endog, endog, **kwargs)

    # Example : 1-dim array, both C- and F-contiguous, length 2
    endog = np.array([1.,2.])
    assert(endog.ndim == 1)
    assert(endog.flags['C_CONTIGUOUS'] == True)
    assert(endog.flags['F_CONTIGUOUS'] == True)
    assert(endog.shape == (2,))
    mod = check_endog(endog, **kwargs)
    mod.filter([])

    # Example : 2-dim array, C-contiguous, long-shaped: (nobs, k_endog)
    endog = np.array([1., 2.]).reshape(2, 1)
    assert(endog.ndim == 2)
    assert(endog.flags['C_CONTIGUOUS'] == True)
    assert(endog.flags['F_CONTIGUOUS'] == False)
    assert(endog.shape == (2, 1))
    mod = check_endog(endog, **kwargs)
    mod.filter([])

    # Example : 2-dim array, C-contiguous, wide-shaped: (k_endog, nobs)
    endog = np.array([1., 2.]).reshape(1, 2)
    assert(endog.ndim == 2)
    assert(endog.flags['C_CONTIGUOUS'] == True)
    assert(endog.flags['F_CONTIGUOUS'] == False)
    assert(endog.shape == (1, 2))
    # raises error because arrays are always interpreted as
    # (nobs, k_endog), which means that k_endog=2 is incompatibile with shape
    # of design matrix (1, 1)
    assert_raises(ValueError, check_endog, endog, **kwargs)

    # Example : 2-dim array, F-contiguous, long-shaped (nobs, k_endog)
    endog = np.array([1., 2.]).reshape(1, 2).transpose()
    assert(endog.ndim == 2)
    assert(endog.flags['C_CONTIGUOUS'] == False)
    assert(endog.flags['F_CONTIGUOUS'] == True)
    assert(endog.shape == (2, 1))
    mod = check_endog(endog, **kwargs)
    mod.filter([])

    # Example : 2-dim array, F-contiguous, wide-shaped (k_endog, nobs)
    endog = np.array([1., 2.]).reshape(2, 1).transpose()
    assert(endog.ndim == 2)
    assert(endog.flags['C_CONTIGUOUS'] == False)
    assert(endog.flags['F_CONTIGUOUS'] == True)
    assert(endog.shape == (1, 2))
    # raises error because arrays are always interpreted as
    # (nobs, k_endog), which means that k_endog=2 is incompatibile with shape
    # of design matrix (1, 1)
    assert_raises(ValueError, check_endog, endog, **kwargs)

    # Example  (failure): 3-dim array
    endog = np.array([1., 2.]).reshape(2, 1, 1)
    # raises error due to direct ndim check in Statsmodels base classes
    assert_raises(ValueError, check_endog, endog, **kwargs)

    # Example : np.array with 2 columns
    # Update kwargs for k_endog=2
    kwargs = {
        'k_states': 1, 'design': [[1], [0.]], 'obs_cov': [[1, 0], [0, 1]],
        'transition': [[1]], 'selection': [[1]], 'state_cov': [[1]],
        'initialization': 'approximate_diffuse'
    }
    endog = np.array([[1., 2.], [3., 4.]])
    mod = check_endog(endog, k_endog=2, **kwargs)
    mod.filter([])

def test_pandas_endog():
    # Test various types of pandas endog inputs (e.g. TimeSeries, etc.)

    # Setup state space matrices / options to initialize test models with
    kwargs = {
        'k_states': 1, 'design': [[1]], 'transition': [[1]],
        'selection': [[1]], 'state_cov': [[1]],
        'initialization': 'approximate_diffuse'
    }

    # Example (failure): pandas.Series, no dates
    endog = pd.Series([1., 2.])
    # raises error due to no dates
    assert_raises(ValueError, check_endog, endog, **kwargs)

    # Example : pandas.Series
    dates = pd.date_range(start='1980-01-01', end='1981-01-01', freq='AS')
    endog = pd.Series([1., 2.], index=dates)
    mod = check_endog(endog, **kwargs)
    mod.filter([])

    # Example : pandas.Series, string datatype
    endog = pd.Series(['a'], index=dates)
    # raises error due to direct type casting check in Statsmodels base classes
    assert_raises(ValueError, check_endog, endog, **kwargs)

    # Example : pandas.TimeSeries
    endog = pd.TimeSeries([1., 2.], index=dates)
    mod = check_endog(endog, **kwargs)
    mod.filter([])

    # Example : pandas.DataFrame with 1 column
    endog = pd.DataFrame({'a': [1., 2.]}, index=dates)
    mod = check_endog(endog, **kwargs)
    mod.filter([])

    # Example (failure): pandas.DataFrame with 2 columns
    endog = pd.DataFrame({'a': [1., 2.], 'b': [3., 4.]}, index=dates)
    # raises error because 2-columns means k_endog=2, but the design matrix
    # set in **kwargs is shaped (1,1)
    assert_raises(ValueError, check_endog, endog, **kwargs)

    # Check behavior of the link maintained between passed `endog` and
    # `mod.endog` arrays
    endog = pd.DataFrame({'a': [1., 2.]}, index=dates)
    mod = check_endog(endog, **kwargs)
    assert(mod.endog.base is not mod.data.orig_endog)
    assert(mod.endog.base is not endog)
    assert(mod.data.orig_endog.values.base is not endog)
    endog.iloc[0, 0] = 2
    # there is no link to mod.endog
    assert_equal(mod.endog, np.r_[1, 2].reshape(2,1))
    # there remains a link to mod.data.orig_endog
    assert_allclose(mod.data.orig_endog, endog)

    # Example : pandas.DataFrame with 2 columns
    # Update kwargs for k_endog=2
    kwargs = {
        'k_states': 1, 'design': [[1], [0.]], 'obs_cov': [[1, 0], [0, 1]],
        'transition': [[1]], 'selection': [[1]], 'state_cov': [[1]],
        'initialization': 'approximate_diffuse'
    }
    endog = pd.DataFrame({'a': [1., 2.], 'b': [3., 4.]}, index=dates)
    mod = check_endog(endog, k_endog=2, **kwargs)
    mod.filter([])


    assert(False)
