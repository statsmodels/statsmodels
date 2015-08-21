"""
Tests for the generic MLEModel

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
import os
import re

import warnings
from statsmodels.tsa.statespace import sarimax, kalman_filter
from statsmodels.tsa.statespace.mlemodel import MLEModel, MLEResultsWrapper
from numpy.testing import assert_almost_equal, assert_equal, assert_allclose, assert_raises
from nose.exc import SkipTest
from .results import results_sarimax

current_path = os.path.dirname(os.path.abspath(__file__))

# Basic kwargs
kwargs = {
    'k_states': 1, 'design': [[1]], 'transition': [[1]],
    'selection': [[1]], 'state_cov': [[1]],
    'initialization': 'approximate_diffuse'
}


def get_dummy_mod(fit=True, pandas=False):
    # This tests time-varying parameters regression when in fact the parameters
    # are not time-varying, and in fact the regression fit is perfect
    endog = np.arange(100)*1.0
    exog = 2*endog

    if pandas:
        index = pd.date_range('1960-01-01', periods=100, freq='MS')
        endog = pd.TimeSeries(endog, index=index)
        exog = pd.TimeSeries(exog, index=index)

    mod = sarimax.SARIMAX(endog, exog=exog, order=(0,0,0), time_varying_regression=True, mle_regression=False)

    if fit:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = mod.fit(disp=-1)
    else:
        res = None
    
    return mod, res


def test_fit_misc():
    true = results_sarimax.wpi1_stationary
    endog = np.diff(true['data'])[1:]

    mod = sarimax.SARIMAX(endog, order=(1,0,1), trend='c')

    # Test optim_hessian={'opg','oim','cs'}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res1 = mod.fit(method='ncg', disp=True, optim_hessian='opg')
        res2 = mod.fit(method='ncg', disp=True, optim_hessian='oim')
        res3 = mod.fit(method='ncg', disp=True, optim_hessian='cs')
        assert_raises(NotImplementedError, mod.fit, method='ncg', disp=False, optim_hessian='a')
    # Check that the Hessians broadly result in the same optimum
    assert_allclose(res1.llf, res2.llf, rtol=1e-2)
    assert_allclose(res1.llf, res3.llf, rtol=1e-2)

    # Test return_params=True
    mod, _ = get_dummy_mod(fit=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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

    # Smoke test for each of the covariance types
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mod.fit(res.params, disp=-1, cov_type='cs')
        assert_equal(res.cov_type, 'cs')
        assert_equal(res.cov_kwds['description'], 'Covariance matrix calculated using numerical (complex-step) differentiation.')
        res = mod.fit(res.params, disp=-1, cov_type='delta')
        assert_equal(res.cov_type, 'delta')
        assert_equal(res.cov_kwds['description'], 'Covariance matrix calculated using numerical differentiation and the delta method (method of propagation of errors) applied to the parameter transformation function.')
        res = mod.fit(res.params, disp=-1, cov_type='oim')
        assert_equal(res.cov_type, 'oim')
        assert_equal(res.cov_kwds['description'], 'Covariance matrix calculated using the observed information matrix described in Harvey (1989).')
        res = mod.fit(res.params, disp=-1, cov_type='opg')
        assert_equal(res.cov_type, 'opg')
        assert_equal(res.cov_kwds['description'], 'Covariance matrix calculated using the outer product of gradients.')
        res = mod.fit(res.params, disp=-1, cov_type='robust')
        assert_equal(res.cov_type, 'robust')
        assert_equal(res.cov_kwds['description'], 'Quasi-maximum likelihood covariance matrix used for robustness to some misspecifications; calculated using the observed information matrix described in Harvey (1989).')
        res = mod.fit(res.params, disp=-1, cov_type='robust_oim')
        assert_equal(res.cov_type, 'robust_oim')
        assert_equal(res.cov_kwds['description'], 'Quasi-maximum likelihood covariance matrix used for robustness to some misspecifications; calculated using the observed information matrix described in Harvey (1989).')
        res = mod.fit(res.params, disp=-1, cov_type='robust_cs')
        assert_equal(res.cov_type, 'robust_cs')
        assert_equal(res.cov_kwds['description'], 'Quasi-maximum likelihood covariance matrix used for robustness to some misspecifications; calculated using numerical (complex-step) differentiation.')
        assert_raises(NotImplementedError, mod.fit, res.params, disp=-1, cov_type='invalid_cov_type')


def test_transform():
    # The transforms in MLEModel are noops
    mod = MLEModel([1,2], **kwargs)

    # Test direct transform, untransform
    assert_allclose(mod.transform_params([2, 3]), [2, 3])
    assert_allclose(mod.untransform_params([2, 3]), [2, 3])    

    # Smoke test for transformation in `filter`, `update`, `loglike`,
    # `loglikeobs`
    mod.filter([], transformed=False)
    mod.update([], transformed=False)
    mod.loglike([], transformed=False)
    mod.loglikeobs([], transformed=False)

    # Note that mod is an SARIMAX instance, and the two parameters are
    # variances
    mod, _ = get_dummy_mod(fit=False)

    # Test direct transform, untransform
    assert_allclose(mod.transform_params([2, 3]), [4, 9])
    assert_allclose(mod.untransform_params([4, 9]), [2, 3])

    # Test transformation in `filter`
    res = mod.filter([2, 3], transformed=True)
    assert_allclose(res.params, [2, 3])

    res = mod.filter([2, 3], transformed=False)
    assert_allclose(res.params, [4, 9])


def test_filter():
    endog = np.array([1., 2.])
    mod = MLEModel(endog, **kwargs)

    # Test return of ssm object
    res = mod.filter([], return_ssm=True)
    assert_equal(isinstance(res, kalman_filter.FilterResults), True)

    # Test return of full results object
    res = mod.filter([])
    assert_equal(isinstance(res, MLEResultsWrapper), True)
    assert_equal(res.cov_type, 'opg')

    # Test return of full results object, specific covariance type
    res = mod.filter([], cov_type='oim')
    assert_equal(isinstance(res, MLEResultsWrapper), True)
    assert_equal(res.cov_type, 'oim')


def test_params():
    mod = MLEModel([1,2], **kwargs)

    # By default start_params raises NotImplementedError
    assert_raises(NotImplementedError, lambda: mod.start_params)
    # But param names are by default an empty array
    assert_equal(mod.param_names, [])

    # We can set them in the object if we want
    mod._start_params = [1]
    mod._param_names = ['a']

    assert_equal(mod.start_params, [1])
    assert_equal(mod.param_names, ['a'])


def check_results(pandas):
    mod, res = get_dummy_mod(pandas=pandas)

    # Test fitted values
    assert_almost_equal(res.fittedvalues[2:], mod.endog[2:].squeeze())

    # Test residuals
    assert_almost_equal(res.resid[2:], np.zeros(mod.nobs-2))

    # Test loglikelihood_burn
    assert_equal(res.loglikelihood_burn, 1)


def test_results(pandas=False):
    check_results(pandas=False)
    check_results(pandas=True)


def test_predict():
    dates = pd.date_range(start='1980-01-01', end='1981-01-01', freq='AS')
    endog = pd.TimeSeries([1,2], index=dates)
    mod = MLEModel(endog, **kwargs)
    res = mod.filter([])

    # Test that predict with start=None, end=None does prediction with full
    # dataset
    assert_equal(res.predict().shape, (mod.nobs,))

    # Test a string value to the dynamic option
    assert_allclose(res.predict(dynamic='1981-01-01'), res.predict())

    # Test an invalid date string value to the dynamic option
    assert_raises(ValueError, res.predict, dynamic='1982-01-01')


def test_forecast():
    # Numpy
    mod = MLEModel([1,2], **kwargs)
    res = mod.filter([])
    assert_allclose(res.forecast(steps=10), np.ones((10,)) * 2)

    # Pandas
    index = pd.date_range('1960-01-01', periods=2, freq='MS')
    mod = MLEModel(pd.Series([1,2], index=index), **kwargs)
    res = mod.filter([])
    assert_allclose(res.forecast(steps=10), np.ones((10,)) * 2)


def test_summary():
    dates = pd.date_range(start='1980-01-01', end='1981-01-01', freq='AS')
    endog = pd.TimeSeries([1,2], index=dates)
    mod = MLEModel(endog, **kwargs)
    res = mod.filter([])

    # Get the summary
    txt = str(res.summary())

    # Test res.summary when the model has dates
    assert_equal(re.search('Sample:\s+01-01-1980', txt) is not None, True)
    assert_equal(re.search('\s+- 01-01-1981', txt) is not None, True)

    # Test res.summary when `model_name` was not provided
    assert_equal(re.search('Model:\s+MLEModel', txt) is not None, True)


def check_endog(endog, nobs=2, k_endog=1, **kwargs):
    # create the model
    mod = MLEModel(endog, **kwargs)
    # the data directly available in the model is the Statsmodels version of
    # the data; it should be 2-dim, C-contiguous, long-shaped:
    # (nobs, k_endog) == (2, 1)
    assert_equal(mod.endog.ndim, 2)
    assert_equal(mod.endog.flags['C_CONTIGUOUS'], True)
    assert_equal(mod.endog.shape, (nobs, k_endog))
    # the data in the `ssm` object is the state space version of the data; it
    # should be 2-dim, F-contiguous, wide-shaped (k_endog, nobs) == (1, 2)
    # and it should share data with mod.endog
    assert_equal(mod.ssm.endog.ndim, 2)
    assert_equal(mod.ssm.endog.flags['F_CONTIGUOUS'], True)
    assert_equal(mod.ssm.endog.shape, (k_endog, nobs))
    assert_equal(mod.ssm.endog.base is mod.endog, True)

    return mod

def test_basic_endog():
    # Test various types of basic python endog inputs (e.g. lists, scalars...)

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

    # Check behavior of the link maintained between passed `endog` and
    # `mod.endog` arrays
    endog = np.array([1., 2.])
    mod = MLEModel(endog, **kwargs)
    assert_equal(mod.endog.base is not mod.data.orig_endog, True)
    assert_equal(mod.endog.base is not endog, True)
    assert_equal(mod.data.orig_endog.base is not endog, True)
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
    assert_equal(endog.ndim, 1)
    assert_equal(endog.flags['C_CONTIGUOUS'], True)
    assert_equal(endog.flags['F_CONTIGUOUS'], True)
    assert_equal(endog.shape, (2,))
    mod = check_endog(endog, **kwargs)
    mod.filter([])

    # Example : 2-dim array, C-contiguous, long-shaped: (nobs, k_endog)
    endog = np.array([1., 2.]).reshape(2, 1)
    assert_equal(endog.ndim, 2)
    assert_equal(endog.flags['C_CONTIGUOUS'], True)
    assert_equal(endog.flags['F_CONTIGUOUS'], False)
    assert_equal(endog.shape, (2, 1))
    mod = check_endog(endog, **kwargs)
    mod.filter([])

    # Example : 2-dim array, C-contiguous, wide-shaped: (k_endog, nobs)
    endog = np.array([1., 2.]).reshape(1, 2)
    assert_equal(endog.ndim, 2)
    assert_equal(endog.flags['C_CONTIGUOUS'], True)
    assert_equal(endog.flags['F_CONTIGUOUS'], False)
    assert_equal(endog.shape, (1, 2))
    # raises error because arrays are always interpreted as
    # (nobs, k_endog), which means that k_endog=2 is incompatibile with shape
    # of design matrix (1, 1)
    assert_raises(ValueError, check_endog, endog, **kwargs)

    # Example : 2-dim array, F-contiguous, long-shaped (nobs, k_endog)
    endog = np.array([1., 2.]).reshape(1, 2).transpose()
    assert_equal(endog.ndim, 2)
    assert_equal(endog.flags['C_CONTIGUOUS'], False)
    assert_equal(endog.flags['F_CONTIGUOUS'], True)
    assert_equal(endog.shape, (2, 1))
    mod = check_endog(endog, **kwargs)
    mod.filter([])

    # Example : 2-dim array, F-contiguous, wide-shaped (k_endog, nobs)
    endog = np.array([1., 2.]).reshape(2, 1).transpose()
    assert_equal(endog.ndim, 2)
    assert_equal(endog.flags['C_CONTIGUOUS'], False)
    assert_equal(endog.flags['F_CONTIGUOUS'], True)
    assert_equal(endog.shape, (1, 2))
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
    kwargs2 = {
        'k_states': 1, 'design': [[1], [0.]], 'obs_cov': [[1, 0], [0, 1]],
        'transition': [[1]], 'selection': [[1]], 'state_cov': [[1]],
        'initialization': 'approximate_diffuse'
    }
    endog = np.array([[1., 2.], [3., 4.]])
    mod = check_endog(endog, k_endog=2, **kwargs2)
    mod.filter([])

def test_pandas_endog():
    # Test various types of pandas endog inputs (e.g. TimeSeries, etc.)

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
    assert_equal(mod.endog.base is not mod.data.orig_endog, True)
    assert_equal(mod.endog.base is not endog, True)
    assert_equal(mod.data.orig_endog.values.base is not endog, True)
    endog.iloc[0, 0] = 2
    # there is no link to mod.endog
    assert_equal(mod.endog, np.r_[1, 2].reshape(2,1))
    # there remains a link to mod.data.orig_endog
    assert_allclose(mod.data.orig_endog, endog)

    # Example : pandas.DataFrame with 2 columns
    # Update kwargs for k_endog=2
    kwargs2 = {
        'k_states': 1, 'design': [[1], [0.]], 'obs_cov': [[1, 0], [0, 1]],
        'transition': [[1]], 'selection': [[1]], 'state_cov': [[1]],
        'initialization': 'approximate_diffuse'
    }
    endog = pd.DataFrame({'a': [1., 2.], 'b': [3., 4.]}, index=dates)
    mod = check_endog(endog, k_endog=2, **kwargs2)
    mod.filter([])
