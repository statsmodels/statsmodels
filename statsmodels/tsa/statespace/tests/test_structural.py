"""
Tests for structural time series models

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
import os

import warnings
from statsmodels.datasets import macrodata
from statsmodels.tsa.statespace import structural
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.tests.results import results_structural
from statsmodels.tools import add_constant
from numpy.testing import assert_equal, assert_almost_equal, assert_raises, assert_allclose
from nose.exc import SkipTest


try:
    import matplotlib.pyplot as plt
    have_matplotlib = True
except ImportError:
    have_matplotlib = False


dta = macrodata.load_pandas().data
dta.index = pd.date_range(start='1959-01-01', end='2009-07-01', freq='QS')


def run_ucm(name):
    true = getattr(results_structural, name)

    for model in true['models']:
        kwargs = model.copy()
        kwargs.update(true['kwargs'])

        # Make a copy of the data
        values = dta.copy()

        freq = kwargs.pop('freq', None)
        if freq is not None:
            values.index = pd.date_range(start='1959-01-01', periods=len(dta),
                                  freq=freq)

        # Test pandas exog
        if 'exog' in kwargs:
            # Default value here is pd.Series object
            exog = np.log(values['realgdp'])

            # Also allow a check with a 1-dim numpy array
            if kwargs['exog'] == 'numpy':
                exog = exog.values.squeeze()

            kwargs['exog'] = exog

        # Create the model
        mod = UnobservedComponents(values['unemp'], **kwargs)

        # Smoke test for starting parameters, untransform, transform
        # Also test that transform and untransform are inverses
        mod.start_params
        assert_allclose(mod.start_params, mod.transform_params(mod.untransform_params(mod.start_params)))

        # Fit the model at the true parameters
        res_true = mod.filter(true['params'])

        # Check that the cycle bounds were computed correctly
        freqstr = freq[0] if freq is not None else values.index.freqstr[0]
        if 'cycle_period_bounds' in kwargs:
            cycle_period_bounds = kwargs['cycle_period_bounds']
        elif freqstr == 'A':
            cycle_period_bounds = (1.5, 12)
        elif freqstr == 'Q':
            cycle_period_bounds = (1.5*4, 12*4)
        elif freqstr == 'M':
            cycle_period_bounds = (1.5*12, 12*12)
        else:
            # If we have no information on data frequency, require the
            # cycle frequency to be between 0 and pi
            cycle_period_bounds = (2, np.inf)

        # Test that the cycle frequency bound is correct
        assert_equal(mod.cycle_frequency_bound,
            (2*np.pi / cycle_period_bounds[1],
             2*np.pi / cycle_period_bounds[0])
        )

        # Test that the likelihood is correct
        rtol = true.get('rtol', 1e-7)
        atol = true.get('atol', 0)
        assert_allclose(res_true.llf, true['llf'], rtol=rtol, atol=atol)

        # Smoke test for plot_components
        if have_matplotlib:
            fig = res_true.plot_components()
            plt.close(fig)

        # Now fit the model via MLE
        with warnings.catch_warnings(record=True) as w:
            res = mod.fit(disp=-1)
            # If we found a higher likelihood, no problem; otherwise check
            # that we're very close to that found by R
            if res.llf <= true['llf']:
                assert_allclose(res.llf, true['llf'], rtol=1e-4)

            # Smoke test for summary
            res.summary()


def test_irregular():
    run_ucm('irregular')


def test_fixed_intercept():
    # Clear warnings
    structural.__warningregistry__ = {}

    with warnings.catch_warnings(record=True) as w:
        run_ucm('fixed_intercept')
        message = ("Specified model does not contain a stochastic element;"
                   " irregular component added.")
        assert_equal(str(w[0].message), message)


def test_deterministic_constant():
    run_ucm('deterministic_constant')


def test_random_walk():
    run_ucm('random_walk')


def test_local_level():
    run_ucm('local_level')


def test_fixed_slope():
    run_ucm('fixed_slope')


def test_fixed_slope():
    # Clear warnings
    structural.__warningregistry__ = {}

    with warnings.catch_warnings(record=True) as w:
        run_ucm('fixed_slope')
        message = ("Specified model does not contain a stochastic element;"
                   " irregular component added.")
        assert_equal(str(w[0].message), message)


def test_deterministic_trend():
    run_ucm('deterministic_trend')


def test_random_walk_with_drift():
    run_ucm('random_walk_with_drift')


def test_local_linear_deterministic_trend():
    run_ucm('local_linear_deterministic_trend')


def test_local_linear_trend():
    run_ucm('local_linear_trend')


def test_smooth_trend():
    run_ucm('smooth_trend')


def test_random_trend():
    run_ucm('random_trend')


def test_cycle():
    run_ucm('cycle')


def test_seasonal():
    run_ucm('seasonal')


def test_reg():
    run_ucm('reg')


def test_rtrend_ar1():
    run_ucm('rtrend_ar1')


def test_lltrend_cycle_seasonal_reg_ar1():
    run_ucm('lltrend_cycle_seasonal_reg_ar1')


def test_mle_reg():
    endog = np.arange(100)*1.0
    exog = endog*2
    # Make the fit not-quite-perfect
    endog[::2] += 0.01
    endog[1::2] -= 0.01

    with warnings.catch_warnings(record=True) as w:
        mod1 = UnobservedComponents(endog, irregular=True, exog=exog, mle_regression=False)
        res1 = mod1.fit(disp=-1)

        mod2 = UnobservedComponents(endog, irregular=True, exog=exog, mle_regression=True)
        res2 = mod2.fit(disp=-1)

    assert_allclose(res1.regression_coefficients.filtered[0, -1], 0.5, atol=1e-5)
    assert_allclose(res2.params[1], 0.5, atol=1e-5)


def test_specifications():
    # Clear warnings
    structural.__warningregistry__ = {}

    endog = [1, 2]

    # Test that when nothing specified, a warning is issued and the model that
    # is fit is one with irregular=True and nothing else.
    with warnings.catch_warnings(record=True) as w:
        mod = UnobservedComponents(endog)

        message = ("Specified model does not contain a stochastic element;"
                   " irregular component added.")
        assert_equal(str(w[0].message), message)
        assert_equal(mod.trend_specification, 'irregular')

    # Test an invalid string trend specification
    assert_raises(ValueError, UnobservedComponents, endog, 'invalid spec')

    # Test that if a trend component is specified without a level component,
    # a warning is issued and a deterministic level component is added
    with warnings.catch_warnings(record=True) as w:
        mod = UnobservedComponents(endog, trend=True, irregular=True)
        message = ("Trend component specified without level component;"
                   " deterministic level component added.")
        assert_equal(str(w[0].message), message)
        assert_equal(mod.trend_specification, 'deterministic trend')

    # Test that if a string specification is provided, a warning is issued if
    # the boolean attributes are also specified
    trend_attributes = ['irregular', 'trend', 'stochastic_level',
                        'stochastic_trend']
    for attribute in trend_attributes:
        with warnings.catch_warnings(record=True) as w:
            kwargs = {attribute: True}
            mod = UnobservedComponents(endog, 'deterministic trend', **kwargs)

            message = ("Value of `%s` may be overridden when the trend"
                       " component is specified using a model string."
                       % attribute)
            assert_equal(str(w[0].message), message)

    # Test that a seasonal with period less than two is invalid
    assert_raises(ValueError, UnobservedComponents, endog, seasonal=1)

def test_start_params():
    # Test that the behavior is correct for multiple exogenous and / or
    # autoregressive components

    # Parameters
    nobs = int(1e4)
    beta = np.r_[10, -2]
    phi = np.r_[0.5, 0.1]

    # Generate data
    np.random.seed(1234)
    exog = np.c_[np.ones(nobs), np.arange(nobs)*1.0]
    eps = np.random.normal(size=nobs)
    endog = np.zeros(nobs+2)
    for t in range(1, nobs):
        endog[t+1] = phi[0] * endog[t] + phi[1] * endog[t-1] + eps[t]
    endog = endog[2:]
    endog += np.dot(exog, beta)
    
    # Now just test that the starting parameters are approximately what they
    # ought to be (could make this arbitrarily precise by increasing nobs,
    # but that would slow down the test for no real gain)
    mod = UnobservedComponents(endog, exog=exog, autoregressive=2)
    assert_allclose(mod.start_params, [1., 0.5, 0.1, 10, -2], atol=1e-1)

def test_forecast():
    endog = np.arange(50) + 10
    exog = np.arange(50)

    mod = UnobservedComponents(endog, exog=exog, level='dconstant', seasonal=4)
    res = mod.smooth([1e-15, 0, 1])

    actual = res.forecast(10, exog=np.arange(50,60)[:,np.newaxis])
    desired = np.arange(50, 60) + 10
    assert_allclose(actual, desired)


def test_misc_exog():
    # Tests for missing data
    nobs = 20
    k_endog = 1
    np.random.seed(1208)
    endog = np.random.normal(size=(nobs, k_endog))
    endog[:4, 0] = np.nan
    exog1 = np.random.normal(size=(nobs, 1))
    exog2 = np.random.normal(size=(nobs, 2))

    index = pd.date_range('1970-01-01', freq='QS', periods=nobs)
    endog_pd = pd.DataFrame(endog, index=index)
    exog1_pd = pd.Series(exog1.squeeze(), index=index)
    exog2_pd = pd.DataFrame(exog2, index=index)

    models = [
        UnobservedComponents(endog, 'llevel', exog=exog1),
        UnobservedComponents(endog, 'llevel', exog=exog2),
        UnobservedComponents(endog, 'llevel', exog=exog2),
        UnobservedComponents(endog_pd, 'llevel', exog=exog1_pd),
        UnobservedComponents(endog_pd, 'llevel', exog=exog2_pd),
        UnobservedComponents(endog_pd, 'llevel', exog=exog2_pd),
    ]

    for mod in models:
        # Smoke tests
        mod.start_params
        res = mod.fit(disp=False)
        res.summary()
        res.predict()
        res.predict(dynamic=True)
        res.get_prediction()

        oos_exog = np.random.normal(size=(1, mod.k_exog))
        res.forecast(steps=1, exog=oos_exog)
        res.get_forecast(steps=1, exog=oos_exog)

        # Smoke tests for invalid exog
        oos_exog = np.random.normal(size=(1))
        assert_raises(ValueError, res.forecast, steps=1, exog=oos_exog)

        oos_exog = np.random.normal(size=(2, mod.k_exog))
        assert_raises(ValueError, res.forecast, steps=1, exog=oos_exog)

        oos_exog = np.random.normal(size=(1, mod.k_exog + 1))
        assert_raises(ValueError, res.forecast, steps=1, exog=oos_exog)

    # Test invalid model specifications
    assert_raises(ValueError, UnobservedComponents, endog, 'llevel',
                  exog=np.zeros((10, 4)))


def test_predict_custom_index():
    np.random.seed(328423)
    endog = pd.DataFrame(np.random.normal(size=50))
    mod = structural.UnobservedComponents(endog, 'llevel')
    res = mod.smooth(mod.start_params)
    out = res.predict(start=1, end=1, index=['a'])
    assert_equal(out.index.equals(pd.Index(['a'])), True)
