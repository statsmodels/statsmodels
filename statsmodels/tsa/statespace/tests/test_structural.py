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
from .results import results_structural
from statsmodels.tools import add_constant
from numpy.testing import assert_equal, assert_almost_equal, assert_raises, assert_allclose
from nose.exc import SkipTest


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

        mod = UnobservedComponents(values['unemp'], **kwargs)
        res = mod.filter(true['params'])

        # Check that the cycle bounds were computed correctly
        freqstr = freq[0] if freq is not None else values.index.freqstr[0]
        if freqstr == 'A':
            cycle_period_bounds = (1.5, 12)
        elif freqstr == 'Q':
            cycle_period_bounds = (1.5*4, 12*4)
        elif freqstr == 'M':
            cycle_period_bounds = (1.5*12, 12*12)
        else:
            # If we have no information on data frequency, require the
            # cycle frequency to be between 0 and pi
            cycle_period_bounds = (2, np.inf)

        assert_equal(mod.cycle_frequency_bound,
            (2*np.pi / cycle_period_bounds[1],
             2*np.pi / cycle_period_bounds[0])
        )

        rtol = true.get('rtol', 1e-7)
        atol = true.get('atol', 0)
        assert_allclose(res.llf, true['llf'], rtol=rtol, atol=atol)


def test_irregular():
    run_ucm('irregular')


def test_fixed_intercept():
    warnings.simplefilter("always")
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
    warnings.simplefilter("always")
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


def test_specifications():
    endog = [1, 2]

    # Test that when nothing specified, a warning is issued and the model that
    # is fit is one with irregular=True and nothing else.
    warnings.simplefilter("always")
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
            print(w)
            assert_equal(str(w[0].message), message)

