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
from .results import results_structural
from statsmodels.tools import add_constant
from numpy.testing import assert_almost_equal, assert_raises, assert_allclose
from nose.exc import SkipTest


dta = macrodata.load_pandas().data
dta.index = pd.date_range(start='1959-01-01', end='2009-07-01', freq='QS')


def test_ntrend(model='model'):
    true = getattr(results_structural, 'ntrend')

    kwargs = true[model].copy()
    kwargs.update(true['kwargs'])

    mod = structural.UnobservedComponents(dta['unemp'], **kwargs)
    mod.update(true['params'])
    res = mod.filter()

    assert_allclose(res.llf, true['llf'])


def test_ntrend_alt():
    test_ntrend('alt_model')


def test_dconstant(model='model'):
    true = getattr(results_structural, 'dconstant')

    kwargs = true[model].copy()
    kwargs.update(true['kwargs'])

    mod = structural.UnobservedComponents(dta['unemp'], **kwargs)
    mod.update(true['params'])
    res = mod.filter()

    assert_allclose(res.llf, true['llf'])


def test_dconstant_alt():
    test_dconstant('alt_model')


def test_llevel(model='model'):
    true = getattr(results_structural, 'llevel')

    kwargs = true[model].copy()
    kwargs.update(true['kwargs'])

    mod = structural.UnobservedComponents(dta['unemp'], **kwargs)
    mod.update(true['params'])
    res = mod.filter()

    assert_allclose(res.llf, true['llf'])


def test_llevel_alt():
    test_llevel('alt_model')


def test_rwalk(model='model'):
    true = getattr(results_structural, 'rwalk')

    kwargs = true[model].copy()
    kwargs.update(true['kwargs'])

    mod = structural.UnobservedComponents(dta['unemp'], **kwargs)
    mod.update(true['params'])
    res = mod.filter()

    assert_allclose(res.llf, true['llf'])


def test_rwalk_alt():
    test_rwalk('alt_model')


def test_dtrend(model='model'):
    true = getattr(results_structural, 'dtrend')

    kwargs = true[model].copy()
    kwargs.update(true['kwargs'])

    mod = structural.UnobservedComponents(dta['unemp'], **kwargs)
    mod.update(true['params'])
    res = mod.filter()

    assert_allclose(res.llf, true['llf'])


def test_dtrend_alt():
    test_dtrend('alt_model')


def test_lldtrend(model='model'):
    true = getattr(results_structural, 'lldtrend')

    kwargs = true[model].copy()
    kwargs.update(true['kwargs'])

    mod = structural.UnobservedComponents(dta['unemp'], **kwargs)
    mod.update(true['params'])
    res = mod.filter()

    assert_allclose(res.llf, true['llf'])


def test_lldtrend_alt():
    test_lldtrend('alt_model')


def test_rwdrift(model='model'):
    true = getattr(results_structural, 'rwdrift')

    kwargs = true[model].copy()
    kwargs.update(true['kwargs'])

    mod = structural.UnobservedComponents(dta['unemp'], **kwargs)
    mod.update(true['params'])
    res = mod.filter()

    assert_allclose(res.llf, true['llf'])


def test_rwdrift_alt():
    test_rwdrift('alt_model')


def test_lltrend(model='model'):
    true = getattr(results_structural, 'lltrend')

    kwargs = true[model].copy()
    kwargs.update(true['kwargs'])

    mod = structural.UnobservedComponents(dta['unemp'], **kwargs)
    mod.update(true['params'])
    res = mod.filter()

    assert_allclose(res.llf, true['llf'])


def test_lltrend_alt():
    test_lltrend('alt_model')


def test_strend(model='model'):
    true = getattr(results_structural, 'strend')

    kwargs = true[model].copy()
    kwargs.update(true['kwargs'])

    mod = structural.UnobservedComponents(dta['unemp'], **kwargs)
    mod.update(true['params'])
    res = mod.filter()

    assert_allclose(res.llf, true['llf'])


def test_strend_alt():
    test_strend('alt_model')


def test_rtrend(model='model'):
    true = getattr(results_structural, 'rtrend')

    kwargs = true[model].copy()
    kwargs.update(true['kwargs'])

    mod = structural.UnobservedComponents(dta['unemp'], **kwargs)
    mod.update(true['params'])
    res = mod.filter()

    assert_allclose(res.llf, true['llf'])


def test_rtrend_alt():
    test_rtrend('alt_model')


def test_cycle():
    true = getattr(results_structural, 'cycle')

    kwargs = true['model'].copy()
    kwargs.update(true['kwargs'])

    mod = structural.UnobservedComponents(dta['unemp'], **kwargs)
    mod.update(true['params'])
    res = mod.filter()

    assert_allclose(res.llf, true['llf'])


def test_seasonal():
    true = getattr(results_structural, 'seasonal')

    kwargs = true['model'].copy()
    kwargs.update(true['kwargs'])

    mod = structural.UnobservedComponents(dta['unemp'], **kwargs)
    mod.update(true['params'])
    res = mod.filter()

    assert_allclose(res.llf, true['llf'], atol=1e-3)


def test_reg():
    true = getattr(results_structural, 'reg')

    kwargs = true['model'].copy()
    kwargs.update(true['kwargs'])

    kwargs['exog'] = np.log(dta['realgdp'])

    mod = structural.UnobservedComponents(dta['unemp'], **kwargs)
    mod.update(true['params'])
    res = mod.filter()

    assert_allclose(res.llf, true['llf'])


def test_rtrend_ar1(model='model'):
    true = getattr(results_structural, 'rtrend_ar1')

    kwargs = true[model].copy()
    kwargs.update(true['kwargs'])

    mod = structural.UnobservedComponents(dta['unemp'], **kwargs)
    mod.update(true['params'])
    res = mod.filter()

    assert_allclose(res.llf, true['llf'])


def test_lltrend_cycle_seasonal_reg_ar1(model='model'):
    true = getattr(results_structural, 'lltrend_cycle_seasonal_reg_ar1')

    kwargs = true[model].copy()
    kwargs.update(true['kwargs'])

    kwargs['exog'] = np.log(dta['realgdp'])

    mod = structural.UnobservedComponents(dta['unemp'], **kwargs)
    mod.update(true['params'])
    res = mod.filter()

    assert_allclose(res.llf, true['llf'])

def test_lltrend_cycle_seasonal_reg_ar1_alt():
    test_lltrend_cycle_seasonal_reg_ar1('alt_model')
