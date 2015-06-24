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
dta.index = pd.date_range(start='1959-01', end='2009-7', freq='QS')


def test_ntrend():
    true = getattr(results_structural, 'ntrend')

    kwargs = true['model'].copy()
    kwargs.update(true['kwargs'])

    mod = structural.UnobservedComponents(dta['unemp'], **kwargs)
    mod.update(true['params'])
    res = mod.filter()

    assert_allclose(res.llf, true['llf'])


def test_dconstant():
    true = getattr(results_structural, 'dconstant')

    kwargs = true['model'].copy()
    kwargs.update(true['kwargs'])

    mod = structural.UnobservedComponents(dta['unemp'], **kwargs)
    mod.update(true['params'])
    res = mod.filter()

    assert_allclose(res.llf, true['llf'])


def test_llevel():
    true = getattr(results_structural, 'llevel')

    kwargs = true['model'].copy()
    kwargs.update(true['kwargs'])

    mod = structural.UnobservedComponents(dta['unemp'], **kwargs)
    mod.update(true['params'])
    res = mod.filter()

    assert_allclose(res.llf, true['llf'])


def test_rwalk():
    true = getattr(results_structural, 'rwalk')

    kwargs = true['model'].copy()
    kwargs.update(true['kwargs'])

    mod = structural.UnobservedComponents(dta['unemp'], **kwargs)
    mod.update(true['params'])
    res = mod.filter()

    assert_allclose(res.llf, true['llf'])


def test_dtrend():
    true = getattr(results_structural, 'dtrend')

    kwargs = true['model'].copy()
    kwargs.update(true['kwargs'])

    mod = structural.UnobservedComponents(dta['unemp'], **kwargs)
    mod.update(true['params'])
    res = mod.filter()

    assert_allclose(res.llf, true['llf'])


def test_lldtrend():
    true = getattr(results_structural, 'lldtrend')

    kwargs = true['model'].copy()
    kwargs.update(true['kwargs'])

    mod = structural.UnobservedComponents(dta['unemp'], **kwargs)
    mod.update(true['params'])
    res = mod.filter()

    assert_allclose(res.llf, true['llf'])


def test_rwdrift():
    true = getattr(results_structural, 'rwdrift')

    kwargs = true['model'].copy()
    kwargs.update(true['kwargs'])

    mod = structural.UnobservedComponents(dta['unemp'], **kwargs)
    mod.update(true['params'])
    res = mod.filter()

    assert_allclose(res.llf, true['llf'])


def test_lltrend():
    true = getattr(results_structural, 'lltrend')

    kwargs = true['model'].copy()
    kwargs.update(true['kwargs'])

    mod = structural.UnobservedComponents(dta['unemp'], **kwargs)
    mod.update(true['params'])
    res = mod.filter()

    assert_allclose(res.llf, true['llf'])


def test_strend():
    true = getattr(results_structural, 'strend')

    kwargs = true['model'].copy()
    kwargs.update(true['kwargs'])

    mod = structural.UnobservedComponents(dta['unemp'], **kwargs)
    mod.update(true['params'])
    res = mod.filter()

    assert_allclose(res.llf, true['llf'])


def test_rtrend():
    true = getattr(results_structural, 'rtrend')

    kwargs = true['model'].copy()
    kwargs.update(true['kwargs'])

    mod = structural.UnobservedComponents(dta['unemp'], **kwargs)
    mod.update(true['params'])
    res = mod.filter()

    assert_allclose(res.llf, true['llf'])


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


def test_rtrend_ar1():
    true = getattr(results_structural, 'rtrend_ar1')

    kwargs = true['model'].copy()
    kwargs.update(true['kwargs'])

    mod = structural.UnobservedComponents(dta['unemp'], **kwargs)
    mod.update(true['params'])
    res = mod.filter()

    assert_allclose(res.llf, true['llf'])


def test_lltrend_cycle_seasonal_reg_ar1():
    true = getattr(results_structural, 'lltrend_cycle_seasonal_reg_ar1')

    kwargs = true['model'].copy()
    kwargs.update(true['kwargs'])

    kwargs['exog'] = np.log(dta['realgdp'])

    mod = structural.UnobservedComponents(dta['unemp'], **kwargs)
    mod.update(true['params'])
    res = mod.filter()

    assert_allclose(res.llf, true['llf'])
