"""
Tests of save / load / remove_data state space functionality.
"""

from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
import os

from statsmodels import datasets
from statsmodels.tsa.statespace import (sarimax, structural, varmax,
                                        dynamic_factor)
from numpy.testing import assert_allclose
macrodata = datasets.macrodata.load_pandas().data


def test_sarimax():
    mod = sarimax.SARIMAX(macrodata['realgdp'].values, order=(4, 1, 0))
    res = mod.smooth(mod.start_params)
    res.summary()
    res.save('test_save_sarimax.p')
    res2 = sarimax.SARIMAXResults.load('test_save_sarimax.p')
    assert_allclose(res.params, res2.params)
    assert_allclose(res.bse, res2.bse)
    assert_allclose(res.llf, res2.llf)
    os.unlink('test_save_sarimax.p')


def test_structural():
    mod = structural.UnobservedComponents(
        macrodata['realgdp'].values, 'llevel')
    res = mod.smooth(mod.start_params)
    res.summary()
    res.save('test_save_structural.p')
    res2 = structural.UnobservedComponentsResults.load(
        'test_save_structural.p')
    assert_allclose(res.params, res2.params)
    assert_allclose(res.bse, res2.bse)
    assert_allclose(res.llf, res2.llf)
    os.unlink('test_save_structural.p')


def test_dynamic_factor():
    mod = dynamic_factor.DynamicFactor(
        macrodata[['realgdp', 'realcons']].diff().iloc[1:].values, k_factors=1,
        factor_order=1)
    res = mod.smooth(mod.start_params)
    res.summary()
    res.save('test_save_dynamic_factor.p')
    res2 = dynamic_factor.DynamicFactorResults.load(
        'test_save_dynamic_factor.p')
    assert_allclose(res.params, res2.params)
    assert_allclose(res.bse, res2.bse)
    assert_allclose(res.llf, res2.llf)
    os.unlink('test_save_dynamic_factor.p')


def test_varmax():
    mod = varmax.VARMAX(
        macrodata[['realgdp', 'realcons']].diff().iloc[1:].values,
        order=(1, 0))
    res = mod.smooth(mod.start_params)
    res.summary()
    res.save('test_save_varmax.p')
    res2 = varmax.VARMAXResults.load(
        'test_save_varmax.p')
    assert_allclose(res.params, res2.params)
    assert_allclose(res.bse, res2.bse)
    assert_allclose(res.llf, res2.llf)
    os.unlink('test_save_varmax.p')
