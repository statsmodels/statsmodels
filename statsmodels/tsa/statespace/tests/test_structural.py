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

def run_ucm(name):
    true = getattr(results_structural, name)

    for model in true['models']:
        kwargs = model.copy()
        kwargs.update(true['kwargs'])

        mod = structural.UnobservedComponents(dta['unemp'], **kwargs)
        mod.update(true['params'])
        res = mod.filter()

        assert_allclose(res.llf, true['llf'])

def test_irregular():
    run_ucm('irregular')

def test_fixed_intercept():
    warnings.simplefilter("always")
    with warnings.catch_warnings(True) as w:
        run_ucm('fixed_intercept')
        message = ("Specified model does not contain a stochastic element;"
                   " irregular component added.")
        assert(str(w[0].message) == message)

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
    with warnings.catch_warnings(True) as w:
        run_ucm('fixed_slope')
        message = ("Specified model does not contain a stochastic element;"
                   " irregular component added.")
        assert(str(w[0].message) == message)

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
