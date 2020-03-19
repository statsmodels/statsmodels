# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:45:51 2020

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from numpy.testing import assert_allclose

from statsmodels.stats.oneway import (confint_effectsize_oneway,
    confint_noncentrality, effectsize_oneway)


def test_oneway_effectsize():
    # examole 3 in Steiger 2004 Beyond the F-test, p. 169
    F = 5
    df1 = 3
    df2 = 76
    nobs = 80

    ci = confint_noncentrality(F, df1, df2, alpha=0.05, alternative="two-sided")

    ci_es = confint_effectsize_oneway(F, df1, df2, alpha=0.05)
    ci_steiger = ci_es.ci_f * np.sqrt(4 / 3)
    res_ci_steiger = [0.1764, 0.7367]
    res_ci_nc = np.asarray([1.8666, 32.563])

    assert_allclose(ci, res_ci_nc, atol=0.0001)
    assert_allclose(ci_es.ci_f_corrected, res_ci_steiger, atol=0.00006)
    assert_allclose(ci_steiger, res_ci_steiger, atol=0.00006)
    assert_allclose(ci_es.ci_f**2, res_ci_nc / nobs, atol=0.00006)
    assert_allclose(ci_es.ci_nc, res_ci_nc, atol=0.0001)


def test_effectsize_power():
    # example and results from PASS documentation
    n_groups = 3
    means = [527.86, 660.43, 649.14]
    vars_ = 107.4304**2
    nobs = 12
    es = effectsize_oneway(means, vars_, nobs, use_var="equal", ddof_between=0)

    alpha = 0.05
    power = 0.8
    nobs_t = nobs * n_groups
    kwds = {'effect_size': es, 'nobs': nobs_t, 'alpha': alpha, 'power': power,
            'k_groups': n_groups}

    from statsmodels.stats.power import FTestAnovaPower

    res_pow = 0.8251
    res_es = 0.559
    kwds_ = kwds.copy()
    del kwds_['power']
    p = FTestAnovaPower().power(**kwds_)
    assert_allclose(p, res_pow, atol=0.0001)
    assert_allclose(es, res_es, atol=0.0006)

    # example unequal sample sizes
    nobs = np.array([15, 9, 9])
    kwds['nobs'] = nobs
    es = effectsize_oneway(means, vars_, nobs, use_var="equal", ddof_between=0)
    kwds['effect_size'] = es
    p = FTestAnovaPower().power(**kwds_)

    res_pow = 0.8297
    res_es = 0.590
    assert_allclose(p, res_pow, atol=0.005)  # lower than print precision
    assert_allclose(es, res_es, atol=0.0006)
