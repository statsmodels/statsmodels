# -*- coding: utf-8 -*-
"""Tests for statistical power calculations

Note:
    test for ttest power are in test_weightstats.py
    tests for chisquare power are in test_gof.py

Created on Sat Mar 09 08:44:49 2013

Author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_almost_equal

import statsmodels.stats.power as smp


def test_normal_power_explicit():
    # a few initial test cases for NormalIndPower
    sigma = 1
    d = 0.3
    nobs = 80
    alpha = 0.05
    res1 = smp.normal_power(d, nobs/2., 0.05)
    res2 = smp.NormalIndPower().power(d, nobs, 0.05)
    res3 = smp.NormalIndPower().solve_power(effect_size=0.3, nobs1=80, alpha=0.05, beta=None)
    res_R = 0.475100870572638
    assert_almost_equal(res1, res_R, decimal=13)
    assert_almost_equal(res2, res_R, decimal=13)
    assert_almost_equal(res3, res_R, decimal=13)


    norm_pow = smp.normal_power(-0.01, nobs/2., 0.05)
    norm_pow_R = 0.05045832927039234
    #value from R: >pwr.2p.test(h=0.01,n=80,sig.level=0.05,alternative="two.sided")
    assert_almost_equal(norm_pow, norm_pow_R, decimal=13)

    norm_pow = smp.NormalIndPower().power(0.01, nobs, 0.05, alternative="1s")
    norm_pow_R = 0.056869534873146124
    #value from R: >pwr.2p.test(h=0.01,n=80,sig.level=0.05,alternative="greater")
    assert_almost_equal(norm_pow, norm_pow_R, decimal=13)

    # Note: negative effect size is same as switching one-sided alternative
    # TODO: should I switch to larger/smaller instead of "one-sided" options
    norm_pow = smp.NormalIndPower().power(-0.01, nobs, 0.05, alternative="1s")
    norm_pow_R = 0.0438089705093578
    #value from R: >pwr.2p.test(h=0.01,n=80,sig.level=0.05,alternative="less")
    assert_almost_equal(norm_pow, norm_pow_R, decimal=13)
