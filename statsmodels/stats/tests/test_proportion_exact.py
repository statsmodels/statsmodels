# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:36:51 2015

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from scipy import stats

from numpy.testing import assert_allclose

from statsmodels.stats._proportion_exact import ExactTwoProportion


def test_prop_exact1():
    # example 1 from Lin and Yang 2009


    n1, n2 = 15, 15
    yo1 = 7
    yo2 = 12

    # one sided
    # reference numbers for 1-sided alternative
    pval_lj_asymp = 0.02909
    pval_lj_exact = 0.03411
    pval_lj_bb = 0.03511

    pt = ExactTwoProportion(yo1, n1, yo2, n2, alternative='todo')
    pval_asymp = pt.pvalue_base
    assert_allclose(pval_asymp, pval_lj_asymp, rtol=0, atol=5e-5)

    # TODO use regression test if no reference value here
    pval_mle = pt.pvalue_exactdist_mle()
    #assert_allclose(pval_mle, pval_lj_mle, rtol=0, atol=5e-5)

    pres = pt.pvalue_exact_sup()
    pval_exact = pres[0]
    assert_allclose(pval_exact, pval_lj_exact, rtol=0, atol=5e-5)

    presbb = pt.pvalue_exact_sup(grid=('bb', 0.001, 1001))
    pval_bb = presbb[0]
    assert_allclose(pval_bb, pval_lj_bb, rtol=0, atol=5e-5)

    # reference numbers from bergers website for bb
    bb_confint = (0.3243, 0.8786)
    bb_pvalue = 0.0351
    p_argmax = 0.3333
    bb_statistic = - (-1.8943)  # Note: I have different sign
    # note currently we prepend p_cmle, low is in row 1 (2nd row)
    p_confint = (presbb[-1][1, 0], presbb[-1][-1, 0])
    assert_allclose(p_confint, bb_confint, rtol=0, atol=5e-4)
    assert_allclose(pt.statistic_base, bb_statistic, rtol=0, atol=5e-4)
    assert_allclose(pval_bb, bb_pvalue, rtol=0, atol=5e-4)
    #assert_allclose(presbb[1], p_argmax, rtol=0, atol=5e-4)
    # skip in this case, not unique
    #x: array(0.6646393071907146)  y: array(0.3333)


    # two sided compared with Berger website
    pt = ExactTwoProportion(yo1, n1, yo2, n2, alternative='2-sided')
    presbb = pt.pvalue_exact_sup(grid=('bb', 0.00001, 1001))
    pval_bb = presbb[0]

    # reference numbers from bergers website for bb
    bb_confint = (0.2389, 0.9264)
    bb_pvalue = 0.0682
    p_argmax = 0.3333
    bb_statistic = (-1.8943)**2

    # note currently we prepend p_cmle
    p_confint = (presbb[-1][1, 0], presbb[-1][-1, 0])
    assert_allclose(p_confint, bb_confint, rtol=0, atol=5e-4)
    assert_allclose(pt.statistic_base, bb_statistic, rtol=0, atol=5e-4)
    assert_allclose(pval_bb, bb_pvalue, rtol=0, atol=5e-4)
    #assert_allclose(presbb[1], p_argmax, rtol=0, atol=5e-4)
    # skip in this case, not unique ? or symmetric
    #x: array(0.6646393071907146)  y: array(0.3333)



def test_prop_exact2():
    # example 1 from Lin and Yang 2009
    n1, n2 = 69, 88
    yo1 = 67
    yo2 = 76

    # two sided
    # reference numbers for 2-sided alternative of Ling Yang
    pval_lj_asymp = 0.01912
    pval_lj_exact = 0.02099
    pval_lj_bb = 0.02151

    pt = ExactTwoProportion(yo1, n1, yo2, n2, alternative='2-sided')
    pval_asymp = pt.pvalue_base
    assert_allclose(pval_asymp, pval_lj_asymp, rtol=0, atol=5e-5)
    pval_mle = pt.pvalue_exactdist_mle()
    #assert_allclose(pval_mle, pval_lj_mle, rtol=0, atol=5e-5)

    pres = pt.pvalue_exact_sup()
    pval_exact = pres[0]
    assert_allclose(pval_exact, pval_lj_exact, rtol=0, atol=5e-5)

    presbb = pt.pvalue_exact_sup(grid=('bb', 0.001, 101))
    pval_bb = presbb[0]
    assert_allclose(pval_bb, pval_lj_bb, rtol=0, atol=5e-5)
