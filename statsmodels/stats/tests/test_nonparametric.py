"""

Created on Fri Jul 05 14:05:24 2013
Aug 15 2020: add brunnermunzel, rank_compare_2indep

Author: Josef Perktold
"""
from statsmodels.compat.python import lzip

from pathlib import Path

import numpy as np
from numpy.testing import (
    assert_,
    assert_allclose,
    assert_almost_equal,
    assert_approx_equal,
)
import pandas as pd
import pytest
from scipy import stats

from statsmodels.sandbox.stats.runs import (
    Runs,
    mcnemar as sbmcnemar,
    runstest_1samp,
    runstest_2samp,
)
from statsmodels.stats.contingency_tables import (
    SquareTable,
    cochrans_q,
    mcnemar,
)
from statsmodels.stats.nonparametric import (
    _compute_rank_placements,
    cohensd2problarger,
    prob_larger_continuous,
    rank_compare_2indep,
    rank_compare_2ordinal,
    samplesize_rank_compare_onetail,
)
from statsmodels.tools.testing import Holder


def _expand_table(table):
    '''expand a 2 by 2 contingency table to observations
    '''
    return np.repeat([[1, 1], [1, 0], [0, 1], [0, 0]], table.ravel(), axis=0)


def test_mcnemar_exact():
    f_obs1 = np.array([[101, 121], [59, 33]])
    f_obs2 = np.array([[101,  70], [59, 33]])
    f_obs3 = np.array([[101,  80], [59, 33]])
    f_obs4 = np.array([[101,  30], [60, 33]])
    f_obs5 = np.array([[101,  10], [30, 33]])
    f_obs6 = np.array([[101,  10], [10, 33]])

    #vassar college online computation
    res1 = 0.000004
    res2 = 0.378688
    res3 = 0.089452
    res4 = 0.00206
    res5 = 0.002221
    res6 = 1.
    stat = mcnemar(f_obs1, exact=True)
    assert_almost_equal([stat.statistic, stat.pvalue], [59, res1], decimal=6)
    stat = mcnemar(f_obs2, exact=True)
    assert_almost_equal([stat.statistic, stat.pvalue], [59, res2], decimal=6)
    stat = mcnemar(f_obs3, exact=True)
    assert_almost_equal([stat.statistic, stat.pvalue], [59, res3], decimal=6)
    stat = mcnemar(f_obs4, exact=True)
    assert_almost_equal([stat.statistic, stat.pvalue], [30, res4], decimal=6)
    stat = mcnemar(f_obs5, exact=True)
    assert_almost_equal([stat.statistic, stat.pvalue], [10, res5], decimal=6)
    stat = mcnemar(f_obs6, exact=True)
    assert_almost_equal([stat.statistic, stat.pvalue], [10, res6], decimal=6)


def test_mcnemar_chisquare():
    f_obs1 = np.array([[101, 121], [59, 33]])
    f_obs2 = np.array([[101,  70], [59, 33]])
    f_obs3 = np.array([[101,  80], [59, 33]])

    #> mcn = mcnemar.test(matrix(c(101, 121,  59,  33),nrow=2))
    res1 = [2.067222e01, 5.450095e-06]
    res2 = [0.7751938,    0.3786151]
    res3 = [2.87769784,   0.08981434]

    stat = mcnemar(f_obs1, exact=False)
    assert_allclose([stat.statistic, stat.pvalue], res1, rtol=1e-6)
    stat = mcnemar(f_obs2, exact=False)
    assert_allclose([stat.statistic, stat.pvalue], res2, rtol=1e-6)
    stat = mcnemar(f_obs3, exact=False)
    assert_allclose([stat.statistic, stat.pvalue], res3, rtol=1e-6)

    # test correction = False
    res1 = [2.135556e01, 3.815136e-06]
    res2 = [0.9379845,   0.3327967]
    res3 = [3.17266187,  0.07488031]

    res = mcnemar(f_obs1, exact=False, correction=False)
    assert_allclose([res.statistic, res.pvalue], res1, rtol=1e-6)
    res = mcnemar(f_obs2, exact=False, correction=False)
    assert_allclose([res.statistic, res.pvalue], res2, rtol=1e-6)
    res = mcnemar(f_obs3, exact=False, correction=False)
    assert_allclose([res.statistic, res.pvalue], res3, rtol=1e-6)


def test_mcnemar_vectorized(reset_randomstate):
    ttk = np.random.randint(5,15, size=(2,2,3))
    with pytest.warns(FutureWarning):
        res = sbmcnemar(ttk, exact=False)
    with pytest.warns(FutureWarning):
        res1 = lzip(*[sbmcnemar(ttk[:, :, i], exact=False) for i in range(3)])
    assert_allclose(res, res1, rtol=1e-13)

    with pytest.warns(FutureWarning):
        res = sbmcnemar(ttk, exact=False, correction=False)
    with pytest.warns(FutureWarning):
        res1 = lzip(*[sbmcnemar(ttk[:, :, i], exact=False, correction=False)
                      for i in range(3)])
    assert_allclose(res, res1, rtol=1e-13)

    with pytest.warns(FutureWarning):
        res = sbmcnemar(ttk, exact=True)
    with pytest.warns(FutureWarning):
        res1 = lzip(*[sbmcnemar(ttk[:, :, i], exact=True) for i in range(3)])
    assert_allclose(res, res1, rtol=1e-13)


def test_symmetry_bowker():
    table = np.array([0, 3, 4, 4, 2, 4, 1, 2, 4, 3, 5, 3, 0, 0, 2, 2, 3, 0, 0,
                      1, 5, 5, 5, 5, 5]).reshape(5, 5)

    res = SquareTable(table, shift_zeros=False).symmetry()
    mcnemar5_1 = dict(statistic=7.001587, pvalue=0.7252951, parameters=(10,),
                      distr='chi2')
    assert_allclose([res.statistic, res.pvalue],
                    [mcnemar5_1['statistic'], mcnemar5_1['pvalue']],
                    rtol=1e-7)

    res = SquareTable(1 + table, shift_zeros=False).symmetry()
    mcnemar5_1b = dict(statistic=5.355988, pvalue=0.8661652, parameters=(10,),
                       distr='chi2')
    assert_allclose([res.statistic, res.pvalue],
                    [mcnemar5_1b['statistic'], mcnemar5_1b['pvalue']],
                    rtol=1e-7)

    table = np.array([2, 2, 3, 6, 2, 3, 4, 3, 6, 6, 6, 7, 1, 9, 6, 7, 1, 1, 9,
                      8, 0, 1, 8, 9, 4]).reshape(5, 5)

    res = SquareTable(table, shift_zeros=False).symmetry()
    mcnemar5_2 = dict(statistic=18.76432, pvalue=0.04336035, parameters=(10,),
                      distr='chi2')
    assert_allclose([res.statistic, res.pvalue],
                    [mcnemar5_2['statistic'], mcnemar5_2['pvalue']],
                    rtol=1.5e-7)

    res = SquareTable(1 + table, shift_zeros=False).symmetry()
    mcnemar5_2b = dict(statistic=14.55256, pvalue=0.1492461, parameters=(10,),
                       distr='chi2')
    assert_allclose([res.statistic, res.pvalue],
                    [mcnemar5_2b['statistic'], mcnemar5_2b['pvalue']],
                    rtol=1e-7)


def test_cochransq():
    #example from dataplot docs, Conovover p. 253
    #http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/cochran.htm
    x = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [0, 1, 0],
                   [1, 1, 0],
                   [0, 0, 0],
                   [1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 0],
                   [0, 0, 1],
                   [0, 1, 0],
                   [1, 1, 1],
                   [1, 1, 1]])
    res_qstat = 2.8
    res_pvalue = 0.246597
    res = cochrans_q(x)
    assert_almost_equal([res.statistic, res.pvalue], [res_qstat, res_pvalue])

    #equivalence of mcnemar and cochranq for 2 samples
    a,b = x[:,:2].T
    res = cochrans_q(x[:, :2])
    with pytest.warns(FutureWarning):
        assert_almost_equal(sbmcnemar(a, b, exact=False, correction=False),
                            [res.statistic, res.pvalue])


def test_cochransq2():
    # from an example found on web, verifies 13.286
    data = np.array('''
        0 0 0 1
        0 0 0 1
        0 0 0 1
        1 1 1 1
        1 0 0 1
        0 1 0 1
        1 0 0 1
        0 0 0 1
        0 1 0 0
        0 0 0 0
        1 0 0 1
        0 0 1 1'''.split(), int).reshape(-1, 4)

    res = cochrans_q(data)
    assert_allclose([res.statistic, res.pvalue], [13.2857143, 0.00405776], rtol=1e-6)


def test_cochransq3():
    # another example compared to SAS
    # in frequency weight format
    cases = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 0, 1],
                      [1, 0, 1],
                      [0, 1, 0],
                      [1, 1, 0],
                      [0, 1, 1],
                      [1, 1, 1]])
    count = np.array([ 6,  2, 16,  4,  2,  6,  4,  6])
    data = np.repeat(cases, count, 0)

    res = cochrans_q(data)
    assert_allclose([res.statistic, res.pvalue], [8.4706, 0.0145], atol=5e-5)

def test_runstest(reset_randomstate):
    #comparison numbers from R, tseries, runs.test
    #currently only 2-sided used
    x = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1])

    z_twosided = 1.386750
    pvalue_twosided = 0.1655179

    #print Runs(x).runs_test(correction=False)
    assert_almost_equal(np.array(Runs(x).runs_test(correction=False)),
                        [z_twosided, pvalue_twosided], decimal=6)


    # compare with runstest_1samp which should have same indicator
    assert_almost_equal(runstest_1samp(x, correction=False),
                        [z_twosided, pvalue_twosided], decimal=6)

    x2 = x - 0.5 + np.random.uniform(-0.1, 0.1, size=len(x))
    assert_almost_equal(runstest_1samp(x2, cutoff=0, correction=False),
                        [z_twosided, pvalue_twosided], decimal=6)

    assert_almost_equal(runstest_1samp(x2, cutoff='mean', correction=False),
                        [z_twosided, pvalue_twosided], decimal=6)
    assert_almost_equal(runstest_1samp(x2, cutoff=x2.mean(), correction=False),
                        [z_twosided, pvalue_twosided], decimal=6)

    # check median
    assert_almost_equal(runstest_1samp(x2, cutoff='median', correction=False),
                        runstest_1samp(x2, cutoff=np.median(x2), correction=False),
                        decimal=6)


def test_runstest_2sample():
    # regression test, checked with MonteCarlo and looks reasonable

    x = [31.8, 32.8, 39.2, 36, 30, 34.5, 37.4]
    y = [35.5, 27.6, 21.3, 24.8, 36.7, 30]
    y[-1] += 1e-6  #avoid tie that creates warning

    res = runstest_2samp(x, y)
    res1 = (0.022428065200812752, 0.98210649318649212)
    assert_allclose(res, res1, rtol=1e-6)

    # check as stacked array
    res2 = runstest_2samp(x, y)
    assert_allclose(res2, res, rtol=1e-6)

    xy = np.concatenate((x, y))
    res_1s = runstest_1samp(xy)
    assert_allclose(res_1s, res1, rtol=1e-6)
    # check cutoff
    res2_1s = runstest_1samp(xy, xy.mean())
    assert_allclose(res2_1s, res_1s, rtol=1e-6)


def test_brunnermunzel_one_sided():
    # copied from scipy with adjustment
    x = [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1]
    y = [3, 3, 4, 3, 1, 2, 3, 1, 1, 5, 4]
    significant = 13

    # revere direction to match our definition
    x, y = y, x

    # Results are compared with R's lawstat package.
    u1, p1 = rank_compare_2indep(x, y
                                 ).test_prob_superior(alternative='smaller')
    u2, p2 = rank_compare_2indep(y, x
                                 ).test_prob_superior(alternative='larger')
    u3, p3 = rank_compare_2indep(x, y
                                 ).test_prob_superior(alternative='larger')
    u4, p4 = rank_compare_2indep(y, x
                                 ).test_prob_superior(alternative='smaller')

    assert_approx_equal(p1, p2, significant=significant)
    assert_approx_equal(p3, p4, significant=significant)
    assert_(p1 != p3)
    assert_approx_equal(u1, 3.1374674823029505,
                        significant=significant)
    assert_approx_equal(u2, -3.1374674823029505,
                        significant=significant)
    assert_approx_equal(u3, 3.1374674823029505,
                        significant=significant)
    assert_approx_equal(u4, -3.1374674823029505,
                        significant=significant)

    # Note: scipy and lawstat tail is reversed compared to test statistic
    assert_approx_equal(p3, 0.0028931043330757342,
                        significant=significant)
    assert_approx_equal(p1, 0.99710689566692423,
                        significant=significant)


def test_brunnermunzel_two_sided():
    # copied from scipy with adjustment
    x = [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1]
    y = [3, 3, 4, 3, 1, 2, 3, 1, 1, 5, 4]
    significant = 13

    # revere direction to match our definition
    x, y = y, x

    # Results are compared with R's lawstat package.
    res1 = rank_compare_2indep(x, y)
    u1, p1 = res1
    t1 = res1.test_prob_superior(alternative='two-sided')
    res2 = rank_compare_2indep(y, x)
    u2, p2 = res2
    t2 = res2.test_prob_superior(alternative='two-sided')

    assert_approx_equal(p1, p2, significant=significant)
    assert_approx_equal(u1, 3.1374674823029505,
                        significant=significant)
    assert_approx_equal(u2, -3.1374674823029505,
                        significant=significant)
    assert_approx_equal(p2, 0.0057862086661515377,
                        significant=significant)

    assert_allclose(t1[0], u1, rtol=1e-13)
    assert_allclose(t2[0], u2, rtol=1e-13)
    assert_allclose(t1[1], p1, rtol=1e-13)
    assert_allclose(t2[1], p2, rtol=1e-13)


def test_rank_compare_2indep1():
    # Example from Munzel and Hauschke 2003
    # data is given by counts, expand to observations
    levels = [-2, -1, 0, 1, 2]
    new = [24, 37, 21, 19, 6]
    active = [11, 51, 22, 21, 7]
    x1 = np.repeat(levels, new)
    x2 = np.repeat(levels, active)

    # using lawstat
    # > brunner.munzel.test(xn, xa) #brunnermunzel.test(x, y)
    res2_t = Holder(statistic=1.1757561456582,
                    df=204.2984239868,
                    pvalue=0.2410606649547,
                    ci=[0.4700629827705593, 0.6183882855872511],
                    prob=0.5442256341789052)

    res = rank_compare_2indep(x1, x2, use_t=False)
    assert_allclose(res.statistic, -res2_t.statistic, rtol=1e-13)
    assert_allclose(res.prob1, 1 - res2_t.prob, rtol=1e-13)
    assert_allclose(res.prob2, res2_t.prob, rtol=1e-13)
    tt = res.test_prob_superior()
    # TODO: return HolderTuple
    # assert_allclose(tt.statistic, res2_t.statistic)
    # TODO: check sign/direction in lawstat
    assert_allclose(tt[0], -res2_t.statistic, rtol=1e-13)

    ci = res.conf_int(alpha=0.05)
    # we compare normal confint with t confint, lower rtol
    assert_allclose(ci, 1 - np.array(res2_t.ci)[::-1], rtol=0.005)
    # test consistency of test and confint
    res_lb = res.test_prob_superior(value=ci[0])
    assert_allclose(res_lb[1], 0.05, rtol=1e-13)
    res_ub = res.test_prob_superior(value=ci[1])
    assert_allclose(res_ub[1], 0.05, rtol=1e-13)

    # test consistency of tost and confint
    # lower margin is binding, alternative larger
    res_tost = res.tost_prob_superior(ci[0], ci[1] * 1.05)
    assert_allclose(res_tost.results_larger.pvalue, 0.025, rtol=1e-13)
    assert_allclose(res_tost.pvalue, 0.025, rtol=1e-13)

    # upper margin is binding, alternative smaller
    res_tost = res.tost_prob_superior(ci[0] * 0.85, ci[1])
    assert_allclose(res_tost.results_smaller.pvalue, 0.025, rtol=1e-13)
    assert_allclose(res_tost.pvalue, 0.025, rtol=1e-13)

    # use t-distribution
    # our ranking is defined as reversed from lawstat, and BM article
    # revere direction to match our definition
    x1, x2 = x2, x1
    res = rank_compare_2indep(x1, x2, use_t=True)
    assert_allclose(res.statistic, res2_t.statistic, rtol=1e-13)
    tt = res.test_prob_superior()
    # TODO: return HolderTuple
    # assert_allclose(tt.statistic, res2_t.statistic)
    # TODO: check sign/direction in lawstat, reversed from ours
    assert_allclose(tt[0], res2_t.statistic, rtol=1e-13)
    assert_allclose(tt[1], res2_t.pvalue, rtol=1e-13)
    assert_allclose(res.pvalue, res2_t.pvalue, rtol=1e-13)
    assert_allclose(res.df, res2_t.df, rtol=1e-13)

    ci = res.conf_int(alpha=0.05)
    assert_allclose(ci, res2_t.ci, rtol=1e-11)
    # test consistency of test and confint
    res_lb = res.test_prob_superior(value=ci[0])
    assert_allclose(res_lb[1], 0.05, rtol=1e-11)
    res_ub = res.test_prob_superior(value=ci[1])
    assert_allclose(res_ub[1], 0.05, rtol=1e-11)

    # test consistency of tost and confint
    # lower margin is binding, alternative larger
    res_tost = res.tost_prob_superior(ci[0], ci[1] * 1.05)
    assert_allclose(res_tost.results_larger.pvalue, 0.025, rtol=1e-10)
    assert_allclose(res_tost.pvalue, 0.025, rtol=1e-10)

    # upper margin is binding, alternative smaller
    res_tost = res.tost_prob_superior(ci[0] * 0.85, ci[1])
    assert_allclose(res_tost.results_smaller.pvalue, 0.025, rtol=1e-10)
    assert_allclose(res_tost.pvalue, 0.025, rtol=1e-10)

    # extras
    # cohen's d
    esd = res.effectsize_normal()
    p = prob_larger_continuous(stats.norm(loc=esd), stats.norm)
    # round trip
    assert_allclose(p, res.prob1, rtol=1e-13)

    # round trip with cohen's d
    pc = cohensd2problarger(esd)
    assert_allclose(pc, res.prob1, rtol=1e-13)

    ci_tr = res.confint_lintransf(1, -1)
    assert_allclose(ci_tr, 1 - np.array(res2_t.ci)[::-1], rtol=0.005)


def test_rank_compare_ord():
    # compare ordinal count version with full version
    # Example from Munzel and Hauschke 2003
    # data is given by counts, expand to observations
    levels = [-2, -1, 0, 1, 2]
    new = [24, 37, 21, 19, 6]
    active = [11, 51, 22, 21, 7]
    x1 = np.repeat(levels, new)
    x2 = np.repeat(levels, active)

    for use_t in [False, True]:
        res2 = rank_compare_2indep(x1, x2, use_t=use_t)
        res1 = rank_compare_2ordinal(new, active, use_t=use_t)
        assert_allclose(res2.prob1, res1.prob1, rtol=1e-13)
        assert_allclose(res2.var_prob, res1.var_prob, rtol=1e-13)
        s1 = str(res1.summary())
        s2 = str(res2.summary())
        assert s1 == s2


def test_rank_compare_vectorized():
    np.random.seed(987126)
    x1 = np.random.randint(0, 20, (50, 3))
    x2 = np.random.randint(5, 25, (50, 3))
    res = rank_compare_2indep(x1, x2)
    tst = res.test_prob_superior(0.5)
    tost = res.tost_prob_superior(0.4, 0.6)

    # smoke test for summary
    res.summary()

    for i in range(3):
        res_i = rank_compare_2indep(x1[:, i], x2[:, i])
        assert_allclose(res.statistic[i], res_i.statistic, rtol=1e-14)
        assert_allclose(res.pvalue[i], res_i.pvalue, rtol=1e-14)
        assert_allclose(res.prob1[i], res_i.prob1, rtol=1e-14)

        tst_i = res_i.test_prob_superior(0.5)
        assert_allclose(tst.statistic[i], tst_i.statistic, rtol=1e-14)
        assert_allclose(tst.pvalue[i], tst_i.pvalue, rtol=1e-14)

        tost_i = res_i.tost_prob_superior(0.4, 0.6)
        assert_allclose(tost.statistic[i], tost_i.statistic, rtol=1e-14)
        assert_allclose(tost.pvalue[i], tost_i.pvalue, rtol=1e-14)


cases_continuous = [
    (
        # X1 and X2 continuous with no ties
        np.array([1.1, 2.2, 3.3, 4.4, 5.5]),
        np.array([6.6, 7.7, 8.8, 9.9, 10.1]),
        # Expected ranks and placements
        Holder(
            n_1=5,
            n_2=5,
            overall_ranks_pooled=np.arange(1, 11),
            overall_ranks_1=np.arange(1, 6),
            overall_ranks_2=np.arange(6, 11),
            within_group_ranks_1=np.arange(1, 6),
            within_group_ranks_2=np.arange(1, 6),
            placements_1=np.repeat(0, 5),
            placements_2=np.repeat(5, 5),
        ),
    )
]
cases_ordinal = [
    (
        # X1 and X2 with ties within a group
        np.array([1, 1, 2, 2, 3]),
        np.array([4, 5, 6, 7, 8]),
        # Expected ranks and placements
        Holder(
            n_1=5,
            n_2=5,
            # First two ties are (1+2)/2=1.5, next two are (3+4)/2=3.5
            overall_ranks_pooled=np.array(
                [1.5, 1.5, 3.5, 3.5, 5, 6, 7, 8, 9, 10]
            ),
            overall_ranks_1=np.array([1.5, 1.5, 3.5, 3.5, 5]),
            overall_ranks_2=np.arange(6, 11),
            within_group_ranks_1=np.array([1.5, 1.5, 3.5, 3.5, 5]),
            within_group_ranks_2=np.arange(1, 6),
            placements_1=np.repeat(0, 5),
            placements_2=np.repeat(5, 5),
        ),
    ),
    (
        # X1 and X2 with ties within and across groups
        np.array([1, 2, 2, 4, 5]),
        np.array([4, 5, 6, 7, 8]),
        # Expected ranks and placements
        Holder(
            n_1=5,
            n_2=5,
            # Ties at (2+3)/2=2.5, (4+5)/2=4.5, (6+7)/2=6.5
            # So 2 -> 2.5, 4 -> 4.5, 5 -> 6.5
            overall_ranks_pooled=np.array(
                # Sorted: 1, (2, 2), (4, 4), (5, 5),  6,  7,  8
                # From:  x1, x1, x2, x1, x2, x1, x2, x2, x2, x2
                [1, 2.5, 2.5, 4.5, 6.5,
                 4.5, 6.5, 8, 9, 10]
            ),
            overall_ranks_1=np.array([1, 2.5, 2.5, 4.5, 6.5]),
            overall_ranks_2=np.array([4.5, 6.5, 8, 9, 10]),
            # Ties in X1 between positions 2 and 3
            within_group_ranks_1=np.array([1, 2.5, 2.5, 4, 5]),
            # No within-group ties in X2
            within_group_ranks_2=np.arange(1, 6),
            placements_1=np.array([0, 0, 0, 4.5 - 4, 6.5 - 5]),
            placements_2=np.array([4.5 - 1, 6.5 - 2, 5, 5, 5]),
        )
    )
]


@pytest.mark.parametrize(
        "test_cases",
        cases_continuous + cases_ordinal
)
def test_compute_rank_placements(test_cases):
    """
    Test the `_compute_rank_placements` helper for
    computing ranks and placements based on two
    input samples. Data validation logic is assumed
    to be handled by the caller and is not necessary
    to test here.
    """
    x1, x2, expected_holder = test_cases
    res = _compute_rank_placements(x1, x2)
    assert_allclose(res.n_1, expected_holder.n_1)
    assert_allclose(res.n_2, expected_holder.n_2)
    assert_allclose(
        res.overall_ranks_pooled, expected_holder.overall_ranks_pooled
    )
    assert_allclose(
        res.overall_ranks_1, expected_holder.overall_ranks_1
    )
    assert_allclose(
        res.overall_ranks_2, expected_holder.overall_ranks_2
    )
    assert_allclose(
        res.within_group_ranks_1, expected_holder.within_group_ranks_1
    )
    assert_allclose(
        res.within_group_ranks_2, expected_holder.within_group_ranks_2
    )
    assert_allclose(res.placements_1, expected_holder.placements_1)
    assert_allclose(res.placements_2, expected_holder.placements_2)


@pytest.fixture(scope="function")
def reference_implementation_results():
    """
    Results from R's rankFD::WMWSSP function.
    """
    parent_dir = Path(__file__).resolve().parent
    results = pd.read_csv(
        parent_dir / "results/results_samplesize_rank_compare_onetail.csv"
    )
    return results

def test_samplesize_rank_compare_onetail(reference_implementation_results):
    """
    Test the `samplesize_rank_compare_onetail` function against the reference
    implementation from R's rankFD package. Examples are taken from the
    reference paper directly. The reference implementation results are
    generated using the `generate_results_samplesize_rank_compare_onetail.R`
    script.
    """
    for _, r_result in reference_implementation_results.iterrows():
        synthetic_sample = np.array(
            r_result["synthetic_sample"].split(","), dtype=np.float64
        )
        reference_sample = np.array(
            r_result["reference_sample"].split(","), dtype=np.float64
        )
        # Convert `prop_reference` to `nobs_ratio`
        nobs_ratio = r_result["prop_reference"] / (1 - r_result["prop_reference"])
        py_result = samplesize_rank_compare_onetail(
            synthetic_sample=synthetic_sample,
            reference_sample=reference_sample,
            alpha=r_result["alpha"],
            power=r_result["power"],
            nobs_ratio=nobs_ratio,
            alternative=r_result["alternative"],
        )
        # Integers can be compared directly
        assert_allclose(
            py_result.nobs_total,
            r_result["nobs_total"],
            rtol=1e-9,
            atol=0,
        )
        assert_allclose(
            py_result.nobs_ref,
            r_result["nobs_ref"],
            rtol=1e-9,
        )
        assert_allclose(
            py_result.nobs_treat,
            r_result["nobs_treat"],
            rtol=1e-9,
        )
        assert_allclose(
            py_result.relative_effect,
            r_result["relative_effect"],
            rtol=1e-9,
            atol=0,
        )


@pytest.mark.parametrize(
    (
        "synthetic_sample, reference_sample,"
        "alpha, power, "
        "nobs_ratio, alternative, "
        "expected_exception, exception_msg"
    ),
    [
        # Invalid alpha (0 or 1 is not allowed)
        (
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            0.0,
            0.8,
            1.0,
            "two-sided",
            ValueError,
            "Alpha must be between 0 and 1",
        ),
        (
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            1.0,
            0.8,
            1.0,
            "two-sided",
            ValueError,
            "Alpha must be between 0 and 1",
        ),
        # Invalid power (0 or 1 is not allowed)
        (
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            0.05,
            0.0,
            1.0,
            "one-sided",
            ValueError,
            "Power must be between 0 and 1",
        ),
        (
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            0.05,
            1.0,
            1.0,
            "one-sided",
            ValueError,
            "Power must be between 0 and 1",
        ),
        # Invalid nobs_ratio (must be strictly positive)
        (
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            0.05,
            0.8,
            0.0,
            "one-sided",
            ValueError,
            "Ratio of reference group to treatment group must be"
            " strictly positive."
        ),
        (
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            0.05,
            0.8,
            -1.0,
            "one-sided",
            ValueError,
            "Ratio of reference group to treatment group must be"
            " strictly positive."
        ),
        # Invalid synthetic sample with missing values (NaN)
        (
            np.array([1, 2, np.nan]),
            np.array([1, 2, 3]),
            0.05,
            0.8,
            0.5,
            "one-sided",
            ValueError,
            "All elements of `synthetic_sample` and `reference_sample`"
            " must be finite; check for missing values."
        ),
        # Empty reference sample
        (
            np.array([1, 2, 3]),
            np.array([]),
            0.05,
            0.8,
            0.5,
            "one-sided",
            ValueError,
            "Both `synthetic_sample` and `reference_sample` must have"
            " at least one element",
        ),
        # Invalid alternative type
        (
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            0.05,
            0.8,
            0.5,
            False,
            ValueError,
            "Alternative must be one of `two-sided`, `larger`,"
            " or `smaller`.",
        ),
        # Invalid alternative value
        (
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            0.05,
            0.8,
            0.5,
            "invalid-alternative",
            ValueError,
            "Alternative must be one of `two-sided`, `larger`,"
            " or `smaller`.",
        ),
        # Relative effect > 0.5 but alternative is smaller
        (
            np.array([4, 5, 6]), # Synthetic sample
            np.array([1, 2, 3]), # Reference sample
            0.05,
            0.8,
            1.0,
            "smaller",
            ValueError,
            "Estimated relative effect is larger than 0.5"
        ),
        # Relative effect < 0.5 but alternative is larger
        (
            np.array([1, 2, 3]), # Synthetic sample
            np.array([4, 6, 8]), # Reference sample
            0.05,
            0.8,
            1.0,
            "larger",
            ValueError,
            "Estimated relative effect is smaller than 0.5"
        ),
        # Relative effect = 0.5
        (
            np.array([1, 2, 3 - 1e-16]),
            np.array([1, 2 + 1e-16, 3]),
            0.05,
            0.80,
            1.0,
            "two-sided",
            ValueError,
            "Estimated relative effect is effectively 0.5"
        ),
    ],
    scope="function",
)
def test_samplesize_rank_compare_onetail_invalid(
    synthetic_sample,
    reference_sample,
    alpha,
    power,
    nobs_ratio,
    alternative,
    expected_exception,
    exception_msg,
):
    """
    Test the samplesize_rank_compare_onetail function with various invalid inputs.
    """
    with pytest.raises(expected_exception, match=exception_msg):
        samplesize_rank_compare_onetail(
            synthetic_sample=synthetic_sample,
            reference_sample=reference_sample,
            alpha=alpha,
            power=power,
            nobs_ratio=nobs_ratio,
            alternative=alternative,
        )
