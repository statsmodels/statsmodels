# -*- coding: utf-8 -*-
"""

Created on Fri Jul 05 14:05:24 2013

Author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from statsmodels.sandbox.stats.runs import mcnemar, cochran_q, Runs

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

    assert_almost_equal(mcnemar(f_obs1, exact=True), [59, res1], decimal=6)
    assert_almost_equal(mcnemar(f_obs2, exact=True), [59, res2], decimal=6)
    assert_almost_equal(mcnemar(f_obs3, exact=True), [59, res3], decimal=6)
    assert_almost_equal(mcnemar(f_obs4, exact=True), [30, res4], decimal=6)
    assert_almost_equal(mcnemar(f_obs5, exact=True), [10, res5], decimal=6)
    assert_almost_equal(mcnemar(f_obs6, exact=True), [10, res6], decimal=6)

    x, y = _expand_table(f_obs2).T  # tuple unpack
    assert_allclose(mcnemar(f_obs2, exact=True),
                    mcnemar(x, y, exact=True), rtol=1e-13)

def test_mcnemar_chisquare():
    f_obs1 = np.array([[101, 121], [59, 33]])
    f_obs2 = np.array([[101,  70], [59, 33]])
    f_obs3 = np.array([[101,  80], [59, 33]])

    #> mcn = mcnemar.test(matrix(c(101, 121,  59,  33),nrow=2))
    res1 = [2.067222e+01, 5.450095e-06]
    res2 = [0.7751938,    0.3786151]
    res3 = [2.87769784,   0.08981434]


    assert_allclose(mcnemar(f_obs1, exact=False), res1, rtol=1e-6)
    assert_allclose(mcnemar(f_obs2, exact=False), res2, rtol=1e-6)
    assert_allclose(mcnemar(f_obs3, exact=False), res3, rtol=1e-6)

    x, y = _expand_table(f_obs2).T  # tuple unpack
    assert_allclose(mcnemar(f_obs2, exact=False),
                    mcnemar(x, y, exact=False), rtol=1e-13)


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
    assert_almost_equal(cochran_q(x), [res_qstat, res_pvalue])

    #equivalence of mcnemar and cochranq for 2 samples
    a,b = x[:,:2].T
    assert_almost_equal(mcnemar(a,b, exact=False, correction=False),
                        cochran_q(x[:,:2]))


def test_cochranq2():
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

    res = cochran_q(data)
    assert_allclose(res, [13.2857143, 0.00405776], rtol=1e-6)


def test_runstest():
    #comparison numbers from R, tseries, runs.test
    #currently only 2-sided used
    x = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1])

    z_twosided = 1.386750
    pvalue_twosided = 0.1655179

    z_greater = 1.386750
    pvalue_greater = 0.08275893

    z_less = 1.386750
    pvalue_less = 0.917241

    #print Runs(x).runs_test(correction=False)
    assert_almost_equal(np.array(Runs(x).runs_test(correction=False)),
                        [z_twosided, pvalue_twosided], decimal=6)
