# -*- coding: utf-8 -*-
"""

Created on Fri Aug 16 13:41:12 2013

Author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_equal, assert_raises

from scipy import stats
from statsmodels.stats.robust_compare import (
        TrimmedMean, anova_bfm, anova_oneway, anova_scale, anova_welch,
        scale_transform, trim_mean, trimboth)

# taken from scipy and adjusted
class Test_Trim(object):
    # test trim functions
    def t_est_trim1(self):
        a = np.arange(11)
        assert_equal(trim1(a, 0.1), np.arange(10))
        assert_equal(trim1(a, 0.2), np.arange(9))
        assert_equal(trim1(a, 0.2, tail='left'), np.arange(2,11))
        assert_equal(trim1(a, 3/11., tail='left'), np.arange(3,11))

    def test_trimboth(self):
        a = np.arange(11)
        a2 = np.arange(24).reshape(6, 4)
        a3 = np.arange(24).reshape(6, 4, order='F')
        assert_equal(trimboth(a, 3/11.), np.arange(3,8))
        assert_equal(trimboth(a, 0.2), np.array([2, 3, 4, 5, 6, 7, 8]))

        assert_equal(trimboth(a2, 0.2),
                     np.arange(4,20).reshape(4,4))
        assert_equal(trimboth(a3, 2/6.),
               np.array([[2, 8, 14, 20],[3, 9, 15, 21]]))
        assert_raises(ValueError, trimboth,
               np.arange(24).reshape(4,6).T, 4/6.)

    def test_trim_mean(self):
        a = np.array([ 4,  8,  2,  0,  9,  5, 10,  1,  7,  3,  6])
        idx = np.array([3, 5, 0, 1, 2, 4])
        a2 = np.arange(24).reshape(6, 4)[idx, :]
        a3 = np.arange(24).reshape(6, 4, order='F')[idx, :]
        assert_equal(trim_mean(a3, 2/6.),
                        np.array([2.5, 8.5, 14.5, 20.5]))
        assert_equal(trim_mean(a2, 2/6.),
                        np.array([10., 11., 12., 13.]))
        idx4 = np.array([1, 0, 3, 2])
        a4 = np.arange(24).reshape(4, 6)[idx4, :]
        assert_equal(trim_mean(a4, 2/6.),
                        np.array([9., 10., 11., 12., 13., 14.]))
        # shuffled arange(24)
        a = np.array([ 7, 11, 12, 21, 16,  6, 22,  1,  5,  0, 18, 10, 17,  9,
                      19, 15, 23, 20,  2, 14,  4, 13,  8,  3])
        assert_equal(trim_mean(a, 2/6.), 11.5)
        assert_equal(trim_mean([5,4,3,1,2,0], 2/6.), 2.5)

        # check axis argument
        np.random.seed(1234)
        a = np.random.randint(20, size=(5, 6, 4, 7))
        for axis in [0, 1, 2, 3, -1]:
            res1 = trim_mean(a, 2/6., axis=axis)
            res2 = trim_mean(np.rollaxis(a, axis), 2/6.)
            assert_equal(res1, res2)

        res1 = trim_mean(a, 2/6., axis=None)
        res2 = trim_mean(a.ravel(), 2/6.)
        assert_equal(res1, res2)

def test_example_smoke():
    # cut and paste from `robust_compare.__main__`` without printing

    examples = ['mc', 'anova', 'trimmed', 'none'] #[-1]
    if 'mc' in examples:
        np.random.seed(19864256)
        nrep = 100
        nobs = np.array([5,10,5,5]) * 3
        mm = (1, 1, 1, 1)
        ss = (0.8, 1, 1, 2)
        #ss = (1, 1, 1, 1)

        # run a Monte Carlo simulation to check size and power of tests
        res_v = np.zeros((nrep, 3))  # without levene
        res_v = np.zeros((nrep, 5))  # with levene
        res = np.zeros((nrep, 6))
        res_w = np.zeros((nrep, 4))
        for ii in range(nrep):
            #xx = [m + s * np.random.randn(n) for n, m, s in zip(nobs, mm, ss)]
            #xx = [m + s * stats.t.rvs(3, size=n) for n, m, s in zip(nobs, mm, ss)]
            xx = [m + s * (stats.lognorm.rvs(1.5, size=n) - stats.lognorm.mean(1.5)) for n, m, s in zip(nobs, mm, ss)]
            #xx = [m + s * (stats.chi2.rvs(3, size=n) - stats.chi2.mean(3)) for n, m, s in zip(nobs, mm, ss)]
            #xxd = [np.abs(x - np.median(x)) for x in xx]
            xxd = [scale_transform(x, center='trimmed', transform='abs',
                                   trim_frac=0.1) for x in xx]
            #bf_anova(*xx)[:2], bf_anova(*xxd)[:2]
            # levene raises exception with unbalanced
            res_v[ii] = np.concatenate((stats.levene(*xx), anova_bfm(xxd)[:3]))
            #res_v[ii] = anova_bfm(xxd)[:3]
            res[ii] = np.concatenate((anova_bfm(xx)[:3],
                                      anova_bfm(xx, trim_frac=0.2)[:3]))
            res_w[ii] = np.concatenate((anova_welch(xx)[:2],
                                        anova_welch(xx, trim_frac=0.2)[:2]))

        #res[:5]
        nobs
        mm
        ss
        '\nlevene BF scale'
        #(res_v[:, [1, 2]] < 0.05).mean(0) # without levene
        (res_v[:, [1, 3, 4]] < 0.05).mean(0) # with levene
        '\nBF'
        (res[:, [1, 2, 4, 5]] < 0.05).mean(0)
        '\nWelch'
        (res_w[:, [1, 3]] < 0.05).mean(0)
        print

    if 'anova' in examples:
        np.random.seed(19864256)
        nobs = np.array([5,10,5,5]) * 3
        mm = (1, 1, 1, 2)
        ss = (0.8, 1, 1, 2)

        xx = [m + s * np.random.randn(n) for n, m, s in zip(nobs, mm, ss)]
        anova_bfm(xx)
        anova_welch(xx)

        npk_yield = np.array([
         49.5, 62.8, 46.8, 57, 59.8, 58.5, 55.5, 56, 62.8, 55.8, 69.5, 55, 62,
         48.8, 45.5, 44.2, 52, 51.5, 49.8, 48.8, 57.2, 59, 53.2, 56
        ])
        npk_block = np.array([
         1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6,
         6 ])
        xyield = [npk_yield[npk_block == idx] for idx in range(1,7)]
        anova_bfm(xyield)
        anova_welch(xyield)
        idx_include = list(range(24))
        del idx_include[15]
        del idx_include[2]
        # unbalanced sample sizes
        npk_block_ub = npk_block[idx_include]
        npk_yield_ub = npk_yield[idx_include]
        xyield_ub = [npk_yield_ub[npk_block_ub == idx] for idx in range(1,7)]
        anova_bfm(xyield_ub)
        anova_welch(xyield_ub)
        anova_welch(xyield_ub, trim_frac=0.01)
        anova_welch(xyield_ub, trim_frac=0.25)


    if 'trimmed' in examples:
        #x = np.random.permutation(np.arange(10))
        x = np.array([4, 9, 3, 1, 6, 5, 7, 10, 2, 8, 50])
        tm = TrimmedMean(x, 0.2)
        vars(tm)
        tm.data_winsorized
        tm.data_trimmed
        tm.mean_trimmed
        tm.mean_winsorized
        tm.var_winsorized
        tm2 = tm.reset_fraction(0.1)
        tm2.data_winsorized
        tm2.data_trimmed

        tm = tm.reset_fraction(0)
        import statsmodels.stats.weightstats as smws
        smws._tstat_generic(tm.mean_trimmed, 0, tm.std_mean_trimmed,
                                  tm.nobs_reduced - 1,
                                  alternative='two-sided', diff=3)
        smws.DescrStatsW(x).ttest_mean(3)
        tm.ttest_mean(3, transform='winsorized')

    x = np.asarray("7.79 9.16 7.64 10.28 9.12 9.24 8.40 8.60 8.04 8.45 9.51 8.15 7.69 8.84 9.92 7.20 9.25 9.45 9.14 9.99 9.21 9.06 8.65 10.70 10.24 8.62 9.94 10.55 10.13 9.78 9.01".split(), float)

