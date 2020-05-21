# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:45:51 2020

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from statsmodels.stats.oneway import (
    confint_effectsize_oneway, confint_noncentrality, effectsize_oneway,
    anova_generic, equivalence_oneway, equivalence_oneway_generic,
    power_equivalence_oneway, power_equivalence_oneway0,
    f2_to_wellek, fstat_to_wellek, wellek_to_f2)


def test_oneway_effectsize():
    # examole 3 in Steiger 2004 Beyond the F-test, p. 169
    F = 5
    df1 = 3
    df2 = 76
    nobs = 80

    ci = confint_noncentrality(F, df1, df2, alpha=0.05,
                               alternative="two-sided")

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
    es = np.sqrt(es)

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
    es = np.sqrt(es)
    kwds['effect_size'] = es
    p = FTestAnovaPower().power(**kwds_)

    res_pow = 0.8297
    res_es = 0.590
    assert_allclose(p, res_pow, atol=0.005)  # lower than print precision
    assert_allclose(es, res_es, atol=0.0006)


class TestOnewayEquivalenc(object):

    @classmethod
    def setup_class(cls):
        y0 = [112.488, 103.738, 86.344, 101.708, 95.108, 105.931,
              95.815, 91.864, 102.479, 102.644]
        y1 = [100.421, 101.966, 99.636, 105.983, 88.377, 102.618,
              105.486, 98.662, 94.137, 98.626, 89.367, 106.204]
        y2 = [84.846, 100.488, 119.763, 103.736, 93.141, 108.254,
              99.510, 89.005, 108.200, 82.209, 100.104, 103.706,
              107.067]
        y3 = [100.825, 100.255, 103.363, 93.230, 95.325, 100.288,
              94.750, 107.129, 98.246, 96.365, 99.740, 106.049,
              92.691, 93.111, 98.243]

        n_groups = 4
        arrs_w = [np.asarray(yi) for yi in [y0, y1, y2, y3]]
        nobs = np.asarray([len(yi) for yi in arrs_w])
        nobs_mean = np.mean(nobs)
        means = np.asarray([yi.mean() for yi in arrs_w])
        stds = np.asarray([yi.std(ddof=1) for yi in arrs_w])
        cls.data = arrs_w  # TODO use `data`
        cls.means = means
        cls.nobs = nobs
        cls.stds = stds
        cls.n_groups = n_groups
        cls.nobs_mean = nobs_mean

    def test_equivalence_equal(self):
        # reference numbers from Jan and Shieh 2019, p. 5
        means = self.means
        nobs = self.nobs
        stds = self.stds
        n_groups = self.n_groups

        eps = 0.5
        res0 = anova_generic(means, stds**2, nobs, use_var="equal")
        f = res0.statistic
        res = equivalence_oneway_generic(f, n_groups, nobs.sum(), eps,
                                         res0.df, alpha=0.05,
                                         margin_type="wellek")
        assert_allclose(res.pvalue, 0.0083, atol=0.001)
        assert_equal(res.df, [3, 46])

        # the agreement for f-stat looks too low
        assert_allclose(f, 0.0926, atol=0.0006)

        res = equivalence_oneway(self.data, eps, use_var="equal",
                                 margin_type="wellek")
        assert_allclose(res.pvalue, 0.0083, atol=0.001)
        assert_equal(res.df, [3, 46])

    def test_equivalence_welch(self):
        # reference numbers from Jan and Shieh 2019, p. 6
        means = self.means
        nobs = self.nobs
        stds = self.stds
        n_groups = self.n_groups
        vars_ = stds**2

        eps = 0.5
        res0 = anova_generic(means, vars_, nobs, use_var="unequal",
                             welch_correction=False)
        f_stat = res0.statistic
        res = equivalence_oneway_generic(f_stat, n_groups, nobs.sum(), eps,
                                         res0.df, alpha=0.05,
                                         margin_type="wellek")
        assert_allclose(res.pvalue, 0.0110, atol=0.001)
        assert_allclose(res.df, [3.0, 22.6536], atol=0.0006)

        # agreement for Welch f-stat looks too low b/c welch_correction=False
        assert_allclose(f_stat, 0.1102, atol=0.007)

        res = equivalence_oneway(self.data, eps, use_var="unequal",
                                 margin_type="wellek")
        assert_allclose(res.pvalue, 0.0110, atol=1e-4)
        assert_allclose(res.df, [3.0, 22.6536], atol=0.0006)
        assert_allclose(res.f_stat, 0.1102, atol=1e-4)  # 0.007)

        # check post-hoc power, JS p. 6
        pow_ = power_equivalence_oneway0(f_stat, n_groups, nobs, eps, res0.df)
        assert_allclose(pow_, 0.1552, atol=0.007)

        pow_ = power_equivalence_oneway(eps, eps, nobs.sum(),
                                        n_groups=n_groups, df=None, alpha=0.05,
                                        margin_type="wellek")
        assert_allclose(pow_, 0.05, atol=1e-13)

        nobs_t = nobs.sum()
        es = effectsize_oneway(means, vars_, nobs, use_var="unequal")
        es = np.sqrt(es)
        es_w0 = f2_to_wellek(es**2, n_groups)
        es_w = np.sqrt(fstat_to_wellek(f_stat, n_groups, nobs_t / n_groups))

        pow_ = power_equivalence_oneway(es_w, eps, nobs_t,
                                        n_groups=n_groups, df=None, alpha=0.05,
                                        margin_type="wellek")
        assert_allclose(pow_, 0.1552, atol=0.007)
        assert_allclose(es_w0, es_w, atol=0.007)

        margin = wellek_to_f2(eps, n_groups)
        pow_ = power_equivalence_oneway(es**2, margin, nobs_t,
                                        n_groups=n_groups, df=None, alpha=0.05,
                                        margin_type="f2")
        assert_allclose(pow_, 0.1552, atol=0.007)
