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
#from .test_weightstats import CheckPowerMixin
from statsmodels.stats.tests.test_weightstats import CheckPowerMixin, Holder


def test_normal_power_explicit():
    # a few initial test cases for NormalIndPower
    sigma = 1
    d = 0.3
    nobs = 80
    alpha = 0.05
    res1 = smp.normal_power(d, nobs/2., 0.05)
    res2 = smp.NormalIndPower().power(d, nobs, 0.05)
    res3 = smp.NormalIndPower().solve_power(effect_size=0.3, nobs1=80, alpha=0.05, power=None)
    res_R = 0.475100870572638
    assert_almost_equal(res1, res_R, decimal=13)
    assert_almost_equal(res2, res_R, decimal=13)
    assert_almost_equal(res3, res_R, decimal=13)


    norm_pow = smp.normal_power(-0.01, nobs/2., 0.05)
    norm_pow_R = 0.05045832927039234
    #value from R: >pwr.2p.test(h=0.01,n=80,sig.level=0.05,alternative="two.sided")
    assert_almost_equal(norm_pow, norm_pow_R, decimal=13)

    norm_pow = smp.NormalIndPower().power(0.01, nobs, 0.05,
                                          alternative="larger")
    norm_pow_R = 0.056869534873146124
    #value from R: >pwr.2p.test(h=0.01,n=80,sig.level=0.05,alternative="greater")
    assert_almost_equal(norm_pow, norm_pow_R, decimal=13)

    # Note: negative effect size is same as switching one-sided alternative
    # TODO: should I switch to larger/smaller instead of "one-sided" options
    norm_pow = smp.NormalIndPower().power(-0.01, nobs, 0.05,
                                          alternative="larger")
    norm_pow_R = 0.0438089705093578
    #value from R: >pwr.2p.test(h=0.01,n=80,sig.level=0.05,alternative="less")
    assert_almost_equal(norm_pow, norm_pow_R, decimal=13)

class TestNormalIndPower1(CheckPowerMixin):

    def __init__(self):

        #> example from above
        # results copied not directly from R
        res2 = Holder()
        res2.n = 80
        res2.d = 0.3
        res2.sig_level = 0.05
        res2.power = 0.475100870572638
        res2.alternative = 'two.sided'
        res2.note = 'NULL'
        res2.method = 'two sample power calculation'

        self.res2 = res2
        self.kwds = {'effect_size': res2.d, 'nobs1': res2.n,
                     'alpha': res2.sig_level, 'power':res2.power, 'ratio': 1}
        self.kwds_extra = {}
        self.cls = smp.NormalIndPower

class TestNormalIndPower2(CheckPowerMixin):

    def __init__(self):
        res2 = Holder()
        #> np = pwr.2p.test(h=0.01,n=80,sig.level=0.05,alternative="less")
        #> cat_items(np, "res2.")
        res2.h = 0.01
        res2.n = 80
        res2.sig_level = 0.05
        res2.power = 0.0438089705093578
        res2.alternative = 'less'
        res2.method = ('Difference of proportion power calculation for' +
                      ' binomial distribution (arcsine transformation)')
        res2.note = 'same sample sizes'

        self.res2 = res2
        self.kwds = {'effect_size': res2.h, 'nobs1': res2.n,
                     'alpha': res2.sig_level, 'power':res2.power, 'ratio': 1}
        self.kwds_extra = {'alternative':'smaller'}
        self.cls = smp.NormalIndPower

class TestNormalIndPower_onesamp1(CheckPowerMixin):

    def __init__(self):
        # forcing one-sample by using ratio=0
        #> example from above
        # results copied not directly from R
        res2 = Holder()
        res2.n = 40
        res2.d = 0.3
        res2.sig_level = 0.05
        res2.power = 0.475100870572638
        res2.alternative = 'two.sided'
        res2.note = 'NULL'
        res2.method = 'two sample power calculation'

        self.res2 = res2
        self.kwds = {'effect_size': res2.d, 'nobs1': res2.n,
                     'alpha': res2.sig_level, 'power':res2.power}
        # keyword for which we don't look for root:
        self.kwds_extra = {'ratio': 0}

        self.cls = smp.NormalIndPower

class TestNormalIndPower_onesamp2(CheckPowerMixin):
    # Note: same power as two sample case with twice as many observations

    def __init__(self):
        # forcing one-sample by using ratio=0
        res2 = Holder()
        #> np = pwr.norm.test(d=0.01,n=40,sig.level=0.05,alternative="less")
        #> cat_items(np, "res2.")
        res2.d = 0.01
        res2.n = 40
        res2.sig_level = 0.05
        res2.power = 0.0438089705093578
        res2.alternative = 'less'
        res2.method = 'Mean power calculation for normal distribution with known variance'

        self.res2 = res2
        self.kwds = {'effect_size': res2.d, 'nobs1': res2.n,
                     'alpha': res2.sig_level, 'power':res2.power}
        # keyword for which we don't look for root:
        self.kwds_extra = {'ratio': 0, 'alternative':'smaller'}

        self.cls = smp.NormalIndPower



class TestChisquarePower(CheckPowerMixin):

    def __init__(self):
        # one example from test_gof, results_power
        res2 = Holder()
        res2.w = 0.1
        res2.N = 5
        res2.df = 4
        res2.sig_level = 0.05
        res2.power = 0.05246644635810126
        res2.method = 'Chi squared power calculation'
        res2.note = 'N is the number of observations'

        self.res2 = res2
        self.kwds = {'effect_size': res2.w, 'nobs': res2.N,
                     'alpha': res2.sig_level, 'power':res2.power}
        # keyword for which we don't look for root:
        # solving for n_bins doesn't work, will not be used in regular usage
        self.kwds_extra = {'n_bins': res2.df + 1}

        self.cls = smp.GofChisquarePower


def test_ftest_power():
    #equivalence ftest, ttest

    for alpha in [0.01, 0.05, 0.1, 0.20, 0.50]:
        res0 = smp.ttest_power(0.01, 200, alpha)
        res1 = smp.ftest_power(0.01, 199, 1, alpha=alpha, ncc=0)
        assert_almost_equal(res1, res0, decimal=6)


    #example from Gplus documentation F-test ANOVA
    #Total sample size:200
    #Effect size "f":0.25
    #Beta/alpha ratio:1
    #Result:
    #Alpha:0.1592
    #Power (1-beta):0.8408
    #Critical F:1.4762
    #Lambda: 12.50000
    res1 = smp.ftest_anova_power(0.25, 200, 0.1592, k_groups=10)
    res0 = 0.8408
    assert_almost_equal(res1, res0, decimal=4)


    # TODO: no class yet
    # examples agains R::pwr
    res2 = Holder()
    #> rf = pwr.f2.test(u=5, v=199, f2=0.1**2, sig.level=0.01)
    #> cat_items(rf, "res2.")
    res2.u = 5
    res2.v = 199
    res2.f2 = 0.01
    res2.sig_level = 0.01
    res2.power = 0.0494137732920332
    res2.method = 'Multiple regression power calculation'

    res1 = smp.ftest_power(np.sqrt(res2.f2), res2.v, res2.u,
                           alpha=res2.sig_level, ncc=1)
    assert_almost_equal(res1, res2.power, decimal=5)

    res2 = Holder()
    #> rf = pwr.f2.test(u=5, v=199, f2=0.3**2, sig.level=0.01)
    #> cat_items(rf, "res2.")
    res2.u = 5
    res2.v = 199
    res2.f2 = 0.09
    res2.sig_level = 0.01
    res2.power = 0.7967191006290872
    res2.method = 'Multiple regression power calculation'

    res1 = smp.ftest_power(np.sqrt(res2.f2), res2.v, res2.u,
                           alpha=res2.sig_level, ncc=1)
    assert_almost_equal(res1, res2.power, decimal=5)

    res2 = Holder()
    #> rf = pwr.f2.test(u=5, v=19, f2=0.3**2, sig.level=0.1)
    #> cat_items(rf, "res2.")
    res2.u = 5
    res2.v = 19
    res2.f2 = 0.09
    res2.sig_level = 0.1
    res2.power = 0.235454222377575
    res2.method = 'Multiple regression power calculation'

    res1 = smp.ftest_power(np.sqrt(res2.f2), res2.v, res2.u,
                           alpha=res2.sig_level, ncc=1)
    assert_almost_equal(res1, res2.power, decimal=5)

# class based version of two above test for Ftest
class TestFtestAnovaPower(CheckPowerMixin):

    def __init__(self):
        res2 = Holder()
        #example from Gplus documentation F-test ANOVA
        #Total sample size:200
        #Effect size "f":0.25
        #Beta/alpha ratio:1
        #Result:
        #Alpha:0.1592
        #Power (1-beta):0.8408
        #Critical F:1.4762
        #Lambda: 12.50000
        #converted to res2 by hand
        res2.f = 0.25
        res2.n = 200
        res2.k = 10
        res2.alpha = 0.1592
        res2.power = 0.8408
        res2.method = 'Multiple regression power calculation'

        self.res2 = res2
        self.kwds = {'effect_size': res2.f, 'nobs': res2.n,
                     'alpha': res2.alpha, 'power':res2.power, 'k_groups': res2.k}
        # keyword for which we don't look for root:
        # solving for n_bins doesn't work, will not be used in regular usage
        self.kwds_extra = {}

        self.cls = smp.FTestAnovaPower

class TestFtestPower(CheckPowerMixin):

    def __init__(self):
        res2 = Holder()
        #> rf = pwr.f2.test(u=5, v=19, f2=0.3**2, sig.level=0.1)
        #> cat_items(rf, "res2.")
        res2.u = 5
        res2.v = 19
        res2.f2 = 0.09
        res2.sig_level = 0.1
        res2.power = 0.235454222377575
        res2.method = 'Multiple regression power calculation'

        self.res2 = res2
        self.kwds = {'effect_size': np.sqrt(res2.f2), 'df_num': res2.v,
                     'df_denom': res2.u, 'alpha': res2.sig_level,
                     'power':res2.power}
        # keyword for which we don't look for root:
        # solving for n_bins doesn't work, will not be used in regular usage
        self.kwds_extra = {}

        self.cls = smp.FTestPower


if __name__ == '__main__':
    test_normal_power_explicit()
    nt = TestNormalIndPower1()
    nt.test_power()
    nt.test_roots()
    nt = TestNormalIndPower_onesamp1()
    nt.test_power()
    nt.test_roots()
