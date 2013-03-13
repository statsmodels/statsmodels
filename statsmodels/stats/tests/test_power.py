# -*- coding: utf-8 -*-
"""Tests for statistical power calculations

Note:
    tests for chisquare power are in test_gof.py

Created on Sat Mar 09 08:44:49 2013

Author: Josef Perktold
"""

import copy

import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose

import statsmodels.stats.power as smp
#from .test_weightstats import CheckPowerMixin
from statsmodels.stats.tests.test_weightstats import Holder

class CheckPowerMixin(object):

    def test_power(self):
        #test against R results
        kwds = copy.copy(self.kwds)
        del kwds['power']
        kwds.update(self.kwds_extra)
        if hasattr(self, 'decimal'):
            decimal = self.decimal
        else:
            decimal = 6
        res1 = self.cls()
        assert_almost_equal(res1.power(**kwds), self.res2.power, decimal=decimal)

    def test_roots(self):
        kwds = copy.copy(self.kwds)
        kwds.update(self.kwds_extra)

        # kwds_extra are used as argument, but not as target for root
        for key in self.kwds:
            # keep print to check whether tests are really executed
            #print 'testing roots', key
            value = kwds[key]
            kwds[key] = None

            result = self.cls().solve_power(**kwds)
            assert_allclose(result, value, rtol=0.001, err_msg=key+' failed')
            # yield can be used to investigate specific errors
            #yield assert_allclose, result, value, 0.001, 0, key+' failed'
            kwds[key] = value  # reset dict

#''' test cases
#one sample
#               two-sided one-sided
#large power     OneS1      OneS3
#small power     OneS2      OneS4
#
#two sample
#               two-sided one-sided
#large power     TwoS1       TwoS3
#small power     TwoS2       TwoS4
#small p, ratio  TwoS4       TwoS5
#'''

class TestTTPowerOneS1(CheckPowerMixin):

    def __init__(self):

        #> p = pwr.t.test(d=1,n=30,sig.level=0.05,type="two.sample",alternative="two.sided")
        #> cat_items(p, prefix='tt_power2_1.')
        res2 = Holder()
        res2.n = 30
        res2.d = 1
        res2.sig_level = 0.05
        res2.power = 0.9995636009612725
        res2.alternative = 'two.sided'
        res2.note = 'NULL'
        res2.method = 'One-sample t test power calculation'

        self.res2 = res2
        self.kwds = {'effect_size': res2.d, 'nobs': res2.n,
                     'alpha': res2.sig_level, 'power':res2.power}
        self.kwds_extra = {}
        self.cls = smp.TTestPower

class TestTTPowerOneS2(CheckPowerMixin):
    # case with small power

    def __init__(self):

        res2 = Holder()
        #> p = pwr.t.test(d=0.2,n=20,sig.level=0.05,type="one.sample",alternative="two.sided")
        #> cat_items(p, "res2.")
        res2.n = 20
        res2.d = 0.2
        res2.sig_level = 0.05
        res2.power = 0.1359562887679666
        res2.alternative = 'two.sided'
        res2.note = '''NULL'''
        res2.method = 'One-sample t test power calculation'

        self.res2 = res2
        self.kwds = {'effect_size': res2.d, 'nobs': res2.n,
                     'alpha': res2.sig_level, 'power':res2.power}
        self.kwds_extra = {}
        self.cls = smp.TTestPower

class TestTTPowerOneS3(CheckPowerMixin):

    def __init__(self):

        res2 = Holder()
        #> p = pwr.t.test(d=1,n=30,sig.level=0.05,type="one.sample",alternative="greater")
        #> cat_items(p, prefix='tt_power1_1g.')
        res2.n = 30
        res2.d = 1
        res2.sig_level = 0.05
        res2.power = 0.999892010204909
        res2.alternative = 'greater'
        res2.note = 'NULL'
        res2.method = 'One-sample t test power calculation'

        self.res2 = res2
        self.kwds = {'effect_size': res2.d, 'nobs': res2.n,
                     'alpha': res2.sig_level, 'power': res2.power}
        self.kwds_extra = {'alternative': 'larger'}
        self.cls = smp.TTestPower

class TestTTPowerOneS4(CheckPowerMixin):

    def __init__(self):

        res2 = Holder()
        #> p = pwr.t.test(d=0.05,n=20,sig.level=0.05,type="one.sample",alternative="greater")
        #> cat_items(p, "res2.")
        res2.n = 20
        res2.d = 0.05
        res2.sig_level = 0.05
        res2.power = 0.0764888785042198
        res2.alternative = 'greater'
        res2.note = '''NULL'''
        res2.method = 'One-sample t test power calculation'

        self.res2 = res2
        self.kwds = {'effect_size': res2.d, 'nobs': res2.n,
                     'alpha': res2.sig_level, 'power': res2.power}
        self.kwds_extra = {'alternative': 'larger'}
        self.cls = smp.TTestPower

class TestTTPowerOneS5(CheckPowerMixin):
    # case one-sided less, not implemented yet

    def __init__(self):

        res2 = Holder()
        #> p = pwr.t.test(d=0.2,n=20,sig.level=0.05,type="one.sample",alternative="less")
        #> cat_items(p, "res2.")
        res2.n = 20
        res2.d = 0.2
        res2.sig_level = 0.05
        res2.power = 0.006063932667926375
        res2.alternative = 'less'
        res2.note = '''NULL'''
        res2.method = 'One-sample t test power calculation'

        self.res2 = res2
        self.kwds = {'effect_size': res2.d, 'nobs': res2.n,
                     'alpha': res2.sig_level, 'power': res2.power}
        self.kwds_extra = {'alternative': 'smaller'}
        self.cls = smp.TTestPower

class TestTTPowerOneS6(CheckPowerMixin):
    # case one-sided less, negative effect size, not implemented yet

    def __init__(self):

        res2 = Holder()
        #> p = pwr.t.test(d=-0.2,n=20,sig.level=0.05,type="one.sample",alternative="less")
        #> cat_items(p, "res2.")
        res2.n = 20
        res2.d = -0.2
        res2.sig_level = 0.05
        res2.power = 0.21707518167191
        res2.alternative = 'less'
        res2.note = '''NULL'''
        res2.method = 'One-sample t test power calculation'

        self.res2 = res2
        self.kwds = {'effect_size': res2.d, 'nobs': res2.n,
                     'alpha': res2.sig_level, 'power': res2.power}
        self.kwds_extra = {'alternative': 'smaller'}
        self.cls = smp.TTestPower


class TestTTPowerTwoS1(CheckPowerMixin):

    def __init__(self):

        #> p = pwr.t.test(d=1,n=30,sig.level=0.05,type="two.sample",alternative="two.sided")
        #> cat_items(p, prefix='tt_power2_1.')
        res2 = Holder()
        res2.n = 30
        res2.d = 1
        res2.sig_level = 0.05
        res2.power = 0.967708258242517
        res2.alternative = 'two.sided'
        res2.note = 'n is number in *each* group'
        res2.method = 'Two-sample t test power calculation'

        self.res2 = res2
        self.kwds = {'effect_size': res2.d, 'nobs1': res2.n,
                     'alpha': res2.sig_level, 'power': res2.power, 'ratio': 1}
        self.kwds_extra = {}
        self.cls = smp.TTestIndPower

class TestTTPowerTwoS2(CheckPowerMixin):

    def __init__(self):

        res2 = Holder()
        #> p = pwr.t.test(d=0.1,n=20,sig.level=0.05,type="two.sample",alternative="two.sided")
        #> cat_items(p, "res2.")
        res2.n = 20
        res2.d = 0.1
        res2.sig_level = 0.05
        res2.power = 0.06095912465411235
        res2.alternative = 'two.sided'
        res2.note = 'n is number in *each* group'
        res2.method = 'Two-sample t test power calculation'

        self.res2 = res2
        self.kwds = {'effect_size': res2.d, 'nobs1': res2.n,
                     'alpha': res2.sig_level, 'power': res2.power, 'ratio': 1}
        self.kwds_extra = {}
        self.cls = smp.TTestIndPower

class TestTTPowerTwoS3(CheckPowerMixin):

    def __init__(self):

        res2 = Holder()
        #> p = pwr.t.test(d=1,n=30,sig.level=0.05,type="two.sample",alternative="greater")
        #> cat_items(p, prefix='tt_power2_1g.')
        res2.n = 30
        res2.d = 1
        res2.sig_level = 0.05
        res2.power = 0.985459690251624
        res2.alternative = 'greater'
        res2.note = 'n is number in *each* group'
        res2.method = 'Two-sample t test power calculation'

        self.res2 = res2
        self.kwds = {'effect_size': res2.d, 'nobs1': res2.n,
                     'alpha': res2.sig_level, 'power':res2.power, 'ratio': 1}
        self.kwds_extra = {'alternative': 'larger'}
        self.cls = smp.TTestIndPower

class TestTTPowerTwoS4(CheckPowerMixin):
    # case with small power

    def __init__(self):

        res2 = Holder()
        #> p = pwr.t.test(d=0.01,n=30,sig.level=0.05,type="two.sample",alternative="greater")
        #> cat_items(p, "res2.")
        res2.n = 30
        res2.d = 0.01
        res2.sig_level = 0.05
        res2.power = 0.0540740302835667
        res2.alternative = 'greater'
        res2.note = 'n is number in *each* group'
        res2.method = 'Two-sample t test power calculation'

        self.res2 = res2
        self.kwds = {'effect_size': res2.d, 'nobs1': res2.n,
                     'alpha': res2.sig_level, 'power':res2.power}
        self.kwds_extra = {'alternative': 'larger'}
        self.cls = smp.TTestIndPower

class TestTTPowerTwoS5(CheckPowerMixin):
    # case with unequal n, ratio>1

    def __init__(self):

        res2 = Holder()
        #> p = pwr.t2n.test(d=0.1,n1=20, n2=30,sig.level=0.05,alternative="two.sided")
        #> cat_items(p, "res2.")
        res2.n1 = 20
        res2.n2 = 30
        res2.d = 0.1
        res2.sig_level = 0.05
        res2.power = 0.0633081832564667
        res2.alternative = 'two.sided'
        res2.method = 't test power calculation'

        self.res2 = res2
        self.kwds = {'effect_size': res2.d, 'nobs1': res2.n1,
                     'alpha': res2.sig_level, 'power':res2.power, 'ratio': 1.5}
        self.kwds_extra = {'alternative': 'two-sided'}
        self.cls = smp.TTestIndPower

class TestTTPowerTwoS6(CheckPowerMixin):
    # case with unequal n, ratio>1

    def __init__(self):

        res2 = Holder()
        #> p = pwr.t2n.test(d=0.1,n1=20, n2=30,sig.level=0.05,alternative="greater")
        #> cat_items(p, "res2.")
        res2.n1 = 20
        res2.n2 = 30
        res2.d = 0.1
        res2.sig_level = 0.05
        res2.power = 0.09623589080917805
        res2.alternative = 'greater'
        res2.method = 't test power calculation'

        self.res2 = res2
        self.kwds = {'effect_size': res2.d, 'nobs1': res2.n1,
                     'alpha': res2.sig_level, 'power':res2.power, 'ratio': 1.5}
        self.kwds_extra = {'alternative': 'larger'}
        self.cls = smp.TTestIndPower



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
                     'alpha': res2.alpha, 'power': res2.power}
        # keyword for which we don't look for root:
        # solving for n_bins doesn't work, will not be used in regular usage
        self.kwds_extra = {'k_groups': res2.k} # rootfinding doesn't work
        self.cls = smp.FTestAnovaPower
        # precision for test_power
        self.decimal = 4

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
                     'power': res2.power}
        # keyword for which we don't look for root:
        # solving for n_bins doesn't work, will not be used in regular usage
        self.kwds_extra = {}
        self.cls = smp.FTestPower
        # precision for test_power
        self.decimal = 5


if __name__ == '__main__':
    test_normal_power_explicit()
    nt = TestNormalIndPower1()
    nt.test_power()
    nt.test_roots()
    nt = TestNormalIndPower_onesamp1()
    nt.test_power()
    nt.test_roots()
