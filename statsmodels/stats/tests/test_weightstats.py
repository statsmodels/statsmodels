'''tests for weightstats, compares with replication

no failures but needs cleanup
update 2012-09-09:
   added test after fixing bug in covariance
   TODOs:
     - I don't remember what all the commented out code is doing
     - should be refactored to use generator or inherited tests
     - still gaps in test coverage
       - value/diff in ttest_ind is tested in test_tost.py
     - what about pandas data structures?

Author: Josef Perktold
License: BSD (3-clause)

'''

import copy

import numpy as np
from scipy import stats
from numpy.testing import assert_almost_equal, assert_equal
from statsmodels.stats.weightstats import \
                DescrStatsW, CompareMeans, ttest_ind
from statsmodels.stats.power import TTestPower, TTestIndPower

class Holder(object):
    pass

class TestWeightstats(object):

    def __init__(self):
        np.random.seed(9876789)
        n1, n2 = 20,20
        m1, m2 = 1, 1.2
        x1 = m1 + np.random.randn(n1)
        x2 = m2 + np.random.randn(n2)
        x1_2d = m1 + np.random.randn(n1, 3)
        x2_2d = m2 + np.random.randn(n2, 3)
        w1_ = 2. * np.ones(n1)
        w2_ = 2. * np.ones(n2)
        w1 = np.random.randint(1,4, n1)
        w2 = np.random.randint(1,4, n2)
        self.x1, self.x2 = x1, x2
        self.w1, self.w2 = w1, w2
        self.x1_2d, self.x2_2d = x1_2d, x2_2d

    def test_weightstats_1(self):
        x1, x2 = self.x1, self.x2
        w1, w2 = self.w1, self.w2
        w1_ = 2. * np.ones(len(x1))
        w2_ = 2. * np.ones(len(x2))

        d1 = DescrStatsW(x1)
#        print ttest_ind(x1, x2)
#        print ttest_ind(x1, x2, usevar='separate')
#        #print ttest_ind(x1, x2, usevar='separate')
#        print stats.ttest_ind(x1, x2)
#        print ttest_ind(x1, x2, usevar='separate', alternative='larger')
#        print ttest_ind(x1, x2, usevar='separate', alternative='smaller')
#        print ttest_ind(x1, x2, usevar='separate', weights=(w1_, w2_))
#        print stats.ttest_ind(np.r_[x1, x1], np.r_[x2,x2])
        assert_almost_equal(ttest_ind(x1, x2, weights=(w1_, w2_))[:2],
                            stats.ttest_ind(np.r_[x1, x1], np.r_[x2,x2]))

    def test_weightstats_2(self):
        x1, x2 = self.x1, self.x2
        w1, w2 = self.w1, self.w2

        d1 = DescrStatsW(x1)
        d1w = DescrStatsW(x1, weights=w1)
        d2w = DescrStatsW(x2, weights=w2)
        x1r = d1w.asrepeats()
        x2r = d2w.asrepeats()
#        print 'random weights'
#        print ttest_ind(x1, x2, weights=(w1, w2))
#        print stats.ttest_ind(x1r, x2r)
        assert_almost_equal(ttest_ind(x1, x2, weights=(w1, w2))[:2],
                            stats.ttest_ind(x1r, x2r), 14)
        #not the same as new version with random weights/replication
#        assert x1r.shape[0] == d1w.sum_weights
#        assert x2r.shape[0] == d2w.sum_weights

        assert_almost_equal(x2r.mean(0), d2w.mean, 14)
        assert_almost_equal(x2r.var(), d2w.var, 14)
        assert_almost_equal(x2r.std(), d2w.std, 14)
        #note: the following is for 1d
        assert_almost_equal(np.cov(x2r, bias=1), d2w.cov, 14)
        #assert_almost_equal(np.corrcoef(np.x2r), d2w.corrcoef, 19)
        #TODO: exception in corrcoef (scalar case)


        #one-sample tests
#        print d1.ttest_mean(3)
#        print stats.ttest_1samp(x1, 3)
#        print d1w.ttest_mean(3)
#        print stats.ttest_1samp(x1r, 3)
        assert_almost_equal(d1.ttest_mean(3)[:2], stats.ttest_1samp(x1, 3), 11)
        assert_almost_equal(d1w.ttest_mean(3)[:2], stats.ttest_1samp(x1r, 3), 11)

    def test_weightstats_3(self):
        x1_2d, x2_2d = self.x1_2d, self.x2_2d
        w1, w2 = self.w1, self.w2

        d1w_2d = DescrStatsW(x1_2d, weights=w1)
        d2w_2d = DescrStatsW(x2_2d, weights=w2)
        x1r_2d = d1w_2d.asrepeats()
        x2r_2d = d2w_2d.asrepeats()

        assert_almost_equal(x2r_2d.mean(0), d2w_2d.mean, 14)
        assert_almost_equal(x2r_2d.var(0), d2w_2d.var, 14)
        assert_almost_equal(x2r_2d.std(0), d2w_2d.std, 14)
        assert_almost_equal(np.cov(x2r_2d.T, bias=1), d2w_2d.cov, 14)
        assert_almost_equal(np.corrcoef(x2r_2d.T), d2w_2d.corrcoef, 14)

#        print d1w_2d.ttest_mean(3)
#        #scipy.stats.ttest is also vectorized
#        print stats.ttest_1samp(x1r_2d, 3)
        t,p,d = d1w_2d.ttest_mean(3)
        assert_almost_equal([t, p], stats.ttest_1samp(x1r_2d, 3), 11)
        #print [stats.ttest_1samp(xi, 3) for xi in x1r_2d.T]
        cm = CompareMeans(d1w_2d, d2w_2d)
        ressm = cm.ttest_ind()
        resss = stats.ttest_ind(x1r_2d, x2r_2d)
        assert_almost_equal(ressm[:2], resss, 14)

##        #doesn't work for 2d, levene doesn't use weights
##        cm = CompareMeans(d1w_2d, d2w_2d)
##        ressm = cm.test_equal_var()
##        resss = stats.levene(x1r_2d, x2r_2d)
##        assert_almost_equal(ressm[:2], resss, 14)

    def test_weightstats_ddof_tests(self):
        # explicit test that ttest and confint are independent of ddof
        # one sample case
        x1_2d = self.x1_2d
        w1 = self.w1

        d1w_d0 = DescrStatsW(x1_2d, weights=w1, ddof=0)
        d1w_d1 = DescrStatsW(x1_2d, weights=w1, ddof=1)
        d1w_d2 = DescrStatsW(x1_2d, weights=w1, ddof=2)

        #check confint independent of user ddof
        res0 = d1w_d0.ttest_mean()
        res1 = d1w_d1.ttest_mean()
        res2 = d1w_d2.ttest_mean()
        # concatenate into one array with np.r_
        assert_almost_equal(np.r_[res1], np.r_[res0], 14)
        assert_almost_equal(np.r_[res2], np.r_[res0], 14)

        res0 = d1w_d0.ttest_mean(0.5)
        res1 = d1w_d1.ttest_mean(0.5)
        res2 = d1w_d2.ttest_mean(0.5)
        assert_almost_equal(np.r_[res1], np.r_[res0], 14)
        assert_almost_equal(np.r_[res2], np.r_[res0], 14)

        #check confint independent of user ddof
        res0 = d1w_d0.confint_mean()
        res1 = d1w_d1.confint_mean()
        res2 = d1w_d2.confint_mean()
        assert_almost_equal(res1, res0, 14)
        assert_almost_equal(res2, res0, 14)


class CheckWeightstats1dMixin(object):

    def test_basic(self):
        x1r = self.x1r
        d1w = self.d1w

        assert_almost_equal(x1r.mean(0), d1w.mean, 14)
        assert_almost_equal(x1r.var(0, ddof=d1w.ddof), d1w.var, 14)
        assert_almost_equal(x1r.std(0, ddof=d1w.ddof), d1w.std, 14)
        var1 = d1w.var_ddof(ddof=1)
        assert_almost_equal(x1r.var(0, ddof=1), var1, 14)
        std1 = d1w.std_ddof(ddof=1)
        assert_almost_equal(x1r.std(0, ddof=1), std1, 14)


        assert_almost_equal(np.cov(x1r.T, bias=1-d1w.ddof), d1w.cov, 14)

        #
        #assert_almost_equal(np.corrcoef(x1r.T), d1w.corrcoef, 14)

    def test_ttest(self):
        x1r = self.x1r
        d1w = self.d1w
        assert_almost_equal(d1w.ttest_mean(3)[:2],
                            stats.ttest_1samp(x1r, 3), 11)

#    def
#        assert_almost_equal(ttest_ind(x1, x2, weights=(w1, w2))[:2],
#                            stats.ttest_ind(x1r, x2r), 14)

    def test_ttest_2sample(self):
        x1, x2 = self.x1, self.x2
        x1r, x2r = self.x1r, self.x2r
        w1, w2 = self.w1, self.w2

        #Note: stats.ttest_ind handles 2d/nd arguments
        res_sp = stats.ttest_ind(x1r, x2r)
        assert_almost_equal(ttest_ind(x1, x2, weights=(w1, w2))[:2],
                            res_sp, 14)

        #check correct ttest independent of user ddof
        cm = CompareMeans(DescrStatsW(x1, weights=w1, ddof=0),
                          DescrStatsW(x2, weights=w2, ddof=1))
        assert_almost_equal(cm.ttest_ind()[:2], res_sp, 14)

        cm = CompareMeans(DescrStatsW(x1, weights=w1, ddof=1),
                          DescrStatsW(x2, weights=w2, ddof=2))
        assert_almost_equal(cm.ttest_ind()[:2], res_sp, 14)


        cm0 = CompareMeans(DescrStatsW(x1, weights=w1, ddof=0),
                          DescrStatsW(x2, weights=w2, ddof=0))
        cm1 = CompareMeans(DescrStatsW(x1, weights=w1, ddof=0),
                          DescrStatsW(x2, weights=w2, ddof=1))
        cm2 = CompareMeans(DescrStatsW(x1, weights=w1, ddof=1),
                          DescrStatsW(x2, weights=w2, ddof=2))

        res0 = cm0.ttest_ind(usevar='separate')
        res1 = cm1.ttest_ind(usevar='separate')
        res2 = cm2.ttest_ind(usevar='separate')
        assert_almost_equal(res1, res0, 14)
        assert_almost_equal(res2, res0, 14)

        #check confint independent of user ddof
        res0 = cm0.confint_diff(usevar='pooled')
        res1 = cm1.confint_diff(usevar='pooled')
        res2 = cm2.confint_diff(usevar='pooled')
        assert_almost_equal(res1, res0, 14)
        assert_almost_equal(res2, res0, 14)

        res0 = cm0.confint_diff(usevar='separate')
        res1 = cm1.confint_diff(usevar='separate')
        res2 = cm2.confint_diff(usevar='separate')
        assert_almost_equal(res1, res0, 14)
        assert_almost_equal(res2, res0, 14)


    def test_confint_mean(self):
        #compare confint_mean with ttest
        d1w = self.d1w
        alpha = 0.05
        low, upp = d1w.confint_mean()
        t, p, d = d1w.ttest_mean(low)
        assert_almost_equal(p, alpha * np.ones(p.shape), 8)
        t, p, d = d1w.ttest_mean(upp)
        assert_almost_equal(p, alpha * np.ones(p.shape), 8)
        t, p, d = d1w.ttest_mean(np.vstack((low, upp)))
        assert_almost_equal(p, alpha * np.ones(p.shape), 8)

class CheckWeightstats2dMixin(CheckWeightstats1dMixin):

    def test_corr(self):
        x1r = self.x1r
        d1w = self.d1w

        assert_almost_equal(np.corrcoef(x1r.T), d1w.corrcoef, 14)


class TestWeightstats1d_ddof(CheckWeightstats1dMixin):

    @classmethod
    def setup_class(self):
        np.random.seed(9876789)
        n1, n2 = 20,20
        m1, m2 = 1, 1.2
        x1 = m1 + np.random.randn(n1, 1)
        x2 = m2 + np.random.randn(n2, 1)
        w1 = np.random.randint(1,4, n1)
        w2 = np.random.randint(1,4, n2)

        self.x1, self.x2 = x1, x2
        self.w1, self.w2 = w1, w2
        self.d1w = DescrStatsW(x1, weights=w1, ddof=1)
        self.d2w = DescrStatsW(x2, weights=w2, ddof=1)
        self.x1r = self.d1w.asrepeats()
        self.x2r = self.d2w.asrepeats()


class TestWeightstats2d(CheckWeightstats2dMixin):

    @classmethod
    def setup_class(self):
        np.random.seed(9876789)
        n1, n2 = 20,20
        m1, m2 = 1, 1.2
        x1 = m1 + np.random.randn(n1, 3)
        x2 = m2 + np.random.randn(n2, 3)
        w1_ = 2. * np.ones(n1)
        w2_ = 2. * np.ones(n2)
        w1 = np.random.randint(1,4, n1)
        w2 = np.random.randint(1,4, n2)
        self.x1, self.x2 = x1, x2
        self.w1, self.w2 = w1, w2

        self.d1w = DescrStatsW(x1, weights=w1)
        self.d2w = DescrStatsW(x2, weights=w2)
        self.x1r = self.d1w.asrepeats()
        self.x2r = self.d2w.asrepeats()

class TestWeightstats2d_ddof(CheckWeightstats2dMixin):

    @classmethod
    def setup_class(self):
        np.random.seed(9876789)
        n1, n2 = 20,20
        m1, m2 = 1, 1.2
        x1 = m1 + np.random.randn(n1, 3)
        x2 = m2 + np.random.randn(n2, 3)
        w1 = np.random.randint(1,4, n1)
        w2 = np.random.randint(1,4, n2)

        self.x1, self.x2 = x1, x2
        self.w1, self.w2 = w1, w2
        self.d1w = DescrStatsW(x1, weights=w1, ddof=1)
        self.d2w = DescrStatsW(x2, weights=w2, ddof=1)
        self.x1r = self.d1w.asrepeats()
        self.x2r = self.d2w.asrepeats()

class TestWeightstats2d_nobs(CheckWeightstats2dMixin):

    @classmethod
    def setup_class(self):
        np.random.seed(9876789)
        n1, n2 = 20,30
        m1, m2 = 1, 1.2
        x1 = m1 + np.random.randn(n1, 3)
        x2 = m2 + np.random.randn(n2, 3)
        w1 = np.random.randint(1,4, n1)
        w2 = np.random.randint(1,4, n2)

        self.x1, self.x2 = x1, x2
        self.w1, self.w2 = w1, w2
        self.d1w = DescrStatsW(x1, weights=w1, ddof=0)
        self.d2w = DescrStatsW(x2, weights=w2, ddof=1)
        self.x1r = self.d1w.asrepeats()
        self.x2r = self.d2w.asrepeats()

def test_ttest_ind_with_uneq_var():

    #from scipy
    # check vs. R
    a = (1, 2, 3)
    b = (1.1, 2.9, 4.2)
    pr = 0.53619490753126731
    tr = -0.68649512735572582
    t, p, df = ttest_ind(a, b, usevar='separate')
    assert_almost_equal([t,p], [tr, pr], 13)

    a = (1, 2, 3, 4)
    pr = 0.84354139131608286
    tr = -0.2108663315950719
    t, p, df = ttest_ind(a, b, usevar='separate')
    assert_almost_equal([t,p], [tr, pr], 13)


class CheckPowerMixin(object):

    def test_power(self):
        #test against R results
        kwds = copy.copy(self.kwds)
        del kwds['beta']
        kwds.update(self.kwds_extra)
        res1 = self.cls()
        assert_almost_equal(res1.power(**kwds), self.res2.power, decimal=6)

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
            assert_almost_equal(result, value, decimal=3, err_msg=key+' failed')
            kwds[key] = value  #reset dict

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
                     'alpha': res2.sig_level, 'beta':res2.power}
        self.kwds_extra = {}
        self.cls = TTestPower

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
                     'alpha': res2.sig_level, 'beta':res2.power}
        self.kwds_extra = {}
        self.cls = TTestPower

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
                     'alpha': res2.sig_level, 'beta': res2.power}
        self.kwds_extra = {'alternative': 'larger'}
        self.cls = TTestPower

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
                     'alpha': res2.sig_level, 'beta': res2.power}
        self.kwds_extra = {'alternative': 'larger'}
        self.cls = TTestPower

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
                     'alpha': res2.sig_level, 'beta': res2.power}
        self.kwds_extra = {'alternative': 'smaller'}
        self.cls = TTestPower

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
                     'alpha': res2.sig_level, 'beta': res2.power}
        self.kwds_extra = {'alternative': 'smaller'}
        self.cls = TTestPower


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
                     'alpha': res2.sig_level, 'beta': res2.power, 'ratio': 1}
        self.kwds_extra = {}
        self.cls = TTestIndPower

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
                     'alpha': res2.sig_level, 'beta': res2.power, 'ratio': 1}
        self.kwds_extra = {}
        self.cls = TTestIndPower

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
                     'alpha': res2.sig_level, 'beta':res2.power, 'ratio': 1}
        self.kwds_extra = {'alternative': 'larger'}
        self.cls = TTestIndPower

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
                     'alpha': res2.sig_level, 'beta':res2.power}
        self.kwds_extra = {'alternative': 'larger'}
        self.cls = TTestIndPower

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
                     'alpha': res2.sig_level, 'beta':res2.power, 'ratio': 1.5}
        self.kwds_extra = {'alternative': 'two-sided'}
        self.cls = TTestIndPower

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
                     'alpha': res2.sig_level, 'beta':res2.power, 'ratio': 1.5}
        self.kwds_extra = {'alternative': 'larger'}
        self.cls = TTestIndPower


#Note: compared to R power, I only added one-sided which is the same as greater
'''
#------------
> p = pwr.t.test(d=1,n=30,sig.level=0.05,type="one.sample",alternative="less")
> cat_items(p, prefix='tt_power1_1l.')
tt_power1_1l.n = 30
tt_power1_1l.d = 1
tt_power1_1l.sig_level = 0.05
tt_power1_1l.power = 1.277755679041093e-12
tt_power1_1l.alternative = 'less'
tt_power1_1l.note = 'NULL'
tt_power1_1l.method = 'One-sample t test power calculation'
> p = pwr.t.test(d=1,n=30,sig.level=0.05,type="two.sample",alternative="less")
> cat_items(p, prefix='tt_power2_1l.')
tt_power2_1l.n = 30
tt_power2_1l.d = 1
tt_power2_1l.sig_level = 0.05
tt_power2_1l.power = 2.203160931468773e-08
tt_power2_1l.alternative = 'less'
tt_power2_1l.note = 'n is number in *each* group'
tt_power2_1l.method = 'Two-sample t test power calculation'

'''

