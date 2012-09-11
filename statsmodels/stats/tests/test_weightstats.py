'''tests for weightstats, compares with replication

no failures but needs cleanup
update 2012-09-09:
   added test after fixing bug in covariance
   TODOs:
     - I don't remember what all the commented out code is doing
     - should be refactored to use generator or inherited tests
     - still gaps in test coverage
     - what about pandas data structures?

Author: Josef Perktold
License: BSD (3-clause)

'''


import numpy as np
from scipy import stats
from numpy.testing import assert_almost_equal, assert_equal
from statsmodels.stats.weightstats import \
                DescrStatsW, CompareMeans, ttest_ind

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
        ressm = CompareMeans(d1w_2d, d2w_2d).ttest_ind()
        resss = stats.ttest_ind(x1r_2d, x2r_2d)
        assert_almost_equal(ressm[:2], resss, 14)
#        print ressm
#        print resss


class CheckWeightstats1d(object):

    def test_basic(self):
        x1r = self.x1r
        d1w = self.d1w

        assert_almost_equal(x1r.mean(0), d1w.mean, 14)
        assert_almost_equal(x1r.var(0, ddof=d1w.ddof), d1w.var, 14)
        assert_almost_equal(x1r.std(0, ddof=d1w.ddof), d1w.std, 14)


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
        assert_almost_equal(ttest_ind(x1, x2, weights=(w1, w2))[:2],
                            stats.ttest_ind(x1r, x2r), 14)

    def test_confint_mean(self):
        #compare confint_mean with ttest
        d1w = self.d1w
        alpha = 0.95
        low, upp = d1w.confint_mean()
        t, p, d = d1w.ttest_mean(low)
        assert_almost_equal(p, alpha * np.ones(p.shape), 10)
        t, p, d = d1w.ttest_mean(upp)
        assert_almost_equal(p, alpha * np.ones(p.shape), 10)
        t, p, d = d1w.ttest_mean(np.vstack((low, upp)))
        assert_almost_equal(p, alpha * np.ones(p.shape), 10)

class CheckWeightstats2d(CheckWeightstats1d):

    def test_corr(self):
        x1r = self.x1r
        d1w = self.d1w

        assert_almost_equal(np.corrcoef(x1r.T), d1w.corrcoef, 14)


class TestWeightstats1d_ddof(CheckWeightstats1d):

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


class TestWeightstats2d(CheckWeightstats2d):

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

class TestWeightstats2d_ddof(CheckWeightstats2d):

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

