'''tests for weightstats, compares with replication

no failures but needs cleanup

Author: Josef Perktod (josef-pktd)
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
        assert_almost_equal(x2r.var(), d2w.var, 14)
        assert_almost_equal(x2r.std(), d2w.std, 14)


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
