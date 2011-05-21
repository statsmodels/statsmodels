# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 15:02:13 2011
@author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_almost_equal

from scikits.statsmodels.sandbox.distributions.multivariate import (
                mvstdtprob, mvstdnormcdf)

class Test_MVN_MVT_prob(object):
    #test for block integratal, cdf, of multivariate t and normal
    #comparison results from R

    def __init__(self):
        self.corr_equal = np.asarray([[1.0, 0.5, 0.5],[0.5,1,0.5],[0.5,0.5,1]])
        self.a = -1 * np.ones(3)
        self.b = 3 * np.ones(3)
        self.df = 4

        corr2 = self.corr_equal.copy()
        corr2[2,1] = -0.5
        self.corr2 = corr2


    def test_mvn_mvt_1(self):
        a, b = self.a, self.b
        df = self.df
        corr_equal = self.corr_equal
        #result from R, mvtnorm with option
        #algorithm = GenzBretz(maxpts = 100000, abseps = 0.000001, releps = 0)
        #     or higher
        probmvt_R = 0.60414   #report, ed error approx. 7.5e-06
        probmvn_R = 0.673970  #reported error approx. 6.4e-07
        assert_almost_equal(probmvt_R, mvstdtprob(a, b, corr_equal, df), 4)
        assert_almost_equal(probmvn_R,
                            mvstdnormcdf(a, b, corr_equal, abseps=1e-5), 4)

        mvn_high = mvstdnormcdf(a, b, corr_equal, abseps=1e-8, maxpts=10000000)
        assert_almost_equal(probmvn_R, mvn_high, 5)
        #this still barely fails sometimes at 6 why?? error is -7.2627419411830374e-007
        #>>> 0.67396999999999996 - 0.67397072627419408
        #-7.2627419411830374e-007
        #>>> assert_almost_equal(0.67396999999999996, 0.67397072627419408, 6)
        #Fail

    def test_mvn_mvt_2(self):
        a, b = self.a, self.b
        df = self.df
        corr2 = self.corr2

        probmvn_R = 0.6472497 #reported error approx. 7.7e-08
        probmvt_R = 0.5881863 #highest reported error up to approx. 1.99e-06
        assert_almost_equal(probmvt_R, mvstdtprob(a, b, corr2, df), 4)
        assert_almost_equal(probmvn_R, mvstdnormcdf(a, b, corr2, abseps=1e-5), 4)

    def test_mvn_mvt_3(self):
        a, b = self.a, self.b
        df = self.df
        corr2 = self.corr2

        #from -inf
        #print 'from -inf'
        a2 = a.copy()
        a2[:] = -np.inf
        probmvn_R = 0.9961141 #using higher precision in R, error approx. 6.866163e-07
        probmvt_R = 0.9522146 #using higher precision in R, error approx. 1.6e-07
        assert_almost_equal(probmvt_R, mvstdtprob(a2, b, corr2, df), 4)
        assert_almost_equal(probmvn_R, mvstdnormcdf(a2, b, corr2, maxpts=100000,
                                                    abseps=1e-5), 4)

    def test_mvn_mvt_4(self):
        a, bl = self.a, self.b
        df = self.df
        corr2 = self.corr2

        #from 0 to inf
        #print '0 inf'
        a2 = a.copy()
        a2[:] = -np.inf
        probmvn_R = 0.1666667 #error approx. 6.1e-08
        probmvt_R = 0.1666667 #error approx. 8.2e-08
        assert_almost_equal(probmvt_R, mvstdtprob(np.zeros(3), -a2, corr2, df), 4)
        assert_almost_equal(probmvn_R,
                            mvstdnormcdf(np.zeros(3), -a2, corr2,
                                         maxpts=100000, abseps=1e-5), 4)

    def test_mvn_mvt_5(self):
        a, bl = self.a, self.b
        df = self.df
        corr2 = self.corr2

        #unequal integration bounds
        #print "ue"
        a3 = np.array([0.5, -0.5, 0.5])
        probmvn_R = 0.06910487 #using higher precision in R, error approx. 3.5e-08
        probmvt_R = 0.05797867 #using higher precision in R, error approx. 5.8e-08
        assert_almost_equal(mvstdtprob(a3, a3+1, corr2, df), probmvt_R, 4)
        assert_almost_equal(probmvn_R, mvstdnormcdf(a3, a3+1, corr2,
                                                maxpts=100000, abseps=1e-5), 4)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['__main__','-vvs','-x'],#,'--pdb', '--pdb-failure'],
                   exit=False)

    print ('Done')
