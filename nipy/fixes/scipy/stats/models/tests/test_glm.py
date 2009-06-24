"""
Test functions for models.GLM
"""

import numpy as np
import numpy.random as R
from numpy.testing import *

import nipy.fixes.scipy.stats.models as SSM
import nipy.fixes.scipy.stats.models.glm as glm
from nipy.fixes.scipy.stats.models.functions import add_constant, xi

W = R.standard_normal

class TestRegression(TestCase):

#    def test_Logistic(self):
#        X = W((40,10))
#        Y = np.greater(W((40,)), 0)
#        family = SSM.family.Binomial()
#        cmodel = glm(design=X, family=SSM.family.Binomial())
#        results = cmodel.fit(Y)
#        self.assertEquals(results.df_resid, 30)

#    def test_Logisticdegenerate(self):
#        X = W((40,10))
#        X[:,0] = X[:,1] + X[:,2]
#        Y = np.greater(W((40,)), 0)
#        family = SSM.family.Binomial()
#        cmodel = glm(design=X, family=SSM.family.Binomial())
       # results = cmodel.fit(Y)
#        self.assertEquals(results.df_resid, 31)

    def test_lbw(self):
        '''
        These tests use the stata lbw data found here:

        http://www.stata-press.com/data/r9/rmain.html

        See STATA manual or http://www.stata.com/help.cgi?glm

        Notes
        ------
        If the STATA examples are not public domain, perhaps
        Jeff Gill's are.

        They can be found here: http://jgill.wustl.edu/research/books.html

        His monograph is Generalized Linear Models: A Unified Approach,
        2000, SAGE QASS series.
        '''
        from exampledata import lbw
        stata_lbw_beta = (-.0271003, -.0151508, 1.262647,
                        .8620792, .9233448, .5418366, 1.832518,
                        .7585135, .4612239)
        stata_lbw_bse = (.0364504, .0069259, .5264101, .4391532,
                        .4008266, .346249, .6916292, .4593768, 1.20459)
# NOTE that these are the same standard errors obtained with R
# to the at least 1e-4
        stata_ll = -100.7239956
        stata_pearson = 182.0233425
        stata_deviance = 201.4479911
        stata_1_div_df_pearson = 1.011241
        stata_1_div_df_deviance = 1.119156
        stata_AIC = 1.1611 # based on llf, so only available for
                           # Newton-Raphson optimization...
        stata_BIC = -742.0655 # based on deviance
        stata_conf_int = ((-.0985418,.0443412),
                        (-.0287253,-.0015763), (.2309024,2.294392),
                        (.0013548,1.722804), (.137739,1.708951),
                        (-.136799,1.220472), (.4769494,3.188086),
                        (-.1418484,1.658875), (-1.899729,2.822176))

        X = lbw()
        X = xi(X, col='race', drop=True)
        des = np.vstack((X['age'], X['lwt'],
                    X['black'], X['other'], X['smoke'], X['ptl'],
                    X['ht'], X['ui'])).T
        des = add_constant(des)
        model = glm(X.low, des,
                family=SSM.family.Binomial())
        results = model.fit(X['low'])
        # returning of all the OLS results is feeling like overload.
        # Maybe we should have to call summary for each
        # but GLM shouldn't inherit these from OLS
        # so...GLM should overload them
        assert_almost_equal(results.theta, stata_lbw_beta, 4)
        bse = np.sqrt(np.diag(results.cov_theta()))
        assert_almost_equal(bse, stata_lbw_bse, 4)
# where is this loss of precision coming from?
# must be the scale, though beta is correct
# is it the default use of Pearson's X2 for scaling?
# play with algorithm

# confidence intervals pull from the wrong distribution for this example
# Standard Errors are the Observed Information (OIM) standard errors
# gives Z score for these...
# STATA gives log-likelihood of each iterations, keep a log to check
# in case of no convergence?

        try:
            from rpy import r
            descols = ['x.%d' % (i+1) for i in range(des1.shape[1])]
            formula = r('y ~ %s-1' % '+'.join(descols)) # -1 bc constant is appended
            frame = r.data_frame(y=X.low, x=des1)
            rglm_res = r.glm(formula, data=frame, family='binomial')
# everything looks good up to this point, but I can't figure out
# how to get a covariance matrix from the results in rpy.
        except ImportError:
            yield nose.tools.assert_true, True


if __name__=="__main__":
    run_module_suite()







