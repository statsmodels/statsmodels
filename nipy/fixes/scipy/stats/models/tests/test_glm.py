"""
Test functions for models.GLM
"""

import numpy as np
import numpy.random as R
from numpy.testing import *

import nipy.fixes.scipy.stats.models as SSM
import nipy.fixes.scipy.stats.models.glm as glm

W = R.standard_normal

class TestRegression(TestCase):

    def test_Logistic(self):
        X = W((40,10))
        Y = np.greater(W((40,)), 0)
        family = SSM.family.Binomial()
        cmodel = glm(design=X, family=SSM.family.Binomial())
        results = cmodel.fit(Y)
        self.assertEquals(results.df_resid, 30)

    def test_Logisticdegenerate(self):
        X = W((40,10))
        X[:,0] = X[:,1] + X[:,2]
        Y = np.greater(W((40,)), 0)
        family = SSM.family.Binomial()
        cmodel = glm(design=X, family=SSM.family.Binomial())
        results = cmodel.fit(Y)
        self.assertEquals(results.df_resid, 31)

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
        from nipy.fixes.scipy.stats.models.regression import xi
        stata_lbw_beta = (.4612239, -.0271003, -.0151508, 1.262647,
                        .8620792, .9233448, .5418366, 1.832518,
                        .7585135)
        stata_lbw_bse = (1.20459, .0364504, .0069259, .5264101, .4391532,
                        .4008266, .346249, .6916292, .4593768)
        stata_ll = -100.7239956
        stata_pearson = 182.0233425
        stata_deviance = 201.4479911
        stata_1_div_df_pearson = 1.011241
        stata_1_div_df_deviance = 1.119156
        stata_AIC = 1.1611 # based on llf, so only available for
                           # Newton-Raphson optimization...
        stata_BIC = -742.0655 # based on deviance
        stata_conf_int = ((-1.899729,2.822176), (-.0985418,.0443412),
                        (-.0287253,-.0015763), (.2309024,2.294392),
                        (.0013548,1.722804), (.137739,1.708951),
                        (-.136799,1.220472), (.4769494,3.188086),
                        (-.1418484,1.658875))

        X = lbw()
        X = xi(X, col='race', drop=True)
        des = np.vstack((X['age'], X['lwt'],
                    X['black'], X['other'], X['smoke'], X['ptl'],
                    X['ht'], X['ui'])).T
            # add axes so we can hstack?
        model = glm(design=des, hascons=False,
                family=SSM.family.Binomial())
# is there currently no choice for a link function in GLM?
# choices in family.py but then one is chosen for you?
        results = model.fit(X['low'])
        # returning of all the OLS results is feeling like overload.
        # Maybe we should have to call summary for each
        # but GLM shouldn't inherit these from OLS
        # so...GLM should overload them
        assert_almost_equal(results.beta, stata_lbw_beta, 4)


# confidence intervals pull from the wrong distribution for this example
# Standard Errors are the Observed Information (OIM) standard errors
# gives Z score for these...
# STATA gives log-likelihood of each iterations, keep a log to check
# in case of no convergence?





