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
        '''
        from exampledata import lbw
        from nipy.fixes.scipy.stats.models.regression import xi
        stata_lbw_beta=(.4612239, -.0271003, -.0151508, 1.262647,
                        .8620792, .9233448, .5418366, 1.832518,
                        .7585135)
        X = lbw()
        X = xi(X, col='race', drop=True)
        des = np.vstack((np.ones((len(X))), X['age'], X['lwt'],
                    X['black'], X['other'], X['smoke'], X['ptl'],
                    X['ht'], X['ui'])).T
            # add axes so we can hstack?
        model = glm(design=des, family=SSM.family.Binomial())
# is there currently no choice for a link function in GLM?
# choices in family.py but then one is chosen for you?
        results = model.fit(X['low'])
        # returning of all the OLS results is feeling like overload.
        # Maybe we should have to call summary for each
        # but GLM shouldn't inherit these from OLS
        # so...GLM should overload them
        assert_almost_equal(results.beta, stata_lbw_beta, 4)
#*********************************
# hascons is causing a lot of problems, since glm has
# multiple calls to initialize
# perhaps the best thing to do is to make adding a constant
# a utility function for right now...and leave it out of initialize
# altogether






