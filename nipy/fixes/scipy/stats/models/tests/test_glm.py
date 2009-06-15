"""
Test functions for models.GLM
"""

import numpy as np
import numpy.random as R
from numpy.testing import *

import nipy.fixes.scipy.stats.models as SSM
import nipy.fixes.scipy.stats.models.glm as models

W = R.standard_normal

class TestRegression(TestCase):

    def test_Logistic(self):
        X = W((40,10))
        Y = np.greater(W((40,)), 0)
        family = SSM.family.Binomial()
        cmodel = models(design=X, family=SSM.family.Binomial())
        results = cmodel.fit(Y)
        self.assertEquals(results.df_resid, 30)

    def test_Logisticdegenerate(self):
        X = W((40,10))
        X[:,0] = X[:,1] + X[:,2]
        Y = np.greater(W((40,)), 0)
        family = SSM.family.Binomial()
        cmodel = models(design=X, family=SSM.family.Binomial())
        results = cmodel.fit(Y)
        self.assertEquals(results.df_resid, 31)

    def test_lbw(self):
        '''
        These tests use the stata lbw data found here:

        http://www.stata-press.com/data/r9/rmain.html

        See STATA manual or http://www.stata.com/help.cgi?glm
        '''
        from exampledata import lbw()
        X = lbw()





