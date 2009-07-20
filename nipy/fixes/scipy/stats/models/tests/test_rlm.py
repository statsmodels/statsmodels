"""
Test functions for models.rlm
"""

import numpy.random as R
from numpy.testing import *

import nipy.fixes.scipy.stats.models.rlm as models

W = R.standard_normal

class check_rlm_results(self):
    pass

class test_rlm(check_rlm_results):
    pass


class TestRegression(TestCase):

    def test_Robust(self):
        X = W((40,10))
        Y = W((40,))
        model = models(Y, X)
        results = model.fit()
        self.assertEquals(results.df_resid, 30)

    def test_Robustdegenerate(self):
        X = W((40,10))
        X[:,0] = X[:,1] + X[:,2]
        Y = W((40,))
        model = models(Y, X)
        results = model.fit()
        self.assertEquals(results.df_resid, 31)



