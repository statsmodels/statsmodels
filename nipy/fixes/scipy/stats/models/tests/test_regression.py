"""
Test functions for models.regression
"""

from numpy.random import standard_normal
from numpy.testing import *

from nipy.fixes.scipy.stats.models.regression import OLS, AR

W = standard_normal

class TestRegression(TestCase):

    def testOLS(self):
        X = W((40,10))
        Y = W((40,))
        model = OLS(Y, X)
        results = model.fit()
        self.assertEquals(results.df_resid, 30)

    def testAR(self):
        X = W((40,10))
        Y = W((40,))
        model = AR(design=X, rho=0.4)
        results = model.fit(Y)
        self.assertEquals(results.df_resid, 30)

    def testOLSdegenerate(self):
        X = W((40,10))
        X[:,0] = X[:,1] + X[:,2]
        Y = W((40,))
        model = OLS(Y,X)
        results = model.fit()
        self.assertEquals(results.df_resid, 31)

    def testARdegenerate(self):
        X = W((40,10))
        X[:,0] = X[:,1] + X[:,2]
        Y = W((40,))
        model = AR(design=X, rho=0.9)
        results = model.fit(Y)
        self.assertEquals(results.df_resid, 31)



