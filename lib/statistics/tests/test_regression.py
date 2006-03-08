import neuroimaging.statistics as S
import unittest
import numpy.random as R
import numpy as N

W = R.standard_normal

class RegressionTest(unittest.TestCase):

    def testOLS(self):
        X = W((40,10))
        Y = W((40,))
        model = S.regression.OLSModel(design=X)
        results = model.fit(Y)
        self.assertEquals(results.df_resid(), 30)

    def testAR(self):
        X = W((40,10))
        Y = W((40,))
        model = S.regression.ARModel(design=X, rho=0.4)
        results = model.fit(Y)
        self.assertEquals(results.df_resid(), 30)

    def testOLSdegenerate(self):
        X = W((40,10))
        X[:,0] = X[:,1] + X[:,2]
        Y = W((40,))
        model = S.regression.OLSModel(design=X)
        results = model.fit(Y)
        self.assertEquals(results.df_resid(), 31)

    def testARdegenerate(self):
        X = W((40,10))
        X[:,0] = X[:,1] + X[:,2]
        Y = W((40,))
        model = S.regression.ARModel(design=X, rho=0.9)
        results = model.fit(Y)
        self.assertEquals(results.df_resid(), 31)



if __name__ == '__main__':
    unittest.main()
