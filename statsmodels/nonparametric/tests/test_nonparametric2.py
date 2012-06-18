import numpy as np
import numpy.testing as npt

import statsmodels.nonparametric as nparam


class TestUKDE(object):
    def setUp(self):
        N = 60
        np.random.seed(123456)
        self.o = np.random.binomial(2, 0.7, size=(N, 1))
        self.o2 = np.random.binomial(3, 0.7, size=(N, 1))
        self.c1 = np.random.normal(size=(N, 1))
        self.c2 = np.random.normal(10, 1, size=(N, 1))
        self.c3 = np.random.normal(10, 2, size=(N, 1))

    def test_pdf_mixeddata_CV_LS(self):
        dens_u = nparam.UKDE(tdat=[self.c1, self.o, self.o2], var_type='coo',
                             bw='cv_ls')
        npt.assert_allclose(dens_u.bw, [0.709195, 0.087333, 0.092500],
                            atol=1e-6)

        # Matches R to 3 decimals; results seem more stable than with R.
        # Can be checked with following code:
        ## import rpy2.robjects as robjects
        ## from rpy2.robjects.packages import importr
        ## NP = importr('np')
        ## r = robjects.r
        ## D = {"S1": robjects.FloatVector(c1), "S2":robjects.FloatVector(c2),
        ##      "S3":robjects.FloatVector(c3), "S4":robjects.FactorVector(o),
        ##      "S5":robjects.FactorVector(o2)}
        ## df = robjects.DataFrame(D)
        ## formula = r('~S1+ordered(S4)+ordered(S5)')
        ## r_bw = NP.npudensbw(formula, data=df, bwmethod='cv.ls')

    def test_pdf_mixeddata_LS_vs_ML(self):
        dens_ls = nparam.UKDE(tdat=[self.c1, self.o, self.o2], var_type='coo',
                             bw='cv_ls')
        dens_ml = nparam.UKDE(tdat=[self.c1, self.o, self.o2], var_type='coo',
                             bw='cv_ml')
        npt.assert_allclose(dens_ls.bw, dens_ml.bw, atol=0, rtol=0.5)

    def test_pdf_mixeddata_CV_ML(self):
        # Test ML cross-validation
        pass

    def test_pdf_continuous(self):
        # Test for only continuous data
        pass

    def test_pdf_ordered(self):
        # Test for only ordered data
        pass

    def test_pdf_unordered(self):
        # Test for only unordered data
        pass

