"""
Tests for discrete choice models : clogit

"""
from statsmodels.discrete.dcm_clogit import CLogit, CLogitResults
from statsmodels.discrete.tests.results.results_dcm_clogit import Travelmodechoice

import numpy as np
import pandas as pd
from collections import OrderedDict
from numpy.testing import assert_almost_equal

DECIMAL_4 = 4

RTOL_4 = 1e-4
RTOL_8 = 1e-8

ATOL_0 = 0
ATOL_1 = 10


class CheckDCMResults(object):
    """
    res2 are the results. res1 are the values from statsmodels

    """
    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_4)

    def test_bse(self):
        assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL_4)

    def test_llf(self):
        assert_almost_equal(self.res1.llf, self.res2.llf, DECIMAL_4)

    def test_llnull(self):
        assert_almost_equal(self.sum1.llnull, self.res2.llnull, DECIMAL_4)

    def test_aic(self):
        assert_almost_equal(self.sum1.aic, self.res2.aic, DECIMAL_4)

    def test_hessian(self):
        np.testing.assert_allclose(self.mod1.hessian(self.res1.params),
                                   self.res2.hessian, rtol=RTOL_4, atol=ATOL_0)

    def test_llrt(self):
        assert_almost_equal(self.sum1.llrt, self.res2.llrt, DECIMAL_4)

    def test_score_at_optimun(self):
        # this test the score evaluated at the optimum (where the score
        # should be zero). Anything smaller (in absolute value) than 1e-5
        # or 1e-8 is fine.

        np.testing.assert_allclose(self.mod1.score(self.res1.params),
                                   self.res2.score, rtol=RTOL_8, atol=ATOL_1)

    def test_score(self):
        # this test the score at parameters different from the optimum.

        import statsmodels.tools.numdiff as nd
        score_by_numdiff = nd.approx_fprime(self.res1.params * 2, \
                                            self.mod1.loglike, centered=True)

        np.testing.assert_allclose(self.mod1.score(self.res1.params * 2),
                                   score_by_numdiff, rtol=RTOL_4, atol=ATOL_1)

    def test_predict(self):
        np.testing.assert_allclose(self.mod1.predict(self.res1.params,
                                                     linear=False),
                                   self.res2.predict, rtol=RTOL_4, atol=ATOL_0)


class TestCLogit(CheckDCMResults):
    """
    Tests the Clogit model
    """

    @classmethod
    def setupClass(cls):
        # set up model

        # Loading data as pandas object
        data = sm.datasets.modechoice.load_pandas()
        data.endog[:5]
        data.exog[:5]
        data.exog['Intercept'] = 1  # include an intercept
        y, X = data.endog, data.exog

        # Names of the variables for the utility function for each alternative
        V = OrderedDict((
            ('air',   ['gc', 'ttme', 'Intercept', 'hinc']),
            ('train', ['gc', 'ttme', 'Intercept']),
            ('bus',   ['gc', 'ttme', 'Intercept']),
            ('car',   ['gc', 'ttme']))
            )
        # Number of common coefficients
        ncommon = 2

        # Describe and fit model
        mod1 = CLogit(y, X, V, ncommon, ref_level = 'car',
                      name_intercept = 'Intercept')
        res1 = mod1.fit()
        sum1 = CLogitResults(mod1)

        cls.mod1 = mod1
        cls.res1 = res1
        cls.sum1 = sum1

        # set up results
        res2 = Travelmodechoice()
        res2.clogit_greene()
        cls.res2 = res2



if __name__ == "__main__":
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x'], exit=False)
