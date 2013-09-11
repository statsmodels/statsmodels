"""
Tests for discrete choice models : clogit

"""

from discrete.dcm_clogit import CLogit, CLogitResults
from discrete.results_dcm_clogit import Travelmodechoice

import numpy as np
import pandas as pd
from collections import OrderedDict
from numpy.testing import assert_almost_equal

DECIMAL_4 = 4
RTOL_4 = 1e-4
ATOL_0 = 0


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

#    def test_score(self):
#        np.testing.assert_allclose(self.mod1.score(self.res1.params),
#                                   self.res2.score, rtol=RTOL_4, atol=ATOL_0)

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

        # load data
        from patsy import dmatrices

        url = "http://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/\
                ModeChoice.csv"
        file_ = "ModeChoice.csv"
        import os
        if not os.path.exists(file_):
            import urllib
            urllib.urlretrieve(url, file_)
        df = pd.read_csv(file_)
        df.describe()

        f = 'mode  ~ ttme+invc+invt+gc+hinc+psize'
        y, X = dmatrices(f, df, return_type='dataframe')

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

        # set up precision
        cls.decimal_tvalues = 3


if __name__ == "__main__":
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x'], exit=False)
