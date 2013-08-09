"""
Tests for discrete choice models : clogit

Results are from R mlogit package

"""
#cur_dir = os.path.abspath(os.path.dirname(__file__))

import sys
sys.path[0]='/home/nuska/github/Statsmodels/statsmodels'

import os
import numpy as np
import pandas as pandas
from patsy import dmatrices

# import all functions from numpy.testing that are needed
from numpy.testing import (assert_almost_equal)

from statsmodels.discrete.dcm_clogit import CLogit
from statsmodels.discrete.tests.results.results_dcm_clogit import Travelmodechoice

DECIMAL_4 = 4

class CheckDCMResults(object):
    """
    res2 are the results. res1 are the values from statsmodels
    """

    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_4)

    def test_hessian(self):
        np.testing.assert_allclose(self.mod1.hessian(self.res1.params),
                                   self.res2.hessian, rtol=1e-4, atol=0)

class TestCLogit(CheckDCMResults):
    """
    Tests the Clogit model using Newton's method for fitting.
    """

    @classmethod
    def setupClass(cls):
        # set up model
        # TODO: use datasets instead
        url = "http://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/ModeChoice.csv"
        file_ = "ModeChoice.csv"

        if not os.path.exists(file_):
            import urllib
            urllib.urlretrieve(url, "ModeChoice.csv")
        df = pandas.read_csv(file_)
        pandas.set_printoptions(max_rows=1000, max_columns=20)
        df.describe()

        nchoices = 4
        nobs = 210
        choice_index = np.arange(nchoices*nobs) % nchoices


        df['hinc_air'] = df['hinc']*(choice_index==0)
        f = 'mode  ~ ttme+invc+invt+gc+hinc+psize+hinc_air'
        y, X = dmatrices(f, df, return_type='dataframe')
        y.head()
        X.head()

        endog = y.to_records()
        endog = endog['mode'].reshape(-1, nchoices)

        dta = X.to_records()
        dta1 = np.array(dta)

        xivar = [['gc', 'ttme', 'Intercept','hinc_air'],
                 ['gc', 'ttme', 'Intercept'],
                 ['gc', 'ttme', 'Intercept'],
                 ['gc', 'ttme' ]]

        xi = []

        for ii in range(nchoices):
            xi.append(dta1[xivar[ii]][choice_index==ii])
        xifloat = [X.ix[choice_index == ii, xi_names].values
                    for ii, xi_names in enumerate(xivar)]

        mod1 = CLogit(endog, xifloat, 2)
        res1 = CLogit(endog, xifloat, 2).fit()

        cls.mod1 = mod1
        cls.res1 = res1

        # set up results
        res2 = Travelmodechoice()
        res2.clogit_greene()
        cls.res2 = res2

        # set up precision
        cls.decimal_tvalues = 3


if __name__ == "__main__":
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x'], exit=False)
