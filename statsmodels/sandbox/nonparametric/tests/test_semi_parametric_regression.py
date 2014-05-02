"""
Tests for the semiparametic regression models. Currentling including
     - SemiLinear
     - SingleIndexModel
The tests are validated against those in [1].

References
----------
[1] R 'np' vignette : http://cran.r-project.org/web/packages/np/vignettes/np.pdf

"""

import numpy as np
import numpy.testing as npt
import statsmodels.api as sm
from statsmodels.sandbox.nonparametric.tests.results.semi_parametric_results import Wage1, BirthWt
from numpy.testing import (assert_, assert_raises, assert_almost_equal,
                           assert_equal, assert_array_equal, assert_allclose)

from statsmodels.sandbox.nonparametric.kernel_extras import SemiLinear
from statsmodels.sandbox.nonparametric.kernel_extras import SingleIndexModel


class CheckNonParametricRegressionResults(object):

    def test_mean_values(self):
        assert_almost_equal(self.model.fit()[0], self.res.mean, 4)

    def test_mfx_values(self):
        assert_almost_equal(self.model.fit()[1][:,0], self.res.mfx, 4)

    def test_bw_values(self):
        assert_almost_equal(self.model.bw, self.res.bw, 4)

    def test_b_values(self):
        assert_almost_equal(self.model.b, self.res.b, 4)

    def test_rsquared_values(self):
        assert_almost_equal(self.model.r_squared(), self.res.r_squared, 4)



class TestSemiLinear(CheckNonParametricRegressionResults):

    @classmethod
    def setupClass(cls):
        data = sm.datasets.wage1.load_pandas()
        data.endog = data.data['lwage'].values
        data.exog_linear = data.data[['female','married','educ','tenure']].values
        data.exog_np = data.data['exper'].values
        cls.model = SemiLinear(data.endog, data.exog_linear, data.exog_np,
                                 'c', 4)
        cls.res = Wage1()
        cls.res.semilinear()

class TestSingleIndexModel(CheckNonParametricRegressionResults):

    @classmethod
    def setupClass(cls):
        # Need to set seed because fitting routine uses a random starting value.
        seed = 430973
        np.random.seed(seed)
        data = sm.datasets.birthwt.load_pandas()
        data.exog = data.data[['smoke','race','ht','ui','ftv','age','lwt']].values
        cls.model = SingleIndexModel(endog=data.endog, exog=data.exog,
                                      var_type='uuuuuuc')
        cls.res = BirthWt()
        cls.res.singleindexmodel()
        


