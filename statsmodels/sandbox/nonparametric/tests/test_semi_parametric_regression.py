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

from statsmodels.sandbox.nonparametric.kernel_extras import SemiLinear
from statsmodels.sandbox.nonparametric.kernel_extras import SingleIndexModel


class CheckNonParametricRegressionResults(object):

	def test_fitted_values(self):
		assert_almost_equal(self.model.fitted, self.res.expected, 4)



class TestSemiLinear(CheckNonParametricRegressionResults):

    @classmethod
    def setupClass(cls):
        data = sm.datasets.wage1.load_pandas()
        data.endog - data.data['lwage'].values
        data.exog_linear = data.data[['female','married','educ','tenure']].values
        data.exog_np = data.data['exper'].values
        self.model = SemiLinear(data.endog, data.exog_linear, data.exog_np,
        						 'c', 4)

class TestSingleIndexModel(CheckNonParametricRegressionResults):

    @classmethod
    def setupClass(cls):
        data = sm.datasets.birthwt.load_pandas()
        data.exog = data.data[['smoke','race','ht','ui','ftv','age','lwt']].values
 		self.model = SingleIndexModel(endog=data.endog, exog=data.exog, 
                          var_type='uuuuuuc')
   


