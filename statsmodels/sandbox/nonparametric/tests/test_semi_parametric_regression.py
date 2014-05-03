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
from nose.tools import nottest
import statsmodels.api as sm
from statsmodels.sandbox.nonparametric.tests.results.semi_parametric_results import Wage1, BirthWt
from numpy.testing import (assert_, assert_raises, assert_almost_equal,
                           assert_equal, assert_array_equal, assert_allclose)

from statsmodels.sandbox.nonparametric.kernel_extras import SemiLinear
from statsmodels.sandbox.nonparametric.kernel_extras import SingleIndexModel

class CheckNonParametricRegressionKnown(object):

    def test_parametric_parameters_locallinear(self):
        assert_allclose(self.model.params, self.expected_params, rtol=0.1)

    def test_nonparametric_fit_locallinear(self):
        assert_allclose(self.model.fittedy, self.expected_fittedy, rtol=0.1)

class CheckNonParametricRegressionResults(object):

    def test_mean_values(self):
        assert_allclose(self.model.fit()[0], self.res.mean, atol=1e-2)

    def test_mfx_values(self):
        assert_allclose(self.model.fit()[1][:,0], self.res.mfx, atol=1e-2)

    def test_bw_values(self):
        assert_allclose(self.model.bw, self.res.bw, atol=1e-2)

    def test_b_values(self):
        assert_allclose(self.model.b, self.res.b, atol=1e-2)

    def test_rsquared_values(self):
        assert_allclose(self.model.r_squared(), self.res.r_squared, atol=1e-2)

# Note that this has local linear hard coded currently - although docmentation suggets
# that is is local cosntant.
class T_estSemiLinearContinuousRegressionKnown(CheckNonParametricRegressionKnown):

    @classmethod
    def setupClass(cls):     
        seed = 430973
        np.random.seed(seed)
        np_vars = 3
        p_vars = 2
        nobs, ntest = 300, 50
        x_np = np.random.uniform(-2, 2, size=(nobs, np_vars))
        x_p = np.random.uniform(-2, 2, size=(nobs, p_vars))
        fparams = np.array([1,-2, 5])
        xb = x_np.sum(1) / 3
        fx = np.dot(np.column_stack((xb**2,xb,np.ones(len(xb)))),fparams.T)
        y = fx + x_p.sum(1)   
        cls.model = SemiLinear(y, x_p, x_np, 'ccc', p_vars)
        
        # Set known parameters
        cls.model.params = cls.model.b
        cls.expected_params = np.array([1,1])

        # Generate new poitns for testing
        x_np_test = np.random.uniform(-2, 2, size=(ntest, np_vars))
        x_p_test = np.random.uniform(-2, 2, size=(ntest, p_vars))
        xb = x_np_test.sum(1) / 3
        fx = np.dot(np.column_stack((xb**2,xb,np.ones(len(xb)))),fparams.T)
        cls.expected_fittedy = fx + x_p_test.sum(1)   
        cls.model.fittedy = cls.model.fit(x_p_test,x_np_test)


# Note that this has local linear hard coded currently - although docmentation suggets
# that is is local cosntant.
class TestSingleIndexContinousRegressionKnown(CheckNonParametricRegressionKnown):

    @classmethod
    def setupClass(cls):     
        seed = 430973
        np.random.seed(seed)
        np_vars = 2
        nobs, ntest = 1000, 10
        beta = np.array([1.0,2.0])
        x_np = np.random.uniform(-2, 2, size=(nobs, np_vars))
        fparams = np.array([1,-2, 1])
        xb = np.dot(x_np,beta)
        y = np.dot(np.column_stack((xb**2,xb,np.ones(len(xb)))),fparams.T) 
        cls.model = SingleIndexModel(y, x_np, var_type='cc')

        # Set known parameters
        cls.model.params = cls.model.b/np.linalg.norm(cls.model.b)
        cls.expected_params = beta/np.linalg.norm(beta)

        # Generate new poitns for testing
        x_np_test = np.random.uniform(-2, 2, size=(ntest, np_vars))
        xb = np.dot(x_np_test,beta)
        cls.expected_fittedy = np.dot(np.column_stack((xb**2,xb,np.ones(len(xb)))),fparams.T) 
        cls.model.fittedy = cls.model.fit(x_np_test)[0]

class TestSemiLinearRegressionResults(CheckNonParametricRegressionResults):

    @classmethod
    def setupClass(cls):
        # Need to set seed because fitting routine uses a random starting value.
        seed = 430973
        np.random.seed(seed)
        data = sm.datasets.wage1.load_pandas()
        data.endog = data.data['lwage'].values
        data.exog_linear = data.data[['female','married','educ','tenure']].values
        data.exog_np = data.data['exper'].values
        cls.model = SemiLinear(data.endog, data.exog_linear, data.exog_np,
                                 'c', 4)
        cls.res = Wage1()
        cls.res.semilinear()

class TestSingleIndexModelRegressionResults(CheckNonParametricRegressionResults):

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

        


