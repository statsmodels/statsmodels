# -*- coding: utf-8 -*-
"""Tests for gam.AdditiveModel and GAM with Polynomials compared to OLS and GLM


Created on Sat Nov 05 14:16:07 2011

Author: Josef Perktold
License: BSD


Notes
-----

TODO: TestGAMGamma: has test failure (GLM looks good),
        adding log-link didn't help
        resolved: gamma doesn't fail anymore after tightening the
                  convergence criterium (rtol=1e-6)
TODO: TestGAMNegativeBinomial: rvs generation doesn't work,
        nbinom needs 2 parameters
TODO: TestGAMGaussianLogLink: test failure,
        but maybe precision issue, not completely off

        but something is wrong, either the testcase or with the link
        >>> tt3.__class__
        <class '__main__.TestGAMGaussianLogLink'>
        >>> tt3.res2.mu_pred.mean()
        3.5616368292650766
        >>> tt3.res1.mu_pred.mean()
        3.6144278964707679
        >>> tt3.mu_true.mean()
        34.821904835958122
        >>>
        >>> tt3.y_true.mean()
        2.685225067611543
        >>> tt3.res1.y_pred.mean()
        0.52991541684645616
        >>> tt3.res2.y_pred.mean()
        0.44626406889363229



one possible change
~~~~~~~~~~~~~~~~~~~
add average, integral based tests, instead of or additional to sup
    * for example mean squared error for mu and eta (predict, fittedvalues)
      or mean absolute error, what's the scale for this? required precision?
    * this will also work for real non-parametric tests

example: Gamma looks good in average bias and average RMSE (RMISE)

>>> tt3 = _estGAMGamma()
>>> np.mean((tt3.res2.mu_pred - tt3.mu_true))/tt3.mu_true.mean()
-0.0051829977497423706
>>> np.mean((tt3.res2.y_pred - tt3.y_true))/tt3.y_true.mean()
0.00015255264651864049
>>> np.mean((tt3.res1.y_pred - tt3.y_true))/tt3.y_true.mean()
0.00015255538823786711
>>> np.mean((tt3.res1.mu_pred - tt3.mu_true))/tt3.mu_true.mean()
-0.0051937668989744494
>>> np.sqrt(np.mean((tt3.res1.mu_pred - tt3.mu_true)**2))/tt3.mu_true.mean()
0.022946118520401692
>>> np.sqrt(np.mean((tt3.res2.mu_pred - tt3.mu_true)**2))/tt3.mu_true.mean()
0.022953913332599746
>>> maxabs = lambda x: np.max(np.abs(x))
>>> maxabs((tt3.res1.mu_pred - tt3.mu_true))/tt3.mu_true.mean()
0.079540546242707733
>>> maxabs((tt3.res2.mu_pred - tt3.mu_true))/tt3.mu_true.mean()
0.079578857986784574
>>> maxabs((tt3.res2.y_pred - tt3.y_true))/tt3.y_true.mean()
0.016282852522951426
>>> maxabs((tt3.res1.y_pred - tt3.y_true))/tt3.y_true.mean()
0.016288391235613865



"""

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from scipy import stats

from statsmodels.sandbox.gam import AdditiveModel
from statsmodels.sandbox.gam import Model as GAM #?
from statsmodels.genmod.families import family, links
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.regression.linear_model import OLS


class Dummy(object):
    pass

class CheckAM(object):

    def test_predict(self):
        assert_almost_equal(self.res1.y_pred,
                            self.res2.y_pred, decimal=2)
        assert_almost_equal(self.res1.y_predshort,
                            self.res2.y_pred[:10], decimal=2)


    def _est_fitted(self):
        #check definition of fitted in GLM: eta or mu
        assert_almost_equal(self.res1.y_pred,
                            self.res2.fittedvalues, decimal=2)
        assert_almost_equal(self.res1.y_predshort,
                            self.res2.fittedvalues[:10], decimal=2)

    def test_params(self):
        #note: only testing slope coefficients
        #constant is far off in example 4 versus 2
        assert_almost_equal(self.res1.params[1:],
                            self.res2.params[1:], decimal=2)
        #constant
        assert_almost_equal(self.res1.params[1],
                            self.res2.params[1], decimal=2)

    def _est_df(self):
        #not used yet, copied from PolySmoother tests
        assert_equal(self.res_ps.df_model(), self.res2.df_model)
        assert_equal(self.res_ps.df_fit(), self.res2.df_model) #alias
        assert_equal(self.res_ps.df_resid(), self.res2.df_resid)

class CheckGAM(CheckAM):

    def test_mu(self):
        #problem with scale for precision
        assert_almost_equal(self.res1.mu_pred,
                            self.res2.mu_pred, decimal=0)
#        assert_almost_equal(self.res1.y_predshort,
#                            self.res2.y_pred[:10], decimal=2)


class BaseAM(object):

    def __init__(self):

        #DGP: simple polynomial
        order = 3
        nobs = 200
        lb, ub = -3.5, 3
        x1 = np.linspace(lb, ub, nobs)
        x2 = np.sin(2*x1)
        x = np.column_stack((x1/x1.max()*1, 1.*x2))
        exog = (x[:,:,None]**np.arange(order+1)[None, None, :]).reshape(nobs, -1)
        idx = list(range((order+1)*2))
        del idx[order+1]
        exog_reduced = exog[:,idx]  #remove duplicate constant
        y_true = exog.sum(1) #/ 4.
        #z = y_true #alias check
        #d = x

        self.nobs = nobs
        self.y_true, self.x, self.exog = y_true, x, exog_reduced



class TestAdditiveModel(BaseAM, CheckAM):

    def __init__(self):
        super(self.__class__, self).__init__() #initialize DGP

        nobs = self.nobs
        y_true, x, exog = self.y_true, self.x, self.exog

        np.random.seed(8765993)
        sigma_noise = 0.1
        y = y_true + sigma_noise * np.random.randn(nobs)

        m = AdditiveModel(x)
        m.fit(y)
        res_gam = m.results #TODO: currently attached to class

        res_ols = OLS(y, exog).fit()

        #Note: there still are some naming inconsistencies
        self.res1 = res1 = Dummy() #for gam model
        #res2 = Dummy() #for benchmark
        self.res2 = res2 = res_ols  #reuse existing ols results, will add additional

        res1.y_pred = res_gam.predict(x)
        res2.y_pred = res_ols.model.predict(res_ols.params, exog)
        res1.y_predshort = res_gam.predict(x[:10])

        slopes = [i for ss in m.smoothers for i in ss.params[1:]]

        const = res_gam.alpha + sum([ss.params[1] for ss in m.smoothers])
        #print const, slopes
        res1.params = np.array([const] + slopes)


class BaseGAM(BaseAM, CheckGAM):

    def init(self):
        nobs = self.nobs
        y_true, x, exog = self.y_true, self.x, self.exog
        if not hasattr(self, 'scale'):
            scale = 1
        else:
            scale = self.scale

        f = self.family

        self.mu_true = mu_true = f.link.inverse(y_true)

        np.random.seed(8765993)
        #y_obs = np.asarray([stats.poisson.rvs(p) for p in mu], float)
        if issubclass(self.rvs.__self__.__class__, stats.rv_discrete):
            # Discrete distributions don't take `scale`.
            y_obs = self.rvs(mu_true, size=nobs)
        else:
            y_obs = self.rvs(mu_true, scale=scale, size=nobs)
        m = GAM(y_obs, x, family=f)  #TODO: y_obs is twice __init__ and fit
        m.fit(y_obs, maxiter=100)
        res_gam = m.results
        self.res_gam = res_gam   #attached for debugging
        self.mod_gam = m   #attached for debugging

        res_glm = GLM(y_obs, exog, family=f).fit()

        #Note: there still are some naming inconsistencies
        self.res1 = res1 = Dummy() #for gam model
        #res2 = Dummy() #for benchmark
        self.res2 = res2 = res_glm  #reuse existing glm results, will add additional

        #eta in GLM terminology
        res2.y_pred = res_glm.model.predict(res_glm.params, exog, linear=True)
        res1.y_pred = res_gam.predict(x)
        res1.y_predshort = res_gam.predict(x[:10]) #, linear=True)

        #mu
        res2.mu_pred = res_glm.model.predict(res_glm.params, exog, linear=False)
        res1.mu_pred = res_gam.mu

        #parameters
        slopes = [i for ss in m.smoothers for i in ss.params[1:]]
        const = res_gam.alpha + sum([ss.params[1] for ss in m.smoothers])
        res1.params = np.array([const] + slopes)


class TestGAMPoisson(BaseGAM):

    def __init__(self):
        super(self.__class__, self).__init__() #initialize DGP

        self.family =  family.Poisson()
        self.rvs = stats.poisson.rvs

        self.init()

class TestGAMBinomial(BaseGAM):

    def __init__(self):
        super(self.__class__, self).__init__() #initialize DGP

        self.family =  family.Binomial()
        self.rvs = stats.bernoulli.rvs

        self.init()

class _estGAMGaussianLogLink(BaseGAM):
    #test failure, but maybe precision issue, not far off
    #>>> np.mean(np.abs(tt.res2.mu_pred - tt.mu_true))
    #0.80409736263199649
    #>>> np.mean(np.abs(tt.res2.mu_pred - tt.mu_true))/tt.mu_true.mean()
    #0.023258245077813208
    #>>> np.mean((tt.res2.mu_pred - tt.mu_true)**2)/tt.mu_true.mean()
    #0.022989403735692578

    def __init__(self):
        super(self.__class__, self).__init__() #initialize DGP

        self.family =  family.Gaussian(links.log)
        self.rvs = stats.norm.rvs
        self.scale = 5

        self.init()


class TestGAMGamma(BaseGAM):

    def __init__(self):
        super(self.__class__, self).__init__() #initialize DGP

        self.family =  family.Gamma(links.log)
        self.rvs = stats.gamma.rvs

        self.init()

class _estGAMNegativeBinomial(BaseGAM):
    #rvs generation doesn't work, nbinom needs 2 parameters

    def __init__(self):
        super(self.__class__, self).__init__() #initialize DGP

        self.family =  family.NegativeBinomial()
        self.rvs = stats.nbinom.rvs

        self.init()

if __name__ == '__main__':
    t1 = TestAdditiveModel()
    t1.test_predict()
    t1.test_params()

    for tt in [TestGAMPoisson, TestGAMBinomial, TestGAMGamma,
               _estGAMGaussianLogLink]: #, TestGAMNegativeBinomial]:
        tt = tt()
        tt.test_predict()
        tt.test_params()
        tt.test_mu
