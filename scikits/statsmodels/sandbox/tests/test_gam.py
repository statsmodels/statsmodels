# -*- coding: utf-8 -*-
"""Tests for gam.AdditiveModel and GAM with Polynomials compared to OLS and GLM


Created on Sat Nov 05 14:16:07 2011

Author: Josef Perktold
License: BSD


Notes
-----

TODO: TestGAMGamma: has test failure (GLM looks good),
        adding log-link didn't help
TODO: TestGAMNegativeBinomial: rvs generation doesn't work,
        nbinom needs 2 parameters
TODO: TestGAMGaussianLogLink: test failure,
        but maybe precision issue, not completely off

missing: Tests with non-default link

"""

import numpy as np
from numpy.testing import assert_almost_equal

from scipy import stats

from scikits.statsmodels.sandbox.gam import AdditiveModel
from scikits.statsmodels.sandbox.gam import Model as GAM #?
from scikits.statsmodels.genmod.families import family, links
from scikits.statsmodels.genmod.generalized_linear_model import GLM
from scikits.statsmodels.regression.linear_model import OLS


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
        assert_equal(self.res_ps.def_model(), self.res2.df_model)
        assert_equal(self.res_ps.def_fit(), self.res2.df_model) #alias
        assert_equal(self.res_ps.def_resid(), self.res2.df_resid)

class CheckGAM(CheckAM):

    def test_mu(self):
        assert_almost_equal(self.res1.mu_pred,
                            self.res2.mu_pred, decimal=2)
#        assert_almost_equal(self.res1.y_predshort,
#                            self.res2.y_pred[:10], decimal=2)


class BaseAM(object):

    def __init__(self):

        #DGP: simple polynomial
        order = 3
        nobs = 1000
        lb, ub = -3.5, 3
        x1 = np.linspace(lb, ub, nobs)
        x2 = np.sin(2*x1)
        x = np.column_stack((x1/x1.max()*1, 1.*x2))
        exog = (x[:,:,None]**np.arange(order+1)[None, None, :]).reshape(nobs, -1)
        idx = range((order+1)*2)
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
        print const, slopes
        res1.params = np.array([const] + slopes)


class BaseGAM(BaseAM, CheckGAM):

    def init(self):
        nobs = self.nobs
        y_true, x, exog = self.y_true, self.x, self.exog

        f = self.family

        self.mu_true = mu_true = f.link.inverse(y_true)

        np.random.seed(8765993)
        #y_obs = np.asarray([stats.poisson.rvs(p) for p in mu], float)
        y_obs = self.rvs(mu_true) #this should work
        m = GAM(y_obs, x, family=f)  #TODO: y_obs is twice __init__ and fit
        m.fit(y_obs)
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
    #test failure, but maybe precision issue, not completely off

    def __init__(self):
        super(self.__class__, self).__init__() #initialize DGP

        self.family =  family.Gaussian(links.log)
        self.rvs = stats.norm.rvs

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

    for tt in [TestGAMPoisson, TestGAMBinomial]: #, TestGAMGaussianLogLink]: #, TestGAMNegativeBinomial]:#TestGAMGamma]:
        tt = tt()
        tt.test_predict()
        tt.test_params()
        tt.test_mu



