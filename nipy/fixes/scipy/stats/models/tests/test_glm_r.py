''' Test results of OLS and GLM against R
note: based on example_compare_r.py
'''


import numpy as np
import numpy.testing as nptest
import scipy
import nose.tools
from rpy import r
from rmodelwrap import RModel

import models as SSM
from models.functions import xi, add_constant
#from nipy.fixes.scipy.stats.models.glm import Model as GlmModel
#from nipy.fixes.scipy.stats.models.glm import glzm as GlmModel
from models.glm import GLMtwo as GlmModel

DECIMAL = 4

def assert_model_similar_base(res1, res2):
    ''' Test if models have similar parameters
    JP: where are the beta standard errors or beta covariance - now included
    res1 is from R'''
    nptest.assert_almost_equal(res1.beta, res2.params, 4)
    nptest.assert_almost_equal(res1.resid, res2.resid, 4)
    nptest.assert_almost_equal(res1.predict, res2.predict, 4)
    nptest.assert_almost_equal(res1.df_resid, res2.df_resid, 4)

def assert_model_similar_glm(res1, res2):
    ''' Test if models have similar parameters
    res1 is from R until differences are removed
    this doesn't work'''
    yield nptest.assert_almost_equal, res1.beta, res2.results.params, 4
    yield nptest.assert_almost_equal, res1.resid, res2.results.dev_resid, 4
    yield nptest.assert_almost_equal, res1.predict, res2.results.predict, 4
    yield nptest.assert_almost_equal, res1.df_resid, res2.results.df_resid, 4
    yield nptest.assert_array_almost_equal, res1.bse, res2.results.bse, 4
#    yield nptest.assert_array_almost_equal, res1.bcov, res2.results.bcov, 4
    yield nptest.assert_array_almost_equal, res1.scale, res2.results.scale, 4
    yield nptest.assert_array_almost_equal, 5, 4, 4


class Check_Glm_R(object):
    def test_beta(self):
        nptest.assert_array_almost_equal(self.res1.beta, self.res2.results.params, DECIMAL)
    def test_bse(self):
        nptest.assert_array_almost_equal(self.res1.bse, self.res2.results.bse, DECIMAL)
#    def test_bcov(self):
#        nptest.assert_array_almost_equal(self.res1.bcov, self.res2.results.cov_params(), DECIMAL)
#   need to sort out cov_params and normalized_cov_params
    def test_deviance(self):
        nptest.assert_array_almost_equal(self.res1.deviance, self.res2.results.deviance, DECIMAL)
    def test_fitted(self):
        nptest.assert_array_almost_equal(self.res1.predictedy, self.res2.results.mu, DECIMAL)
    def test_fitted(self):
        nptest.assert_array_almost_equal(self.res1.resid, self.res2.resid, DECIMAL)
    def test_fitted(self):
        nptest.assert_array_almost_equal(self.res1.predict, self.res2.results.predict, DECIMAL)
    def test_scale(self):
        nptest.assert_array_almost_equal(self.res1.scale, self.res2.results.scale, DECIMAL)
#    @nptest.decorators.knownfailureif(True)
    def test_aic(self):
        nptest.assert_almost_equal(self.res1.aic,
                self.res2.results.information_criteria()['aic'], DECIMAL)
    def test_df(self):
        nptest.assert_almost_equal(self.res1.df_resid, self.res2.df_resid, 4)
        nptest.assert_almost_equal(self.res1.df[2], self.res2.df_model+1, 4) #Note needs to be changed
    def _est_test(self):
        nptest.assert_array_almost_equal(5, 4, 4)


class test_glm_ols(Check_Glm_R):
    def __init__(self):
    #def test_ols():
        from exampledata import y, x
        rlm_res = RModel(y, x, r.lm)
        rsum = r.summary(rlm_res.results)

        ols_res = SSM.regression.OLS(y,x).fit()
        model_glmols = SSM.glm.GLMtwo(y,x)
        glmols_res = model_glmols.fit()

        famg = r.gaussian(link = "identity")
        rglmgau_res = RModel(y, x, r.glm, family=famg)
        #print rglmgau_res.bse

        glmols_res.beta = glmols_res.params
        glmols_res.resid = glmols_res.resid_dev
        assert_model_similar_base(rlm_res, ols_res)
        assert_model_similar_base(rlm_res, glmols_res)
        assert_model_similar_base(glmols_res, ols_res)
        #assert_model_similar_glm(rglmgau_res, model_glmols)
        self.res1 = rglmgau_res
        self.res2 = model_glmols

class test_glm_binomial(Check_Glm_R):
    def __init__(self):
    #def test_glm_binomial():
        #copy from glm_prints.py
        from exampledata import lbw
        from models.functions import xi
        #from nipy.fixes.scipy.stats.models.glm import Model as GlmModel
        #from nipy.fixes.scipy.stats.models.glm import glzm as GlmModel
        from models.glm import GLMtwo as GlmModel
        X=lbw()
        X=xi(X, col='race', drop=True)
        des = np.column_stack((X['age'],X['lwt'],X['black'],X['other'],X['smoke'], X['ptl'], X['ht'], X['ui']))
        des = add_constant(des)
        model_glmbin = GlmModel(X.low, des, family=SSM.family.Binomial())
        glmbin_res = model_glmbin.fit()

        rglmbin_res = RModel(X.low, des, r.glm, family='binomial')
##        print glmbin_res.beta
##        print rglmbin_res.beta
##        print glmbin_res.bse
##        print rglmbin_res.bse
##        assert_model_similar_glm(rglmbin_res, model_glmbin)
        self.res1 = rglmbin_res
        self.res2 = model_glmbin

class _est_glm_binomial_fraction(Check_Glm_R):
    #skip until later
    def __init__(self):
        from models.datasets.star98.data import load
        data = load()
        print 'data.exog .shape', data.exog.shape
        print 'data.endog .shape', data.endog.shape
        #JP: convert to bernoulli observations, converges but wrong results
        indicator = (data.endog[:,0]>data.endog[:,1]).astype(int)
        data.exog = add_constant(data.exog)
        trials = data.endog[:,:2].sum(1)
        fraction = data.endog[:,0]/trials

        rglmbinfrac_res = RModel(fraction, data_exog, r.glm, family='binomial')
        model_glmbinfrac = GlmModel(data.endog, data.exog, family = SSM.family.Binomial())
        glmbinfrac_res = model_glmbinfrac.fit()
        #assert_almost_equal(R_beta, results.params)
        self.res1 = rglmbinfrac_res
        self.res2 = model_glmbinfrac

class test_glm_gamma(Check_Glm_R):
    def __init__(self):
    #def test_glm_gamma():
        # A Gamma example, from McCullagh & Nelder (1989, pp. 300-2)
        #taken from R help
        ##clotting <- data.frame(
        ##    u = c(5,10,15,20,30,40,60,80,100),
        ##    lot1 = c(118,58,42,35,27,25,21,19,18),
        ##    lot2 = c(69,35,26,21,18,16,13,12,12))
        ##summary(glm(lot1 ~ log(u), data=clotting, family=Gamma))
        u = np.array([5,10,15,20,30,40,60,80,100])
        lot1 = np.array([118,58,42,35,27,25,21,19,18])
        lot2 = np.array([69,35,26,21,18,16,13,12,12])
        lots = np.column_stack((np.ones(9), lot1, lot2))
        rglmgam_res = RModel(u, lots, r.glm, family='Gamma')
        modelgam = GlmModel(u, lots, family=SSM.family.Gamma(SSM.family.links.inverse))
        glmgam_res = modelgam.fit()#[:,None])
##        print glmgam_res.beta
##        print rglmgam_res.beta
##        print glmgam_res.bse
##        print rglmgam_res.bse
        #assert_model_similar_glm(rglmgam_res, modelgam)
        self.res1 = rglmgam_res
        self.res2 = modelgam


class test_glm_gammadata(Check_Glm_R):
    def __init__(self):
        #fixing test_glm.test_gamma
        ##    def test_gamma(self):
        ##        '''
        ##        The following are from the R script in models.datasets.cpunish
        ##        '''

        from models.datasets.scotland.data import load

        R_beta = (4.961768e-05, 2.034423e-05, 2.034423e-05, -7.181429e-05,
            1.118520e-04, -1.467515e-07, -5.186831e-04, -1.776527e-02)
        R_bse = (1.621577e-05, 5.320802e-04, 2.711664e-05, 4.057691e-05,
            1.236569e-07, 2.402534e-04, 7.460253e-07, 1.147922e-02)
        R_null_dev = 0.536072
        R_df_model = 31
        R_resid_dev = 0.087389
        R_df_resid = 24
        R_AIC = 182.95
        R_dispersion = 0.003584283

        data = load()
        data.exog = add_constant(data.exog)
        modeltestgam = GlmModel(data.endog, data.exog, family = SSM.family.Gamma(SSM.family.links.inverse))
        glmtestgam_res = modeltestgam.fit()
        rglmtestgam_res = RModel(data.endog, data.exog, r.glm, family='Gamma')
        #assert_almost_equal(R_beta, results.beta)
    ##    print glmtestgam_res.beta
    ##    print rglmtestgam_res.beta
    ##    print glmtestgam_res.bse
    ##    print rglmtestgam_res.bse
        #assert_model_similar_glm(rglmtestgam_res, modeltestgam)
        self.res1 = rglmtestgam_res
        self.res2 = modeltestgam


#def test_glm_poisson():
class test_glm_poisson(Check_Glm_R):
    def __init__(self):

        #checking Poisson test
        ##    def test_poisson(self):
        ##        '''
        ##        The following are from the R script in models.datasets.cpunish
        ##        Slightly different than published results, but should be correct
        ##        Probably due to rounding in cleaning?
        ##        '''

        from models.datasets.cpunish.data import load

        R_beta = (0.0002611, 0.0778180, -0.0949311, 0.2969349, 2.3011833,
                -18.7220680, -6.8014799)
        R_bse = (5.1871e-05, 7.9402e-02, 2.2919e-02, 4.3752e-01, 4.2838e-01,
                4.2840e+00, 4.1468e+00)
        # REPORTS Z VALUE of these
        # Dispersion parameter = 1
        R_null_dev = 136.573
        R_df_model = 16
        R_resid_dev = 18.592
        R_df_resid = 10
        R_AIC = 77.85

        data = load()
        data.exog[:,3] = np.log(data.exog[:,3])
        data.exog = add_constant(data.exog)
        #modeltestpoiss = glzm(data_exog, hascons=True, family=SSM.family.Poisson())
        modeltestpoiss = GlmModel(data.endog, data.exog, family=SSM.family.Poisson())
        glmtestpoiss_res = modeltestpoiss.fit()
        rglmtestpoiss_res = RModel(data.endog, data.exog, r.glm, family='poisson')
        #assert_almost_equal(R_beta, results.beta)
        #assert_almost_equal(R_bse, results.bse)
        print glmtestpoiss_res.params
        print rglmtestpoiss_res.beta
        print glmtestpoiss_res.bse
        print rglmtestpoiss_res.bse
        #logl = np.sum(modeltestpoiss.Y *np.log(glmtestpoiss_res.mu) - glmtestpoiss_res.mu-np.log(scipy.factorial(modeltestpoiss.Y)))
        #aic = -2*logl + 2*(modeltestpoiss.df_model + 1)
        print glmtestpoiss_res.information_criteria()['aic']
        print rglmtestpoiss_res.rsum['aic']
        #assert_model_similar_glm(rglmtestpoiss_res, modeltestpoiss)
        self.res1 = rglmtestpoiss_res
        self.res2 = modeltestpoiss



class test_glm_gausslog(Check_Glm_R):
    def __init__(self):
        nobs = 100
        x = np.arange(nobs)[:,None]
        y = 1.0 + 2.0*x + x**2 + 0.1 * np.random.randn(nobs,1)
        X = np.c_[np.ones((nobs,1)),x,x**2]


        lny = np.exp(-(-1.0 + 0.02*x + 0.0001*x**2)) + 0.001 * np.random.randn(nobs,1)
        lny = lny.ravel()
        modelglmtestgaulog = GlmModel(lny, X, family=SSM.family.Gaussian(SSM.family.links.log))
        glmtestgaulog_res = modelglmtestgaulog.fit()
        famg = r.gaussian(link = "log")
        rglmtestgaulog_res = RModel(lny, X, r.glm, family=famg)
##        print glmtestgaulog_res.beta
##        print rglmtestgaulog_res.beta
##        print glmtestgaulog_res.bse
##        print rglmtestgaulog_res.bse
        self.res1 = rglmtestgaulog_res
        self.res2 = modelglmtestgaulog

class test_glm_gaussinv(Check_Glm_R):
    def __init__(self):
        nobs = 100
        x = np.arange(nobs)[:,None]
        y = 1.0 + 2.0*x + x**2 + 0.1 * np.random.randn(nobs,1)
        X = np.c_[np.ones((nobs,1)),x,x**2]

        #Note: power link is currently not allowed in Gaussian
        powy = np.power(1.0 + 0.001*x + 0.0001*x**2, -1) + 0.001 * np.random.randn(nobs,1)
        powy = powy.ravel()
        modelglmtestgauinv = GlmModel(powy, X, family=SSM.family.Gaussian(SSM.family.links.inverse))
        glmtestgauinv_res = modelglmtestgauinv.fit()
        famg = r.gaussian(link = "inverse")
        rglmtestgauinv_res = RModel(powy, X, r.glm, family=famg)
##        print glmtestgauinv_res.beta
##        print rglmtestgauinv_res.beta
##        print glmtestgauinv_res.bse
##        print rglmtestgauinv_res.bse
        self.res1 = rglmtestgauinv_res
        self.res2 = modelglmtestgauinv

if __name__=="__main__":
    nptest.run_module_suite()
