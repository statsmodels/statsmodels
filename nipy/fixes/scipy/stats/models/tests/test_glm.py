"""
Test functions for models.GLM
"""

import numpy as np
import numpy.random as R
from numpy.testing import *

import models
from models.glm import GLMtwo as GLM
from models.functions import add_constant, xi
from scipy import stats

W = R.standard_normal

#TODO: This all neds to be refactored.  Use setup and teardown methods
# to load data and test each one differently?
# decide on the best way

tests = {'lbw'=models.family.Binomial(), 'cpunish'=models.family.Poisson(), 'scotland'=models.family.Gamma()}
# finish design on this
# dictionary mapping dataset to family, results, residuals
# move all results to an external file and have them ordered the same way
# or in a dictionary

#class TestRegression(TestCase):
class TestRegression(object):
# this should allow the decorators to work
# see http://thread.gmane.org/gmane.comp.python.scientific.devel/9118

    def test_Logistic(self):
        X = W((40,10))
        Y = np.greater(W((40,)), 0)
        cmodel = GLM(Y,X, family=models.family.Binomial())
        results = cmodel.fit()
#        self.assertEquals(results.df_resid, 30)
        assert_equal(results.df_resid, 30)

    def test_Logisticdegenerate(self):
        X = W((40,10))
        X[:,0] = X[:,1] + X[:,2]
        Y = np.greater(W((40,)), 0)
        cmodel = GLM(Y, X, family=models.family.Binomial())
        results = cmodel.fit()
#        self.assertEquals(results.df_resid, 31)
        assert_equal(results.df_resid, 31)

    def test_bernouilli(self):
        '''
        These tests use the stata lbw data found here:

        http://www.stata-press.com/data/r9/rmain.html

        The tests results were obtained with R
        '''
        from exampledata import lbw
        from glm_test_resids import lbw_resids

        R_params = (-.02710031, -.01515082, 1.26264728,
                        .86207916, .92334482, .54183656, 1.83251780,
                        .75851348, .46122388)
#        stata_lbw_bse = (.0364504, .0069259, .5264101, .4391532,
#                        .4008266, .346249, .6916292, .4593768, 1.20459)
# NOTE: Stata's standard errors are different for MLE (based OIM)
# Their irls standard errors are the same (based on EIM) and no
# likelihood estimate is calculated
# this means that covariance matrix is different for Newton method
# ? this may also be where the discrepancy for LLF comes from
# if the LLF is calculated similar to
# http://www.mathworks.com/access/helpdesk/help/toolbox/ident/index.html?/access/helpdesk/help/toolbox/ident/ref/aic.html

        R_bse = (0.036449917, 0.006925765, 0.526405169, 0.439146744,
            0.400820976, 0.346246857, 0.691623875, 0.459373871, 1.204574885)
        R_AIC = 219.447991133
        R_deviance = 201.447991133
        R_df_null = 188
        R_null_deviance = 234.671996193219
        R_df_resid = 180
        R_dispersion = 1
        R_logLik = -100.7239955662511

        X = lbw()
        X = xi(X, col='race', drop=True)
        des = np.vstack((X['age'], X['lwt'],
                    X['black'], X['other'], X['smoke'], X['ptl'],
                    X['ht'], X['ui'])).T
        des = add_constant(des)
        model = GLM(X.low, des,
                family=models.family.Binomial())
        results = model.fit()
        resids = np.column_stack((results.resid_pearson, results.resid_dev,
                    results.resid_working, results.resid_anscombe,
                    results.resid_response))
        assert_almost_equal(resids, lbw_resids, 5)
        assert_almost_equal(results.params, R_params, 4)
        assert_almost_equal(results.bse, R_bse, 4)
        assert_equal(results.df_model,R_df_null-R_df_resid)
        assert_equal(results.df_resid,R_df_resid)
        assert_equal(results.scale,R_dispersion)
        aic=results.information_criteria()['aic']
        assert_almost_equal(aic, R_AIC, 5)
        assert_almost_equal(results.deviance, R_deviance, 5)
        assert_almost_equal(results.null_deviance, R_null_deviance)
        assert_almost_equal(results.llf, R_logLik, 5)

# STATA
        Stata_PearsonX2 = 182.0233425
        Stata_AIC = 1.1611  # scaled by nobs
        Stata_BIC = -742.0665
        Stata_estat_AIC = 219.448   # the same as R
        Stata_estat_BIC = 248.6237  # no idea

        assert_almost_equal(results.pearsonX2, Stata_PearsonX2, 5)
        bic = results.information_criteria()['bic']
        assert_almost_equal(bic, Stata_BIC, 4)
        assert_almost_equal(aic/results.nobs, Stata_AIC, 4)

        try:
            from rpy import r
            descols = ['x.%d' % (i+1) for i in range(des.shape[1])]
            formula = r('y ~ %s-1' % '+'.join(descols))
            frame = r.data_frame(y=X.low, x=des)
            rglm_res = r.glm(formula, data=frame, family='binomial')
        except ImportError:
            yield nose.tools.assert_true, True

    ### Poission Family ###
    def test_poisson(self):
        '''
        The following are from the R script in models.datasets.cpunish
        Slightly different than published results, but should be correct
        Probably due to rounding in cleaning?
        '''

        from models.datasets.cpunish.data import load
        from glm_test_resids import cpunish_resids

        R_params = (2.611017e-04, 7.781801e-02, -9.493111e-02, 2.969349e-01,
                2.301183e+00, -1.872207e+01, -6.801480e+00)
        R_bse = (5.187132e-05, 7.940193e-02, 2.291926e-02, 4.375164e-01,
                4.283826e-01, 4.283961e+00, 4.146850e+00)
        R_null_dev = 136.57281747225
        R_df_null = 16
        R_deviance = 18.59164
        R_df_resid = 10
        R_AIC = 77.85466
        dispersion = 1

        data = load()
        data.exog[:,3] = np.log(data.exog[:,3])
        data.exog = add_constant(data.exog)
        results = GLM(data.endog, data.exog,
                    family=models.family.Poisson()).fit()
        resids = np.column_stack((results.resid_pearson, results.resid_dev,
                    results.resid_working, results.resid_anscombe,
                    results.resid_response))

        assert_almost_equal(resids, cpunish_resids, 5)
        assert_equal(results.df_model, R_df_null-R_df_resid)
        assert_equal(results.df_resid, R_df_resid)
        assert_almost_equal(results.null_deviance, R_null_dev, 5)
        assert_equal(results.scale, dispersion)
        assert_almost_equal(results.params, R_params, 5)
        assert_almost_equal(results.bse, R_bse, 4)
        assert_almost_equal(results.deviance, R_deviance, 5)
        aic=results.information_criteria()['aic']
        assert_almost_equal(aic, R_AIC, 5)

# Stata

        Stata_AIC = 4.579686 # scaled by nobs
        Stata_BIC = -9.740492
        Stata_PearsonX2 = 24.75374835
        Stata_llf = -31.92732831
        Stata_estat_AIC = 77.85466 # same as R
        Stata_estat_BIC = 83.68715

        assert_almost_equal(results.llf, Stata_llf, 5)
        assert_almost_equal(results.pearsonX2, Stata_PearsonX2, 5)
        bic = results.information_criteria()['bic']
        assert_almost_equal(bic, Stata_BIC, 5)
        assert_almost_equal(aic/results.nobs, Stata_AIC, 5)

    def test_gamma(self):
        '''
        The following are from the R script in models.datasets.scotland
        '''

        from models.datasets.scotland.data import load
        from glm_test_resids import scotland_resids

        R_params = (4.961768e-05, 2.034423e-03, -7.181429e-05, 1.118520e-04,
                -1.467515e-07, -5.186831e-04, -2.42717498e-06, -1.776527e-02)
        R_bse = (1.621577e-05, 5.320802e-04, 2.711664e-05, 4.057691e-05,
            1.236569e-07, 2.402534e-04, 7.460253e-07, 1.147922e-02)
        R_null_dev = 0.536072
        R_df_null = 31
        R_deviance = 0.087388516417
        R_df_resid = 24
        R_AIC = 182.95
        R_dispersion = 0.003584283

        data = load()
        data.exog = add_constant(data.exog)
        results = GLM(data.endog, data.exog, family=models.family.Gamma()).fit()
        resids = np.column_stack((results.resid_pearson, results.resid_dev,
                    results.resid_working, results.resid_anscombe,
                    results.resid_response))
        assert_almost_equal(resids, scotvote_resids, 3)
        assert_almost_equal(results.params, R_params, 5)
        assert_almost_equal(results.bse, R_bse, 5)
        assert_almost_equal(results.scale, R_dispersion, 5)
        assert_equal(results.df_resid, R_df_resid)
        assert_equal(results.df_model, R_df_null-R_df_resid)
        assert_almost_equal(results.null_deviance, R_null_dev, 5)
        assert_almost_equal(results.deviance, R_deviance, 5)
        aic=results.information_criteria()['aic']

        @dec.knownfailureif(True)
        def test_aic():
            assert_almost_equal(aic, R_AIC, 4)  # Waiting for R ML help
        test_aic()

# why does this skip all the tests in here if this is added?

# Stata
        Stata_AIC = 10.72212
        Stata_BIC = -83.09027
        Stata_estat_AIC = 343.1079
        Stata_estat_BIC = 354.8338
        Stata_PearsonX2 = .0860228056
        Stata_llf = -163.5539382 # this is given by the formula with phi=1
        R_llf = -82.47352   # this is ever so close the answer given
                            # by the formula with phi=dispersion

        @dec.knownfailureif(True)
        def test_llf():
            assert_almost_equal(results.llf, Stata_llf, 5)
        test_llf()

        assert_almost_equal(results.PearsonX2, Stata_PearsonX2, 5)
        bic = results.information_criteria()['bic']
        assert_almost_equal(bic, Stata_BIC, 5)

        print 'Some insight into LLF failure in comments'

    def test_binomial(self):
        '''
        Test the Binomial distribution with binomial data from repeated trials
        data.endog is (# of successes, # of failures)
        '''
        R_params =  (-0.0168150366,  0.0099254766, -0.0187242148,
            -0.0142385609, 0.2544871730,  0.2406936644,  0.0804086739,
            -1.9521605027, -0.3340864748, -0.1690221685,  0.0049167021,
            -0.0035799644, -0.0140765648, -0.0040049918, -0.0039063958,
            0.0917143006,  0.0489898381,  0.0080407389,  0.0002220095,
            -0.0022492486, 2.9588779262)
        R_bse = (4.339467e-04, 6.013714e-04, 7.435499e-04, 4.338655e-04,
            2.994576e-02, 5.713824e-02, 1.392359e-02, 3.168109e-01,
            6.126411e-02, 3.270139e-02, 1.253877e-03, 2.254633e-04,
            1.904573e-03, 4.739838e-04, 9.623650e-04, 1.450923e-02,
            7.451666e-03, 1.499497e-03, 2.988794e-05, 3.489838e-04,
            1.546712e+00)
        R_null_dev = 34345.3688931
        R_df_null = 302
        R_deviance = 4078.76541772
        R_df_resid = 282
        R_AIC = 6039.22511799
        R_dispersion = 1.0

        from models.datasets.star98.data import load
        from glm_test_resids import star98_resids

        data = load()
        data.exog = add_constant(data.exog)
        trials = data.endog[:,:2].sum(axis=1)
        results = GLM(data.endog, data.exog, family=models.family.Binomial()).\
                    fit(data_weights = trials)
        resids = np.column_stack((results.resid_pearson, results.resid_dev,
                    results.resid_working, results.resid_anscombe,
                    results.resid_response))

        assert_almost_equal(resids, star98_resids, 2) # need to see which
                                                    # resids are so
                                                    #   imprecise
        assert_almost_equal(results.params, R_params, 4)
        assert_almost_equal(results.bse, R_bse, 4)
#        assert_almost_equal(results.resid_dev,R_dev_resid)
        assert_almost_equal(results.deviance, R_deviance, 5)
        aic=results.information_criteria()['aic']
        assert_almost_equal(aic, R_AIC)
        assert_almost_equal(results.null_deviance, R_null_dev, 5)
        assert_equal(results.df_model, R_df_null-R_df_resid)
        assert_equal(results.df_resid, R_df_resid)
        assert_almost_equal(results.scale, R_dispersion)

# Stata

        Stata_AIC = 19.93144
        Stata_BIC = 2467.494
        Stata_PearsonX2 = 4051.921614
        Stata_estat_AIC = 6039.226  # same as R
        Stata_estat_BIC = 6117.214
        Stata_llf = -2998.612928    # same as R

        bic = results.information_criteria()['bic']
        assert_almost_equal(results.pearsonX2, Stata_PearsonX2, 5)
        assert_almost_equal(bic, Stata_BIC, 5)
        assert_almost_equal(results.llf, Stata_llf, 5)

# why do these fail when the exact same thing doesn't
# fail at the interpreter
    @dec.knownfailureif(True)
    def test_gaussian(self):
        from models.datasets.longley.data import load
        data = load()
        data.exog = add_constant(data.exog)
        ols_res = models.regression.OLS(data.endog,data.exog).fit()
        glm_res = GLM(data.endog,data.exog,
                        family=models.family.Gaussian()).fit()
        assert_array_equal(glm_res.params,ols_res.params)
        assert_equal(glm_res.llf,ols_res.llf)
        ols_aic = ols_res.information_criteria()['aic']
        glm_aic = glm_res.information_criteria()['aic']
        assert_equal(glm_aic,glm_aic)
        assert_equal(glm_res.bse,ols_res.bse)
        assert_equal(glm_res.scale, ols_res.scale)
        assert_equal(glm_res.resid_dev, ols_res.resid)
        assert_equal(glm_res.resid_pearson, ols_res.resid)
        assert_equal(glm_res.resid_working, ols_res.resid)
        assert_equal(glm_res.resid_response, ols_res.resid)
        assert_equal(glm_res.resid_anscombe, ols_res.resid)

    @dec.slow   # useful to know but could make it smaller subset of the data
    def test_inverse_guassian(self):
        '''
        Tests the Inverse Gaussian family in GLM.

        Notes
        -----
        Used the rndivgx.ado file provided by Hardin and Hilbe to
        generate the data.
        '''
        from exampledata import inv_gauss
        Y,X = inv_gauss()
        X = add_constant(X)
# This is SLOW...but so is Stata...R isn't particularly slow
        results = GLM(Y, X, family=models.family.InverseGaussian()).fit()

#        np.random.seed(54321)
#        x1 = np.abs(stats.norm.ppf((np.random.random(5000))))
#        x2 = np.abs(stats.norm.ppf((np.random.random(5000))))
#        X = np.column_stack((x1,x2))
#        X = add_constant(X)
#        params = np.array([.5, -.25, 1])
#        eta = np.dot(X, params)
#        mu = 1/np.sqrt(eta)
#        sigma = .5
#       This isn't correct.  Errors need to be normally distributed
#       But Y needs to be Inverse Gaussian, so we could build it up
#       by throwing out data?
#       Refs: Lai (2009) Generating inverse Gaussian random variates by
#        approximation
# Atkinson (1982) The simulation of generalized inverse gaussian and
#        hyperbolic random variables seems to be the canonical ref
#        Y = np.dot(X,params) + np.random.wald(mu, sigma, 1000)
#        model = GLM(Y, X, family=models.family.InverseGaussian(link=\
#            models.family.links.identity))

        R_params = (0.4519770, -0.2508288, 1.0359574)
        R_bse = (0.03148291, 0.02237211, 0.03429943)
        R_null_dev = 1520.673165475461
        R_df_null = 4999
        R_deviance = 1423.943980407997
        R_df_resid = 4997
        R_AIC = 5059.41911646446
        R_dispersion = 0.2867266359127567
        assert_almost_equal(results.params, R_params,5)
        assert_almost_equal(results.bse, R_bse,5)
        assert_almost_equal(results.null_deviance, R_null_dev,5)
        assert_almost_equal(results.df_resid+results.df_model,R_df_null,5)
        assert_almost_equal(results.deviance, R_deviance,5)
        assert_almost_equal(results.df_resid, R_df_resid,5)
        aic = results.information_criteria()['aic']
        @dec.knownfailureif(True)   # why doesn't this work?
        def test_aic(self):
            assert_almost_equal(aic, R_AIC, 5)
        test_aic()  # is this bad form?
        assert_almost_equal(results.scale, R_dispersion, 5)

# Stata

        Stata_AIC = 1.55228
        Stata_BIC = -41136.47
        Stata_PearsonX2 = 1432.771536
        Stata_estat_AIC = 7761.401
        Stata_estat_BIC = 7780.952
        Stata_llf = -3877.700354
        R_llf = -2525.70955823223

        print 'Figure out LLF for Inverse Gaussian'

    def test_negativebinomial():
        from models.datasets.committee.data import load
        data = load()
        data.exog[:,2] = np.log(data.exog[:,2])
        interaction = data.exog[:,2]*data.exog[:,1]
        data.exog = np.column_stack((data.exog,interaction))
        data.exog = add_constant(data.exog)

# Stata
        Stata_AIC = None
        Stata_BIC = None
        Stata_PearsonX2 = None
        Stata_llf = None
        Stata_params = None

if __name__=="__main__":
    run_module_suite()







