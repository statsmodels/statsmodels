"""
Test functions for models.GLM
"""

import numpy as np
import numpy.random as R
from numpy.testing import *

import models
from models.glm import Model as GLM
from models.functions import add_constant, xi

W = R.standard_normal

class TestRegression(TestCase):

    def test_Logistic(self):
        X = W((40,10))
        Y = np.greater(W((40,)), 0)
        cmodel = GLM(Y,X, family=models.family.Binomial())
        results = cmodel.fit()
        self.assertEquals(results.df_resid, 30)

    def test_Logisticdegenerate(self):
        X = W((40,10))
        X[:,0] = X[:,1] + X[:,2]
        Y = np.greater(W((40,)), 0)
        cmodel = GLM(Y, X, family=models.family.Binomial())
        results = cmodel.fit()
        self.assertEquals(results.df_resid, 31)

    def test_bernouilli(self):
        '''
        These tests use the stata lbw data found here:

        http://www.stata-press.com/data/r9/rmain.html

        See STATA manual or http://www.stata.com/help.cgi?glm

        Notes
        ------
        If the STATA examples are not public domain, perhaps
        Jeff Gill's are.

        They can be found here: http://jgill.wustl.edu/research/books.html

        His monograph is Generalized Linear Models: A Unified Approach,
        2000, SAGE QASS series.
        '''
        from exampledata import lbw
        stata_lbw_beta = (-.0271003, -.0151508, 1.262647,
                        .8620792, .9233448, .5418366, 1.832518,
                        .7585135, .4612239)
        stata_lbw_bse = (.0364504, .0069259, .5264101, .4391532,
                        .4008266, .346249, .6916292, .4593768, 1.20459)
# NOTE that these are the same standard errors obtained with R
# to the at least 1e-4
        stata_ll = -100.7239956
        stata_pearson = 182.0233425
        stata_deviance = 201.4479911
        stata_1_div_df_pearson = 1.011241
        stata_1_div_df_deviance = 1.119156
        stata_AIC = 1.1611 # based on llf, so only available for
                           # Newton-Raphson optimization...
        stata_BIC = -742.0655 # based on deviance
        stata_conf_int = ((-.0985418,.0443412),
                        (-.0287253,-.0015763), (.2309024,2.294392),
                        (.0013548,1.722804), (.137739,1.708951),
                        (-.136799,1.220472), (.4769494,3.188086),
                        (-.1418484,1.658875), (-1.899729,2.822176))

        X = lbw()
        X = xi(X, col='race', drop=True)
        des = np.vstack((X['age'], X['lwt'],
                    X['black'], X['other'], X['smoke'], X['ptl'],
                    X['ht'], X['ui'])).T
        des = add_constant(des)
        model = GLM(X.low, des,
                family=models.family.Binomial())
        results = model.fit()
        # returning of all the OLS results is feeling like overload.
        # Maybe we should have to call summary for each
        # but GLM shouldn't inherit these from OLS
        # so...GLM should overload them
        assert_almost_equal(results.theta, stata_lbw_beta, 4)
        bse = np.sqrt(np.diag(results.cov_theta()))
        assert_almost_equal(bse, stata_lbw_bse, 4)
# where is this loss of precision coming from?
# must be the scale, though beta is correct
# is it the default use of Pearson's X2 for scaling?
# play with algorithm

# confidence intervals pull from the wrong distribution for this example
# Standard Errors are the Observed Information (OIM) standard errors
# gives Z score for these...
# STATA gives log-likelihood of each iterations, keep a log to check
# in case of no convergence?

        try:
            from rpy import r
            descols = ['x.%d' % (i+1) for i in range(des1.shape[1])]
            formula = r('y ~ %s-1' % '+'.join(descols)) # -1 bc constant is appended
            frame = r.data_frame(y=X.low, x=des1)
            rglm_res = r.glm(formula, data=frame, family='binomial')
# everything looks good up to this point, but I can't figure out
# how to get a covariance matrix from the results in rpy.
        except ImportError:
            yield nose.tools.assert_true, True

    ### Poission Link ###
    def test_poisson(self):
        '''
        The following are from the R script in models.datasets.cpunish
        Slightly different than published results, but should be correct
        Probably due to rounding in cleaning?
        '''
        R_params = (0.0002611, 0.0778180, -0.0949311, 0.2969349, 2.3011833,
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
        from models.datasets.punish.data import load
        data = load()
        data.exog[3] = np.log(data.exog[3])
        data.exog = add_constant(data.exog[3])
        results = GLM(data.endog, data.exog).fit()
# Estimates are wrong...
# Confirm R findings with stata

    def test_gamma(self):
        '''
        The following are from the R script in models.datasets.cpunish
        '''
        R_params = (4.961768e-05, 2.034423e-05, 2.034423e-05, -7.181429e-05,
            1.118520e-04, -1.467515e-07, -5.186831e-04, -1.776527e-02)
# IT LOOKS LIKE THE DATA IN THE EXAMPLE HAS BEEN SCALED THOUGH THE NUMBERS
# ARE STILL CORRECT, THE ORDER OF MAGNITUDE ISN'T, SHOULDN'T BE A BIG DEAL
        R_bse = (1.621577e-05, 5.320802e-04, 2.711664e-05, 4.057691e-05,
            1.236569e-07, 2.402534e-04, 7.460253e-07, 1.147922e-02)
        R_null_dev = 0.536072
        R_df_model = 31
        R_resid_dev = 0.087389
        R_df_resid = 24
        R_AIC = 182.95
        R_dispersion = 0.003584283
        from models.datasets.scotland.data import load
        data = load()
        data.exog = add_constant(data.exog)
        results = GLM(data.endog, data.exog, family = models.family.Gamma()).fit()
# results aren't correct...
# tested this on an older NIPY and those results are WILDLY incorrect...hmm

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

        R_null_dev = 34345.4
        R_df_model = 302
        R_resid_dev = 4078.8
        R_df_resid = 282
        R_AIC = 6039.2
        # Binomial algorithm not working yet...

if __name__=="__main__":
    run_module_suite()







