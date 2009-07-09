"""
Test functions for models.GLM
"""

import numpy as np
import numpy.random as R
from numpy.testing import *

import models
from models.glm import GLMtwo as GLM
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

        The tests results were obtained with R
        '''
        from exampledata import lbw
        R_params = (-.02710031, -.01515082, 1.26264728,
                        .86207916, .92334482, .54183656, 1.83251780,
                        .75851348, .46122388)
#        stata_lbw_bse = (.0364504, .0069259, .5264101, .4391532,
#                        .4008266, .346249, .6916292, .4593768, 1.20459)
        R_bse = (0.036449917, 0.006925765, 0.526405169, 0.439146744,
            0.400820976, 0.346246857, 0.691623875, 0.459373871, 1.204574885)
        R_resid_dev = (-0.8651253,-0.5232320,-0.8802559,-1.1510657,-1.1919416,
        -0.7500286,-0.5224024,-0.8987449,-0.7051695,-0.7806846,-0.9237283,
        -0.6450358,-1.6748694,-1.2588746,-0.9284774,-0.9284774,-1.1071944,
        -0.7644361,-1.0617464,-0.7280951,-0.6707875,-0.7464557,-0.2350923,
        -0.7086997,-1.0074762,-0.3390030,-0.8174635,-0.3790278,-0.9343678,
        -0.9936664,-0.9936664,-1.1404540,-1.3545388,-0.3850418,-0.8445326,
        -0.6282585,-0.7189247,-0.7174399,-0.3584186,-0.7381099,-0.8888506,
        -0.3245726,-0.8369003,-0.3911441,-1.3035317,-1.3035317,-0.4155653,
        -0.7283597,-0.5202702,-1.3347700,-1.1266600,-0.7131899,-0.7316492,
        -0.8373502,-0.8141982,-0.8698529,-1.5166623,-0.5511397,-0.8690038,
        -0.8119817,-0.8119817,-0.7570985,-0.7936768,-0.4130233,-1.4452542,
        -0.9819551,-0.7681837,-0.4371675,-1.0229172,-0.8890208,-1.2526359,
        -0.8379450,-1.1273107,-1.0103085,-0.7599179,-0.4512080,-0.4294064,
        -0.8014238,-1.2948703,-0.3055890,-0.4762033,-0.3145360,-0.7396891,
        -0.7440471,-0.7373433,-1.1637857,-0.8678458,-0.4735202,-0.2874572,
        -0.4970595,-0.4575171,-0.7017040,-0.8407121,-1.9116213,-0.7599179,
        -0.4227577,-0.3680950,-0.6767091,-0.6767091,-0.4112950,-0.5389859,
        -1.1412950,-0.9924978,-0.5458204,-0.7790978,-0.8714566,-0.4924397,
        -0.3614674,-0.8185763,-0.6987977,-0.2793659,-0.7974596,-0.6726544,
        -0.6500031,-0.5705221,-0.6446022,-0.5125319,-0.6624322,-0.4957642,
        -0.9545973,-0.4020905,-0.5517396,-0.5403788,-0.4830677,-0.4615187,
        -0.4590960,-0.3932058,-0.8091885,-0.5165954,-0.3745662, 0.8017404,
        1.8605156, 0.8887301,0.6552655, 1.1283597, 1.9244575, 1.1821359,
        1.3177365, 0.9715987, 1.1152029,1.6537851, 0.6841140, 1.6641798,
        1.6566830, 1.4794559, 1.8136166, 1.6733383,1.6532902, 1.5332827,
        1.3309810, 1.0204559, 1.8884508, 1.2067270, 1.4184715,2.1805419,
        0.7525877, 1.0581685, 1.1272681, 1.2716852, 0.7094879, 1.5046175,
        1.3527267, 1.5620618, 1.8032720, 0.7798185, 1.0490513, 1.2717369,
        1.5465618,1.3775399, 1.8940894, 1.5346157, 1.0704799, 1.0105668,
        1.2060057, 1.6738861,1.6378022, 1.7022925, 1.5729326, 1.3351545,
        1.4202979, 0.9129041, 1.5415584,2.1465826, 0.8334436, 1.5365427,
        1.4275696, 1.0940890, 0.8106274, 0.9059955)

        R_AIC = 219.447991133
        R_deviance = 201.447991133
        R_df_null = 188
        R_null_deviance = 234.67
        R_df_resid = 180
        R_dispersion = 1

        X = lbw()
        X = xi(X, col='race', drop=True)
        des = np.vstack((X['age'], X['lwt'],
                    X['black'], X['other'], X['smoke'], X['ptl'],
                    X['ht'], X['ui'])).T
        des = add_constant(des)
        model = GLM(X.low, des,
                family=models.family.Binomial())
        results = model.fit()
        assert_almost_equal(results.params, stata_lbw_beta, 4)
        assert_almost_equal(results.bse, R_lbw_bse, 4)
        assert_almost_equal(reslts.resid_dev, R_resid_dev)
        assert_equal(results.df_model,R_df_null-R_df_resid)
        assert_equal(results.df_resid,R_df_resid)
        assert_equal(results.scale,R_dispersion)
        aic=results.information_criteria()['aic']
        assert_almost_equal(aic, R_AIC, 5)
        assert_almost_equal(results.deviance, R_deviance, 5)
        assert_almost_equal(results.null_deviance,R_null_deviance)


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

    ### Poission Family ###
    def test_poisson(self):
        '''
        The following are from the R script in models.datasets.cpunish
        Slightly different than published results, but should be correct
        Probably due to rounding in cleaning?
        '''

        from models.datasets.cpunish.data import load

        R_params = (2.611017e-04, 7.781801e-02, -9.493111e-02, 2.969349e-01,
                2.301183e+00, -1.872207e+01, -6.801480e+00)
        R_bse = (5.187132e-05, 7.940193e-02, 2.291926e-02, 4.375164e-01,
                4.283826e-01, 4.283961e+00, 4.146850e+00)
# REPORTS Z VALUE of these
# Dispersion parameter = 1
        R_null_dev = 136.57281747225
        R_df_null = 16
        R_deviance = 18.59164
        R_df_resid = 10
        R_AIC = 77.85466
        dispersion = 1
        R_resid_dev =  (0.29637762, 0.27622019, 2.97778783, 0.16114971,
                0.59618385, 0.80661890, 0.05295480, -0.27663840, -0.68349824,
                -1.27435608, -1.40212820, -1.50517303, -1.09541344, 0.10649195,
                0.02081086,  0.56713011, -0.77000601)


        data = load()
        data.exog[:,3] = np.log(data.exog[:,3])
        data.exog = add_constant(data.exog)
        results = GLM(data.endog, data.exog, family=models.family.Poisson()).fit()
        assert_equal(results.df_model, R_df_null-R_df_resid)
        assert_equal(results.df_resid, R_df_resid)
        assert_almost_equal(results.null_deviance, R_null_dev, 5)
        assert_almost_equal(results.resid_dev, R_resid_dev, 5)
        assert_equal(results.scale, dispersion)
        assert_almost_equal(results.params, R_params, 5)
        assert_almost_equal(results.bse, R_bse, 4) # loss of precision here?
        assert_almost_equal(results.deviance, R_deviance, 5)
        aic=results.information_criteria()['aic']
        assert_almost_equal(aic, R_AIC, 5)


    def test_gamma(self):
        '''
        The following are from the R script in models.datasets.scotland
        '''

        from models.datasets.scotland.data import load

        R_params = (4.961768e-05, 2.034423e-03, -7.181429e-05, 1.118520e-04,
                -1.467515e-07, -5.186831e-04, -2.42717498e-06, -1.776527e-02)
        R_bse = (1.621577e-05, 5.320802e-04, 2.711664e-05, 4.057691e-05,
            1.236569e-07, 2.402534e-04, 7.460253e-07, 1.147922e-02)
        R_resid_dev =  (0.042568563, -0.018383239, 0.055082238, -0.022977589,
                -0.025268887, -0.149532908, -0.019868088, 0.066162991,
                0.020073265, -0.008439840, -0.044965040,  0.011540922,
                0.054396013,  0.087608095, 0.064554711, 0.002202139,
                0.094741416, -0.068453449, 0.051075224,  0.012929287,
                0.022411015, 0.014675644, 0.035977240, -0.067655338,
                -0.051342143, -0.025684194, -0.083922361, -0.033451556,
                0.003269332, -0.026545926, 0.011790996, -0.033632483 )
        R_null_dev = 0.536072
        R_df_null = 31
        R_deviance = 0.087388516417
        R_df_resid = 24
        R_AIC = 182.95
        R_dispersion = 0.003584283
        R_deviance = 0.087388516417

        data = load()
        data.exog = add_constant(data.exog)
        results = GLM(data.endog, data.exog, family = models.family.Gamma()).fit()
        assert_almost_equal(results.params, R_params, 5)
        assert_almost_equal(results.bse, R_bse, 5)
        assert_almost_equal(results.scale, R_dispersion, 5)
        assert_equal(results.df_resid, R_df_resid)
        assert_equal(results.df_model, R_df_null-R_df_resid)
        assert_almost_equal(results.null_deviance, R_null_dev, 5)
        assert_almost_equal(results.resid_dev, R_resid_dev, 5)
        assert_almost_equal(results.deviance, R_deviance, 5)
        aic=results.information_criteria()['aic']
#        assert_almost_equal(aic, R_AIC, 4)  # this is weird, check with stata
                                            # alternative definition for small
                                            # sample?

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
        R_dev_resid = ( -1.334256458, 0.992728455, 4.294470098, 0.206894088,
        -3.741874249,-8.707047805, 1.304068389,-2.317121054,-4.971653259,
        -4.393625160, 2.600229151, 1.065540479, -5.425537139,-7.660005497,
        -4.217550992, 0.723167032, 0.834363159, 0.416675868, -0.970035938,
        -0.782334441, 0.466473574,-3.155173763,-0.389217633,-1.050168160,
        -0.502673781,-2.489617253,-1.401620316,-0.762768510,-5.735793521,
        -1.816793284, -2.810127248,-3.134834134,12.808605079,-3.974417371,
        -4.502314206,-0.383046908, 5.681476004,-1.537744186, 2.220927636,
        -4.934040089,-1.438250198,-0.969250025, -0.615352207,-1.768229343,
        -2.098397377, 1.672247643,-2.410978450, 1.798111440, 2.770852718,
        -1.969879820, 2.820828679, 0.453236445, 2.574167435, 0.805680023,
        -1.801406348, 0.409013265,-8.191203360, 0.335615250,-4.447425747,
        -3.125469321, 2.538375360,-0.533944466,-4.982624429, 0.612699814,
        -3.401977913, 3.499001295, -1.848302578,-5.139410952,-2.857662177,
        -1.364043831,-4.450589638,-3.346225058, -1.926510722,-5.788875147,
        -3.318569660, 0.338694485, 0.003329199,-2.350610237, 1.252569576,
        0.242382902, 2.336331414, 2.959004700,-2.064560212,-3.810752574,
        -2.339958836, 3.223746254, 2.616084893, 2.299807511, 4.487808088,
        -2.760215934,-0.249823961,-1.842806619,-2.158056351,-3.229906926,
        4.403640304, 2.343688636, -3.033037640, 3.851561449, 0.301027921,
        -0.477748679,-1.287369919, 5.412982846, -3.474665072, 6.069468317,
        3.093292914, 4.429434909, 5.127840065, 0.046422843, -4.445502814,
        3.495503233, 0.740656931,-4.149989976, 5.284099651,-1.474377021,
        2.683785283,10.709754044, 3.559575322,-0.963120352, 2.215340494,
        -0.566184244, -4.250766589,-1.062074791,-2.788089243,-1.580506149,
        3.476558194,-2.983706656, 5.383305689, 0.837572109, 3.311905338,
        -0.131010894,-0.021684902, 5.542584452, 4.051065778, 1.008120157,
        -0.160678177,-2.168514007, 2.360294546, 0.883415885, 1.400441211,
        -2.659848538,-0.045988429,-4.916084144,-0.150371455,-0.708368682,
        -3.043926805,-1.416089035,-2.641143062, 1.629251287,-4.007115080,
        -2.420206273, -0.559020654,-1.553079713,-0.356280754,-6.007937698,
        4.431061552,-0.846502147, -0.030891758, 0.882950327, 2.354802454,
        4.443950190, 3.274531427, 1.671568704, 6.318256183, 3.609283460,
        5.266765056, 2.166439528,-0.528912983, 1.753311152, 9.645887633,
        -1.659680195, 0.159736391,-6.811618885, 3.307876031,-3.484225388,
        -6.058533837,-5.965386218,-3.671399062,-0.925475485,-5.335722562,
        3.920336040, -6.689663120,-6.313295073,-0.203637507, 2.997104027,
        -2.589584249, 2.070769483, -4.676225158,-6.128804762, 2.005050145,
        -4.722452053,-0.617245398,-1.726009560, 4.270882663, 0.138745512,
        0.061974893,-0.515304599,-5.061446083, 4.937638455, 0.415797640,
        1.566899631,-1.021203850, 2.707712207,-1.715103652,-1.194170359,
        -9.521708591,-2.494494450,-0.525266524, 2.390562527, 2.165117170,
        -6.029969073, 3.750211191,-1.891887880,-5.901026851, 0.363578800,
        -0.187548059,-3.324974744, -3.576628945, 1.699847472, 2.692024014,
        2.430085272, 8.313899876, 4.607147106, 0.977350556,-2.067638427,
        7.549328503, 1.095498007,15.419988557, 0.753824909, 1.546177139,
        0.051919759, 6.788717090,-1.317320725, 3.135709663,-2.257594491,
        -3.698017144,-9.079854895, 1.311996661, 5.251527704,-6.765137459,
        1.474918289, 0.886123439,10.227906799, 2.522305945,-0.335658914,
        1.201799033,-0.191671077, -1.151020171,-3.322033001, 1.628908376,
        -2.316396795,-2.989808845, 0.643925659, 1.873754688,-4.155081719,
        -2.006701091,-3.064178370, 3.060439474,-4.165236970, -2.767332179,
        -4.150768022, 2.110032932, 4.137209095,-0.278715160, 0.917757955,
        -6.416451198,-3.777582425,-3.149747862,-0.889500541, 2.817961474,
        -3.409322924, 0.323271465, 2.483646473,-2.109787869, 1.337577071,
        1.258690811, 4.623757670, -1.038796541, 0.649642673,-2.908309592,
        -0.700208309, 0.906367289,-0.237098171, 8.471029387, 4.797313016,
        -1.427900716,-6.238046152,-1.133251003, 2.659185792, -0.697884901,
        -1.402815510, 3.494765807,-5.093262434, 1.164156482, 7.463401093,
        -0.293802718, 2.130703879, 6.704178485,-0.187232103,-1.807567978,
        -4.095491562, -2.324381089,-7.183973130, 3.086117147)

        R_null_dev = 34345.3688931
        R_df_null = 302
        R_deviance = 4078.76541772
        R_df_resid = 282
        R_AIC = 6039.22511799
        R_dispersion = 1.0

        from models.datasets.star98.data import load
        data = load()
        data.exog = add_constant(data.exog)
        trials = data.endog[:,:2].sum(1)
        results = GLM(data.endog, data.exog, family=models.family.Binomial()).\
                    fit(data_weights = trials)
        assert_almost_equal(results.params, R_params, 4)
        assert_almost_equal(results.bse, R_bse, 4)
        assert_almost_equal(results.resid_dev,R_dev_resid)
        assert_almost_equal(results.deviance, R_deviance, 5)
        aic=results.information_criteria()['aic']
        assert_almost_equal(aic, R_AIC)
        assert_almost_equal(results.null_deviance, R_null_dev, 5)
        assert_equal(results.df_model, R_df_null-R_df_resid)
        assert_equal(results.df_resid, R_df_resid)
        assert_almost_equal(results.scale, R_dispersion)

        def test_gaussian(self):
            pass

        def test_inverse_gaussian(self):
            pass

if __name__=="__main__":
    run_module_suite()







