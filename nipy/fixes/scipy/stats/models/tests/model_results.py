import numpy as np
import os
import models
import glm_test_resids
#TODO: Streamline this with RModelwrap

def generated_data():
    '''
    Returns `Y` and `X` from test_data.bin

    Returns
    -------
    Y : array
        Endogenous Data
    X : array
        Exogenous Data
    '''
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            "test_data.bin")
    data = np.fromfile(filename, "<f8")
    data.shape = (126,15)
    y = data[:,0]
    x = data[:,1:]
    return y,x

### GLM MODELS ###

class lbw(object):
    '''
    The LBW data can be found here

    http://www.stata-press.com/data/r9/rmain.html

    X is the entire data as a record array.
    '''
    def __init__(self):
        # data set up for data not in datasets
        filename="stata_lbw_glm.csv"
        data=np.recfromcsv(filename, converters={4: lambda s: s.strip("\"")})
        data = models.functions.xi(data, col='race', drop=True)
        self.endog = data.low
        design = np.column_stack((data['age'], data['lwt'],
                    data['black'], data['other'], data['smoke'], data['ptl'],
                    data['ht'], data['ui']))
        self.exog = models.functions.add_constant(design)
        # Results for Canonical Logit Link
        self.params = (-.02710031, -.01515082, 1.26264728,
                        .86207916, .92334482, .54183656, 1.83251780,
                        .75851348, .46122388)
        self.bse = (0.036449917, 0.006925765, 0.526405169,
                0.439146744, 0.400820976, 0.346246857, 0.691623875,
                0.459373871, 1.204574885)
        self.aic_R = 219.447991133
        self.aic_Stata = 1.1611
        self.deviance = 201.447991133
        self.scale = 1
        self.llf = -100.7239955662511
        self.null_deviance = 234.671996193219
        self.bic = -742.0665
        self.df_resid = 180
        self.df_model = 8
        self.df_null = 188
        self.pearsonX2 = 182.0233425
        self.resids = glm_test_resids.lbw_resids

class cpunish(object):
    '''
    The following are from the R script in models.datasets.cpunish
    Slightly different than published results, but should be correct
    Probably due to rounding in cleaning?
    '''
    def __init__(self):
        self.params = (2.611017e-04, 7.781801e-02, -9.493111e-02, 2.969349e-01,
                2.301183e+00, -1.872207e+01, -6.801480e+00)
        self.bse = (5.187132e-05, 7.940193e-02, 2.291926e-02, 4.375164e-01,
                4.283826e-01, 4.283961e+00, 4.146850e+00)
        self.null_deviance = 136.57281747225
        self.df_null = 16
        self.deviance = 18.59164
        self.df_resid = 10
        self.df_model = 6
        self.aic_R = 77.85466   # same as Stata
        self.aic_Stata = 4.579686
        self.bic = -9.740492
        self.llf = -31.92732831
        self.scale = 1
        self.pearsonX2 = 24.75374835
        self.resids = glm_test_resids.cpunish_resids

class scotvote(object):
    def __init__(self):
        self.params = (4.961768e-05, 2.034423e-03, -7.181429e-05, 1.118520e-04,
                -1.467515e-07, -5.186831e-04, -2.42717498e-06, -1.776527e-02)
        self.bse = (1.621577e-05, 5.320802e-04, 2.711664e-05, 4.057691e-05,
            1.236569e-07, 2.402534e-04, 7.460253e-07, 1.147922e-02)
        self.null_deviance = 0.536072
        self.df_null = 31
        self.deviance = 0.087388516417
        self.df_resid = 24
        self.df_model = 7
        self.aic_R = 182.947045954721
        self.aic_Stata = 10.72212
        self.bic = -83.09027
        self.llf = -163.5539382 # from Stata, same as ours with scale = 1
        self.llf_R = -82.47352  # Very close to ours as is
        self.scale = 0.003584283
        self.pearsonX2 = .0860228056
        self.resids = glm_test_resids.scotvote_resids

class star98(object):
    def __init__(self):
        self.params = (-0.0168150366,  0.0099254766, -0.0187242148,
            -0.0142385609, 0.2544871730,  0.2406936644,  0.0804086739,
            -1.9521605027, -0.3340864748, -0.1690221685,  0.0049167021,
            -0.0035799644, -0.0140765648, -0.0040049918, -0.0039063958,
            0.0917143006,  0.0489898381,  0.0080407389,  0.0002220095,
            -0.0022492486, 2.9588779262)
        self.bse = (4.339467e-04, 6.013714e-04, 7.435499e-04, 4.338655e-04,
            2.994576e-02, 5.713824e-02, 1.392359e-02, 3.168109e-01,
            6.126411e-02, 3.270139e-02, 1.253877e-03, 2.254633e-04,
            1.904573e-03, 4.739838e-04, 9.623650e-04, 1.450923e-02,
            7.451666e-03, 1.499497e-03, 2.988794e-05, 3.489838e-04,
            1.546712e+00)
        self.null_deviance = 34345.3688931
        self.df_null = 302
        self.deviance = 4078.76541772
        self.df_resid = 282
        self.df_model = 20
        self.aic_R = 6039.22511799
        self.aic_Stata = 19.93144
        self.bic = 2467.494
        self.llf = -2998.612928
        self.scale = 1.
        self.pearsonX2 = 4051.921614
        self.resids = glm_test_resids.star98_resids

class inv_gauss():
    '''
    Data was generated by Hardin and Hilbe using Stata.
    Note only the first 5000 observations are used because
    the models code currently uses np.eye.
    '''
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

    def __init__(self):
        # set up data #
        filename="inv_gaussian.csv"
        data=np.genfromtxt(filename, delimiter=",", skiprows=1)
        self.endog = data[:5000,0]
        self.exog = data[:5000,1:]
        self.exog = models.functions.add_constant(self.exog)
        # Results
#NOTE: loglikelihood difference in R vs. Stata vs. Models
# is the same as gamma
        self.params = (0.4519770, -0.2508288, 1.0359574)
        self.bse = (0.03148291, 0.02237211, 0.03429943)
        self.null_deviance = 1520.673165475461
        self.df_null = 4999
        self.deviance = 1423.943980407997
        self.df_resid = 4997
        self.df_model = 2
        self.aic_R = 5059.41911646446
        self.aic_Stata = 1.55228
        self.bic = -41136.47
        self.llf = -3877.700354 # same as ours with scale set to 1
        self.llf_R = -2525.70955823223  # this is close to our defintion
        self.scale = 0.2867266359127567
        self.pearsonX2 = 1432.771536
        self.resids = glm_test_resids.invgauss_resids


### REGRESSION TESTS ###

class longley(object):
    '''
    The results for the Longley dataset were obtained from NIST

    http://www.itl.nist.gov/div898/strd/general/dataarchive.html

    Other results were obtained from Stata
    '''
    def __init__(self):
        self.params = ( 15.0618722713733, -0.358191792925910E-01,
                 -2.02022980381683, -1.03322686717359, -0.511041056535807E-01,
                 1829.15146461355, -3482258.63459582)
        self.bse = (84.9149257747669, 0.334910077722432E-01,
                   0.488399681651699, 0.214274163161675, 0.226073200069370,
                   455.478499142212, 890420.383607373)
        self.conf_int = [(-177.0291,207.1524),
                   (-.111581,.0399428),(-3.125065,-.9153928),
                   (-1.517948,-.5485049),(-.5625173,.4603083),
                   (798.7873,2859.515),(-5496529,-1467987)]
        self.scale = 92936.0061673238
        self.Rsq = 0.995479004577296
        self.adjRsq = 0
        self.df_model = 6
        self.df_resid = 9
        self.ESS = 184172401.944494
        self.SSR = 836424.055505915
        self.MSE_model = 30695400.3240823
        self.MSE_resid = 92936.0061673238
        self.F = 330.285339234588
        self.llf = -109.6174
        self.AIC = 233.2349
        self.BIC = 238.643
#    sas_bse_HC0=(51.22035, 0.02458, 0.38324, 0.14625, 0.15821,
#                428.38438, 832212,)
#    sas_bse_HC1=(68.29380, 0.03277, 0.51099, 0.19499, 0.21094,
#                571.17917, 1109615)
#    sas_bse_HC2=(67.49208, 0.03653, 0.55334, 0.20522, 0.22324,
#                617.59295, 1202370)
#    sas_bse_HC3=(91.11939, 0.05562, 0.82213, 0.29879, 0.32491,
#                922.80784, 1799477)





