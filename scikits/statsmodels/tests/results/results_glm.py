"""
Results for test_glm.py.

Hard-coded from R or Stata
"""
import numpy as np
import glm_test_resids
import os
import scikits.statsmodels as sm

class Longley(object):
    """
    Tests GLM with Gaussian Family and default link.

    Results are from Stata and R
    """
    def __init__(self):
        self.resids = np.array([[ 267.34002976,  267.34002976,  267.34002976,
            267.34002976, 267.34002976],
            [ -94.0139424 ,  -94.0139424 ,  -94.0139424 ,  -94.0139424 ,
             -94.0139424 ],
            [  46.28716776,   46.28716776,   46.28716776,   46.28716776,
              46.28716776],
           [-410.11462193, -410.11462193, -410.11462193, -410.11462193,
            -410.11462193],
           [ 309.71459076,  309.71459076,  309.71459076,  309.71459076,
             309.71459076],
           [-249.31121533, -249.31121533, -249.31121533, -249.31121533,
            -249.31121533],
           [-164.0489564 , -164.0489564 , -164.0489564 , -164.0489564 ,
            -164.0489564 ],
           [ -13.18035687,  -13.18035687,  -13.18035687,  -13.18035687,
             -13.18035687],
           [  14.3047726 ,   14.3047726 ,   14.3047726 ,   14.3047726 ,
              14.3047726 ],
           [ 455.39409455,  455.39409455,  455.39409455,  455.39409455,
             455.39409455],
           [ -17.26892711,  -17.26892711,  -17.26892711,  -17.26892711,
             -17.26892711],
           [ -39.05504252,  -39.05504252,  -39.05504252,  -39.05504252,
             -39.05504252],
           [-155.5499736 , -155.5499736 , -155.5499736 , -155.5499736 ,
            -155.5499736 ],
           [ -85.67130804,  -85.67130804,  -85.67130804,  -85.67130804,
             -85.67130804],
           [ 341.93151396,  341.93151396,  341.93151396,  341.93151396,
             341.93151396],
           [-206.75782519, -206.75782519, -206.75782519, -206.75782519,
            -206.75782519]])
        self.null_deviance = 185008826 # taken from R.
        self.params = np.array([  1.50618723e+01,  -3.58191793e-02,
            -2.02022980e+00, -1.03322687e+00,  -5.11041057e-02,
            1.82915146e+03, -3.48225863e+06])
        self.bse = np.array([8.49149258e+01,   3.34910078e-02, 4.88399682e-01,
          2.14274163e-01,   2.26073200e-01,   4.55478499e+02, 8.90420384e+05])
        self.aic_R = 235.23486961695903 # R adds 2 for dof to AIC
        self.aic_Stata = 14.57717943930524  # stata divides by nobs
        self.deviance = 836424.0555058046   # from R
        self.scale = 92936.006167311629
        self.llf = -109.61743480847952
        self.null_deviance = 68445976650.0
        self.bic_Stata = 836399.1760177979 # no bic in R?
        self.df_model = 6
        self.df_resid = 9
        self.chi2 = 1981.711859508729    #TODO: taken from Stata not available
                                        # in sm yet
#        self.pearson_chi2 = 836424.1293162981   # from Stata (?)
        self.fittedvalues = np.array([60055.659970240202, 61216.013942398131,
                     60124.71283224225, 61597.114621930756, 62911.285409240052,
                     63888.31121532945, 65153.048956395127, 63774.180356866214,
                     66004.695227399934, 67401.605905447621,
                     68186.268927114084,  66552.055042522494,
                     68810.549973595422, 69649.67130804155, 68989.068486039061,
                     70757.757825193927])

class GaussianLog(object):
    """
    Uses generated data.  These results are from R and Stata.
    """
    def __init__(self):
        self.resids = np.genfromtxt('./glm_gaussian_log_resid.csv', ',')
        self.null_deviance = 56.691617808182208
        self.params = np.array([9.99964386e-01,-1.99896965e-02,
            -1.00027232e-04])
        self.bse = np.array([1.42119293e-04, 1.20276468e-05, 1.87347682e-07])
        self.aic_R = -1103.8187213072656 # adds 2 for dof for scale
        self.aic_Stata = -11.05818072104212 # divides by nobs for e(aic)
        self.deviance = 8.68876986288542e-05
        self.scale = 8.9574946938163984e-07 # from R but e(phi) in Stata
        self.llf = 555.9093606536328
        self.null_deviance = 56.691617808182208
        self.bic_Stata = -446.7014211525822
        self.df_model = 2
        self.df_resid = 97
        self.chi2 = 33207648.86501769
        self.fittedvalues = np.array([2.7181850213327747,  2.664122305869506,
             2.6106125414084405, 2.5576658143523567, 2.5052916730829535,
             2.4534991313100165, 2.4022966718815781, 2.3516922510411282,
             2.3016933031175575, 2.2523067456332542, 2.2035389848154616,
             2.1553959214958001, 2.107882957382607, 2.0610050016905817,
             2.0147664781120667, 1.969171332114154, 1.9242230385457144,
             1.8799246095383746, 1.8362786026854092, 1.7932871294825108,
             1.7509518640143886, 1.7092740518711942, 1.6682545192788105,
             1.6278936824271399, 1.5881915569806042, 1.5491477677552221,
             1.5107615585467538, 1.4730318020945796, 1.4359570101661721,
             1.3995353437472129, 1.3637646233226499, 1.3286423392342188,
             1.2941656621002184, 1.2603314532836074, 1.2271362753947765,
             1.1945764028156565, 1.162647832232141, 1.1313462931621328,
             1.1006672584668622, 1.0706059548334832, 1.0411573732173065,
             1.0123162792324054, 0.98407722347970683, 0.95643455180206194,
             0.92938241545618494, 0.90291478119174029, 0.87702544122826565,
             0.85170802312101246, 0.82695599950720078, 0.80276269772458597,
             0.77912130929465073, 0.75602489926313921, 0.73346641539106316,
             0.71143869718971686, 0.68993448479364294, 0.66894642766589496,
             0.64846709313034534, 0.62848897472617915, 0.60900450038011367,
             0.5900060403922629, 0.57148591523195513, 0.55343640314018494,
             0.5358497475357491, 0.51871816422248385, 0.50203384839536769,
             0.48578898144361343, 0.46997573754920047, 0.45458629007964013,
             0.4396128177740814, 0.42504751072218311, 0.41088257613548018,
             0.39711024391126759, 0.38372277198930843, 0.37071245150195081,
             0.35807161171849949, 0.34579262478494655, 0.33386791026040569,
             0.32228993945183393, 0.31105123954884056, 0.30014439756060574,
             0.28956206405712448, 0.27929695671718968, 0.26934186368570684,
             0.25968964674310463, 0.25033324428976694, 0.24126567414856051,
             0.23248003618867552, 0.22396951477412205, 0.21572738104035141,
             0.20774699500257574, 0.20002180749946474, 0.19254536197598673,
             0.18531129610924435, 0.17831334328122878, 0.17154533390247831,
             0.16500119659068577, 0.15867495920834204, 0.15256074976354628,
             0.14665279717814039, 0.14094543192735109])



class Lbw(object):
    '''
    The LBW data can be found here

    http://www.stata-press.com/data/r9/rmain.html
    '''
    def __init__(self):
        # data set up for data not in datasets
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            "stata_lbw_glm.csv")
        data=np.recfromcsv(filename, converters={4: lambda s: s.strip("\"")})
        data = sm.tools.categorical(data, col='race', drop=True)
        self.endog = data.low
        design = np.column_stack((data['age'], data['lwt'],
                    data['_black'], data['_other'], data['smoke'], data['ptl'],
                    data['ht'], data['ui']))
        self.exog = sm.add_constant(design)
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
        self.pearson_chi2 = 182.0233425
        self.resids = glm_test_resids.lbw_resids

class Cancer(object):
    '''
    The Cancer data can be found here

    http://www.stata-press.com/data/r10/rmain.html
    '''
    def __init__(self):
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            "stata_cancer_glm.csv")
        data = np.recfromcsv(filename)
        self.endog = data.studytime
        design = np.column_stack((data.age,data.drug))
        design = sm.tools.categorical(design, col=1, drop=True)
        design = np.delete(design, 1, axis=1) # drop first dummy
        self.exog = sm.add_constant(design)

class Medpar1(object):
    '''
    The medpar1 data can be found here

    http://www.stata-press.com/data/hh2/medpar1
    '''
    def __init__(self):
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            "stata_medpar1_glm.csv")
        data = np.recfromcsv(filename, converters ={1: lambda s: s.strip("\"")})
        self.endog = data.los
        design = np.column_stack((data.admitype, data.codes))
        design = sm.tools.categorical(design, col=0, drop=True)
        design = np.delete(design, 1, axis=1) # drop first dummy
        self.exog = sm.add_constant(design)


class Cpunish(object):
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
        self.pearson_chi2 = 24.75374835
        self.resids = glm_test_resids.cpunish_resids

class Scotvote(object):
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
        self.pearson_chi2 = .0860228056
        self.resids = glm_test_resids.scotvote_resids

class Star98(object):
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
        self.pearson_chi2 = 4051.921614
        self.resids = glm_test_resids.star98_resids

class InvGauss():
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
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            "inv_gaussian.csv")
        data=np.genfromtxt(filename, delimiter=",", dtype=float)[1:]
        self.endog = data[:5000,0]
        self.exog = data[:5000,1:]
        self.exog = sm.add_constant(self.exog)
        # Results
#NOTE: loglikelihood difference in R vs. Stata vs. Models
# is the same situation as gamma
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
        self.pearson_chi2 = 1432.771536
        self.resids = glm_test_resids.invgauss_resids
