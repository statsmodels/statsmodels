import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.nonparametric as nparam


class MyTest(object):
    def setUp(self):
        N = 60
        np.random.seed(123456)
        self.o = np.random.binomial(2, 0.7, size=(N, 1))
        self.o2 = np.random.binomial(3, 0.7, size=(N, 1))
        self.c1 = np.random.normal(size=(N, 1))
        self.c2 = np.random.normal(10, 1, size=(N, 1))
        self.c3 = np.random.normal(10, 2, size=(N, 1))
        self.noise = np.random.normal(size=(N,1))
        b0 = 0.3; b1 =  1.2; b2 = 3.7 # regression coefficients
        self.y = b0 + b1 * self.c1 + b2 * self.c2 + self.noise

        # Italy data from R's np package (the first 50 obs) R>> data (Italy)

        self.Italy_gdp = [  8.556,  12.262,   9.587,   8.119,   5.537,   6.796,   8.638,
         6.483,   6.212,   5.111,   6.001,   7.027,   4.616,   3.922,
         4.688,   3.957,   3.159,   3.763,   3.829,   5.242,   6.275,
         8.518,  11.542,   9.348,   8.02 ,   5.527,   6.865,   8.666,
         6.672,   6.289,   5.286,   6.271,   7.94 ,   4.72 ,   4.357,
         4.672,   3.883,   3.065,   3.489,   3.635,   5.443,   6.302,
         9.054,  12.485,   9.896,   8.33 ,   6.161,   7.055,   8.717,
         6.95 ]

        self.Italy_year = [1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951,
       1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1952,
       1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952,
       1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952, 1953, 1953,
       1953, 1953, 1953, 1953, 1953, 1953]
        
        # OECD panel data from NP  R>> data(oecdpanel)
        self.growth = [-0.0017584 ,  0.00740688,  0.03424461,  0.03848719,  0.02932506,
        0.03769199,  0.0466038 ,  0.00199456,  0.03679607,  0.01917304,
       -0.00221   ,  0.00787269,  0.03441118, -0.0109228 ,  0.02043064,
       -0.0307962 ,  0.02008947,  0.00580313,  0.00344502,  0.04706358,
        0.03585851,  0.01464953,  0.04525762,  0.04109222, -0.0087903 ,
        0.04087915,  0.04551403,  0.036916  ,  0.00369293,  0.0718669 ,
        0.02577732, -0.0130759 , -0.01656641,  0.00676429,  0.08833017,
        0.05092105,  0.02005877,  0.00183858,  0.03903173,  0.05832116,
        0.0494571 ,  0.02078484,  0.09213897,  0.0070534 ,  0.08677202,
        0.06830603, -0.00041   ,  0.0002856 ,  0.03421225, -0.0036825 ]

        self.oecd = [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
       0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
       0, 0, 0, 0]
        
class TestUKDE(MyTest):

    def test_pdf_mixeddata_CV_LS(self):
        dens_u = nparam.UKDE(tdat=[self.c1, self.o, self.o2], var_type='coo',
                             bw='cv_ls')
        npt.assert_allclose(dens_u.bw, [0.709195, 0.087333, 0.092500],
                            atol=1e-6)
        

        # Matches R to 3 decimals; results seem more stable than with R.
        # Can be checked with following code:
        ## import rpy2.robjects as robjects
        ## from rpy2.robjects.packages import importr
        ## NP = importr('np')
        ## r = robjects.r
        ## D = {"S1": robjects.FloatVector(c1), "S2":robjects.FloatVector(c2),
        ##      "S3":robjects.FloatVector(c3), "S4":robjects.FactorVector(o),
        ##      "S5":robjects.FactorVector(o2)}
        ## df = robjects.DataFrame(D)
        ## formula = r('~S1+ordered(S4)+ordered(S5)')
        ## r_bw = NP.npudensbw(formula, data=df, bwmethod='cv.ls')

    def test_pdf_mixeddata_LS_vs_ML(self):
        dens_ls = nparam.UKDE(tdat=[self.c1, self.o, self.o2], var_type='coo',
                             bw='cv_ls')
        dens_ml = nparam.UKDE(tdat=[self.c1, self.o, self.o2], var_type='coo',
                             bw='cv_ml')
        npt.assert_allclose(dens_ls.bw, dens_ml.bw, atol=0, rtol=0.5)

    def test_pdf_mixeddata_CV_ML(self):
        # Test ML cross-validation
        dens_ml = nparam.UKDE(tdat=[self.c1, self.o, self.c2], var_type='coc',
                             bw='cv_ml')
        
        
        
    def test_pdf_continuous(self):
        # Test for only continuous data
        
        dens = nparam.UKDE (tdat = [self.growth, self.Italy_gdp], var_type = 'cc',
                            bw = 'cv_ls')
        sm_result = np.squeeze(dens.pdf()[0:5])  # take the first data points from the training set
        R_result = [1.6202284, 0.7914245, 1.6084174, 2.4987204, 1.3705258]

        ## CODE TO REPRODUCE THE RESULTS IN R
        ## library(np)
        ## data(oecdpanel)
        ## data (Italy)
        ## bw <-npudensbw(formula = ~oecdpanel$growth[1:50] + Italy$gdp[1:50], bwmethod ='cv.ls')
        ## fhat <- fitted(npudens(bws=bw))
        ## fhat[1:5]
        
        npt.assert_allclose(sm_result, R_result, atol = 1e-3)
        
        
    def test_pdf_ordered(self):
        # Test for only ordered data
        dens = nparam.UKDE (tdat = [self.oecd], var_type ='o', bw = 'cv_ls')
        sm_result = np.squeeze(dens.pdf()[0:5])
        R_result = [0.7236395, 0.7236395, 0.2763605, 0.2763605, 0.7236395]
        npt.assert_allclose(sm_result, R_result, atol = 1e-1)  # lower tol here. only 2nd decimal
        
        
    def test_unordered_CV_LS(self):
        dens = nparam.UKDE(tdat=[self.growth, self.oecd], var_type = 'cu', bw ='cv_ls')
        R_result = [0.0052051, 0.05835941]
        npt.assert_allclose(dens.bw, R_result, atol = 1e-2)
        

    def test_beta(self, a=2, b=5, N=250):
        support = np.random.beta(a,b, size = N)
        rv = stats.beta(a,b)
        ix = np.argsort(support)
        
        dens_normal = nparam.UKDE(tdat=[support], var_type='c', bw='normal_reference')
        dens_cvls = nparam.UKDE(tdat=[support], var_type='c', bw='cv_ls')
        dens_cvml = nparam.UKDE(tdat=[support], var_type='c', bw='cv_ml')
        plt.plot(support[ix],rv.pdf(support[ix]), label='Actual')
        plt.plot(support[ix],dens_normal.pdf()[ix],label='Scott')
        plt.plot(support[ix],dens_cvls.pdf()[ix], label='CV_LS')
        plt.plot(support[ix],dens_cvml.pdf()[ix], label='CV_ML')
        plt.title("Nonparametric Estimation of the Density of Beta Distributed Random Variable")
        plt.legend(('Actual','Scott','CV_LS','CV_ML'))
        plt.show()
        
    def test_f(self, df=100, dn=100, N=250):
        support = np.random.f(dn,df, size = N)
        rv = stats.f(df,dn)
        ix = np.argsort(support)
        
        dens_normal = nparam.UKDE(tdat=[support], var_type='c', bw='normal_reference')
        dens_cvls = nparam.UKDE(tdat=[support], var_type='c', bw='cv_ls')
        dens_cvml = nparam.UKDE(tdat=[support], var_type='c', bw='cv_ml')
        plt.plot(support[ix],rv.pdf(support[ix]), label='Actual')
        plt.plot(support[ix],dens_normal.pdf()[ix],label='Scott')
        plt.plot(support[ix],dens_cvls.pdf()[ix], label='CV_LS')
        plt.plot(support[ix],dens_cvml.pdf()[ix], label='CV_ML')
        plt.title("Nonparametric Estimation of the Density of f Distributed Random Variable")
        plt.legend(('Actual','Scott','CV_LS','CV_ML'))
        plt.show()
    def test_pareto(self, a=2, N=150):
        support = np.random.pareto(a, size = N)
        rv = stats.pareto(a)
        ix = np.argsort(support)
        
        dens_normal = nparam.UKDE(tdat=[support], var_type='c', bw='normal_reference')
        dens_cvls = nparam.UKDE(tdat=[support], var_type='c', bw='cv_ls')
        dens_cvml = nparam.UKDE(tdat=[support], var_type='c', bw='cv_ml')
        plt.plot(support[ix],rv.pdf(support[ix]), label='Actual')
        plt.plot(support[ix],dens_normal.pdf()[ix],label='Scott')
        plt.plot(support[ix],dens_cvls.pdf()[ix], label='CV_LS')
        plt.plot(support[ix],dens_cvml.pdf()[ix], label='CV_ML')
        plt.title("Nonparametric Estimation of the Density of Pareto Distributed Random Variable")
        plt.legend(('Actual','Scott','CV_LS','CV_ML'))
        plt.show()
        
    def test_laplace(self, mu=0, s=1, N=250):
        support = np.random.laplace(mu, s, size = N)
        rv = stats.laplace(mu, s)
        ix = np.argsort(support)
        
        dens_normal = nparam.UKDE(tdat=[support], var_type='c', bw='normal_reference')
        dens_cvls = nparam.UKDE(tdat=[support], var_type='c', bw='cv_ls')
        dens_cvml = nparam.UKDE(tdat=[support], var_type='c', bw='cv_ml')
        plt.plot(support[ix],rv.pdf(support[ix]), label='Actual')
        plt.plot(support[ix],dens_normal.pdf()[ix],label='Scott')
        plt.plot(support[ix],dens_cvls.pdf()[ix], label='CV_LS')
        plt.plot(support[ix],dens_cvml.pdf()[ix], label='CV_ML')
        plt.title("Nonparametric Estimation of the Density of Laplace Distributed Random Variable")
        plt.legend(('Actual','Scott','CV_LS','CV_ML'))
        plt.show()
        
    def test_weibull(self, a=1, N=250):
        support = np.random.weibull(a, size = N)
        rv = stats.weibull_min(a)
        
        ix = np.argsort(support)
        dens_normal = nparam.UKDE(tdat=[support], var_type='c', bw='normal_reference')
        dens_cvls = nparam.UKDE(tdat=[support], var_type='c', bw='cv_ls')
        dens_cvml = nparam.UKDE(tdat=[support], var_type='c', bw='cv_ml')
        plt.plot(support[ix],rv.pdf(support[ix]), label='Actual')
        plt.plot(support[ix],dens_normal.pdf()[ix],label='Scott')
        plt.plot(support[ix],dens_cvls.pdf()[ix], label='CV_LS')
        plt.plot(support[ix],dens_cvml.pdf()[ix], label='CV_ML')
        plt.title("Nonparametric Estimation of the Density of Laplace Distributed Random Variable")
        plt.legend(('Actual','Scott','CV_LS','CV_ML'))
        plt.show()

    def test_power(self, a=0.7, N=250):
        support = np.random.power(a, size = N)
        rv = stats.powerlaw(a)
        
        ix = np.argsort(support)
        dens_normal = nparam.UKDE(tdat=[support], var_type='c', bw='normal_reference')
        dens_cvls = nparam.UKDE(tdat=[support], var_type='c', bw='cv_ls')
        dens_cvml = nparam.UKDE(tdat=[support], var_type='c', bw='cv_ml')
        plt.plot(support[ix],rv.pdf(support[ix]), label='Actual')
        plt.plot(support[ix],dens_normal.pdf()[ix],label='Scott')
        plt.plot(support[ix],dens_cvls.pdf()[ix], label='CV_LS')
        plt.plot(support[ix],dens_cvml.pdf()[ix], label='CV_ML')
        plt.title("Nonparametric Estimation of the Density of Power-law Distributed Random Variable")
        plt.legend(('Actual','Scott','CV_LS','CV_ML'))
        plt.show()
                
    def test_poisson(self, a=2, N=250):
        support = np.random.poisson(a, size = N)
        rv = stats.poisson(a)
        
        ix = np.argsort(support)
        dens_normal = nparam.UKDE(tdat=[support], var_type='o', bw='normal_reference')
        dens_cvls = nparam.UKDE(tdat=[support], var_type='o', bw='cv_ls')
        dens_cvml = nparam.UKDE(tdat=[support], var_type='o', bw='cv_ml')
        plt.plot(support[ix],rv.pmf(support[ix]), label='Actual')
        plt.plot(support[ix],dens_normal.pdf()[ix],label='Scott')
        plt.plot(support[ix],dens_cvls.pdf()[ix], label='CV_LS')
        plt.plot(support[ix],dens_cvml.pdf()[ix], label='CV_ML')
        plt.title("Nonparametric Estimation of the Density of Poisson Distributed Random Variable")
        plt.legend(('Actual','Scott','CV_LS','CV_ML'))
        plt.show()
    def test_continuous_cdf(self,edat=None):
        dens = nparam.UKDE (tdat = [self.Italy_gdp, self.growth], var_type = 'cc',
                            bw = 'cv_ml')
        sm_result = dens.cdf()[0:5]
        R_result = [0.192180770,0.299505196,0.557303666,
                    0.513387712, 0.210985350]
        npt.assert_allclose(sm_result, R_result, atol = 1e-3)
    def test_mixeddata_cdf(self, edat=None):
        dens = nparam.UKDE(tdat = [self.Italy_gdp, self.oecd], var_type='cu',
                           bw='cv_ml')
        sm_result = dens.cdf()[0:5]
        R_result =  [0.54700010, 0.65907039, 0.89676865, 0.74132941, 0.25291361]
        npt.assert_allclose(sm_result, R_result, atol = 1e-3)
         

class TestCKDE(MyTest):
    def test_mixeddata_CV_LS (self):
        dens_ls = nparam.CKDE(tydat=[self.Italy_gdp],txdat=[self.Italy_year],
                                           dep_type='c',indep_type='o',bw='cv_ls')
        npt.assert_allclose(dens_ls.bw, [1.6448, 0.2317373], atol=1e-3)  # Values from the estimation in R with the same data
        
    def test_continuous_CV_ML (self):

        dens_ml = nparam.CKDE (tydat=[self.Italy_gdp],txdat=[self.growth],
                                           dep_type='c',indep_type='c',bw='cv_ml')
        npt.assert_allclose(dens_ml.bw, [0.5341164,0.04510836], atol = 1e-3) # Results from R

    def test_unordered_CV_LS (self):
        dens_ls = nparam.CKDE(tydat=[self.oecd],txdat=[self.growth],
                                           dep_type='u',indep_type='c',bw='cv_ls')
        print dens_ls.bw
        print "Test completed"
        #not a good match. needs more work
        

    def test_pdf_continuous(self):
        dens = nparam.CKDE (tydat = [self.growth], txdat = [self.Italy_gdp],
                            dep_type ='c', indep_type ='c', bw='cv_ml')
        sm_result = np.squeeze(dens.pdf()[0:5])
        R_result = [11.97964, 12.73290, 13.23037, 13.46438, 12.22779]
        npt.assert_allclose(sm_result, R_result, atol = 1e-3)
        
    
    def test_pdf_mixeddata(self):
        dens = nparam.CKDE(tydat=[self.Italy_gdp],txdat=[self.Italy_year],
                                           dep_type='c',indep_type='o',bw='cv_ls')
        sm_result = np.squeeze(dens.pdf()[0:5])
        
        R_result = [0.08469226, 0.01737731, 0.05679909, 0.09744726, 0.15086674]

        ## CODE TO REPRODUCE IN R
        ## library(np)
        ## data (Italy)
        ## bw <- npcdensbw(formula = Italy$gdp[1:50]~ordered(Italy$year[1:50]),bwmethod='cv.ls')
        ## fhat <- fitted(npcdens(bws=bw))
        ## fhat[1:5]
        npt.assert_allclose(sm_result, R_result, atol = 1e-3)

    def test_continuous_normal_ref(self):
        # test for normal reference rule of thumb with continuous data
        
        dens_nm = nparam.CKDE(tydat=[self.Italy_gdp],txdat=[self.growth],
                                           dep_type='c',indep_type='c',bw='normal_reference')
        sm_result = dens_nm.bw
        R_result = [1.283532,0.01535401]
        npt.assert_allclose(sm_result, R_result, atol = 1e-1)  # Here we need a smaller tolerance.check!

    def test_continuous_cdf(self):
        dens_nm = nparam.CKDE(tydat=[self.Italy_gdp],
                              txdat=[self.growth], dep_type='c',indep_type='c',bw='normal_reference')
        sm_result = dens_nm.cdf()[0:5]
        R_result = [0.81304920, 0.95046942, 0.86878727, 0.71961748, 0.38685423]
        npt.assert_allclose(sm_result, R_result, atol = 1e-3)
        
    def test_mixeddata_cdf(self):
        dens = nparam.CKDE(tydat=[self.Italy_gdp],txdat=[self.Italy_year],
                                           dep_type='c',indep_type='o',bw='cv_ls')
        sm_result = dens.cdf()[0:5]
        R_result = [0.8118257, 0.9724863, 0.8843773, 0.7720359, 0.4361867]
        npt.assert_allclose(sm_result, R_result, atol = 1e-3)
        
class TestReg(MyTest):
    def test_ordered_CVLC(self):
        model = nparam.Reg(tydat=[self.Italy_gdp], txdat=[self.Italy_year],
                           var_type='o', bw='cv_lc')
        sm_bw = model.bw
        R_bw = 0.1390096

        sm_mean = model.Cond_Mean()[0]
        R_mean = 6.190486

        sm_R2 = model.R2()
        R_R2 = 0.1435323
        
        
        ## CODE TO REPRODUCE IN R
        ## library(np)
        ## data(Italy)
        ## attach(Italy)
        ## bw <- npregbw(formula=gdp[1:50]~ordered(year[1:50]))
        npt.assert_allclose(sm_bw, R_bw, atol = 1e-2)
        npt.assert_allclose(sm_mean, R_mean, atol = 1e-2)
        npt.assert_allclose(sm_R2, R_R2, atol = 1e-2)
    def test_continuousdata_CVLC(self):
        model = nparam.Reg(tydat=[self.y], txdat=[self.c1,self.c2],
                           var_type='cc', bw='cv_lc')
        # Bandwidth
        sm_bw = model.bw
        R_bw = [0.6163835, 0.1649656]
        # Conditional Mean
        sm_mean = model.Cond_Mean()[0:5]
        R_mean = [31.49157, 37.29536, 43.72332, 40.58997, 36.80711]
        # R-Squared
        sm_R2 = model.R2()
        R_R2 = 0.956381720885
        
        npt.assert_allclose(sm_bw, R_bw, atol = 1e-2)
        npt.assert_allclose(sm_mean, R_mean, atol = 1e-2)
        npt.assert_allclose(sm_R2, R_R2, atol = 1e-2)
        
    
