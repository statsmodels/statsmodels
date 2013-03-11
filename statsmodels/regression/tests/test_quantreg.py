import os
import statsmodels.api as sm
import numpy as np
from numpy.testing import *
import statsmodels
from statsmodels.regression.quantreg import QuantReg
from patsy import dmatrices
import pandas as pd
from patsy import dmatrices
from statsmodels.regression.quantreg import QuantReg
import statsmodels.api as sm
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.regression.quantreg import QuantReg
#from results_quantreg import *
execfile('results_quantreg.py') # Importing mixes table row order

DECIMAL_14 = 14
DECIMAL_10 = 10
DECIMAL_9 = 9
DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2
DECIMAL_1 = 1
DECIMAL_0 = 0

idx = ['income', 'Intercept']
class CheckModelResults(object):
    def test_params(self):
        assert_almost_equal(np.array(self.res1.params.ix[idx]),
                            self.res2.table[:,0], DECIMAL_3)
    def sparsity(self):
        assert_almost_equal(np.array(self.res1.sparsity),
                            self.res2.sparsity, DECIMAL_3)
    def bandwidth(self):
        assert_almost_equal(np.array(self.res1.bandwidth),
                            self.res2.bandwidth, DECIMAL_3)
    def test_bse(self):
        assert_almost_equal(np.array(self.res1.bse.ix[idx]),
                            self.res2.table[:,1], DECIMAL_3)
    def test_conf_int(self):
        assert_almost_equal(self.res1.conf_int().ix[idx],
                self.res2.table[:,-2:], DECIMAL_3)
    def test_nobs(self):
        assert_almost_equal(self.res1.nobs, self.res2.N, DECIMAL_3)
    def test_df_model(self):
        assert_almost_equal(self.res1.df_model, self.res2.df_m, DECIMAL_3)
    def test_df_resid(self):
        assert_almost_equal(self.res1.df_resid, self.res2.df_r, DECIMAL_3)
    def test_prsquared(self):
        assert_almost_equal(self.res1.prsquared, self.res2.psrsquared, DECIMAL_3)


d = {('biw','bofinger'): biweight_bofinger,
     ('biw','chamberlain'): biweight_chamberlain,
     ('biw','hsheather'): biweight_hsheather,
     ('cos','bofinger'): cosine_bofinger,
     ('cos','chamberlain'): cosine_chamberlain,
     ('cos','hsheather'): cosine_hsheather,
     ('gau','bofinger'): gaussian_bofinger,
     ('gau','chamberlain'): gaussian_chamberlain,
     ('gau','hsheather'): gaussian_hsheather,
     ('par','bofinger'): parzen_bofinger,
     ('par','chamberlain'): parzen_chamberlain,
     ('par','hsheather'): parzen_hsheather,
     ('rec','bofinger'): rectangle_bofinger,
     ('rec','chamberlain'): rectangle_chamberlain,
     ('rec','hsheather'): rectangle_hsheather,
     ('tri','bofinger'): triangle_bofinger,
     ('tri','chamberlain'): triangle_chamberlain,
     ('tri','hsheather'): triangle_hsheather,
     ('epa', 'bofinger'): epanechnikov_bofinger,
     ('epa', 'chamberlain'): epanechnikov_chamberlain,
     ('epa', 'hsheather'): epanechnikov_hsheather,
     #('epa2', 'bofinger'): epan2_bofinger,
     ('epa2', 'chamberlain'): epan2_chamberlain,
     ('epa2', 'hsheather'): epan2_hsheather
     }

def setup_fun(kernel='gau', bandwidth='bofinger'):
    data = sm.datasets.engel.load_pandas().data
    y, X = dmatrices('foodexp ~ income', data, return_type='dataframe')
    statsm = QuantReg(y, X).fit(kernel=kernel, bandwidth=bandwidth)
    stata = d[(kernel, bandwidth)]
    return statsm, stata

class TestEpanechnikovBofinger(CheckModelResults):
    @classmethod
    def setUp(cls):
        cls.res1, cls.res2 = setup_fun('epa', 'bofinger')

class TestEpanechnikovChamberlain(CheckModelResults):
    @classmethod
    def setUp(cls):
        cls.res1, cls.res2 = setup_fun('epa', 'chamberlain')

class TestEpanechnikovHsheather(CheckModelResults):
    @classmethod
    def setUp(cls):
        cls.res1, cls.res2 = setup_fun('epa', 'hsheather')

class TestGaussianBofinger(CheckModelResults):
    @classmethod
    def setUp(cls):
        cls.res1, cls.res2 = setup_fun('gau', 'bofinger')

class TestGaussianChamberlain(CheckModelResults):
    @classmethod
    def setUp(cls):
        cls.res1, cls.res2 = setup_fun('gau', 'chamberlain')

class TestGaussianHsheather(CheckModelResults):
    @classmethod
    def setUp(cls):
        cls.res1, cls.res2 = setup_fun('gau', 'hsheather')

class TestBiweightBofinger(CheckModelResults):
    @classmethod
    def setUp(cls):
        cls.res1, cls.res2 = setup_fun('biw', 'bofinger')

class TestBiweightChamberlain(CheckModelResults):
    @classmethod
    def setUp(cls):
        cls.res1, cls.res2 = setup_fun('biw', 'chamberlain')

class TestBiweightHsheather(CheckModelResults):
    @classmethod
    def setUp(cls):
        cls.res1, cls.res2 = setup_fun('biw', 'hsheather')

class TestCosineBofinger(CheckModelResults):
    @classmethod
    def setUp(cls):
        cls.res1, cls.res2 = setup_fun('cos', 'bofinger')

class TestCosineChamberlain(CheckModelResults):
    @classmethod
    def setUp(cls):
        cls.res1, cls.res2 = setup_fun('cos', 'chamberlain')

class TestCosineHsheather(CheckModelResults):
    @classmethod
    def setUp(cls):
        cls.res1, cls.res2 = setup_fun('cos', 'hsheather')

class TestParzeneBofinger(CheckModelResults):
    @classmethod
    def setUp(cls):
        cls.res1, cls.res2 = setup_fun('par', 'bofinger')

class TestParzeneChamberlain(CheckModelResults):
    @classmethod
    def setUp(cls):
        cls.res1, cls.res2 = setup_fun('par', 'chamberlain')

class TestParzeneHsheather(CheckModelResults):
    @classmethod
    def setUp(cls):
        cls.res1, cls.res2 = setup_fun('par', 'hsheather')

class TestTriangleBofinger(CheckModelResults):
    @classmethod
    def setUp(cls):
        cls.res1, cls.res2 = setup_fun('tri', 'bofinger')

class TestTriangleChamberlain(CheckModelResults):
    @classmethod
    def setUp(cls):
        cls.res1, cls.res2 = setup_fun('tri', 'chamberlain')

class TestTriangleHsheather(CheckModelResults):
    @classmethod
    def setUp(cls):
        cls.res1, cls.res2 = setup_fun('tri', 'hsheather')

#class TestRectangleBofinger(CheckModelResults):
    #@classmethod
    #def setUp(cls):
        #cls.res1, cls.res2 = setup_fun('rec', 'bofinger')

#class TestRectangleChamberlain(CheckModelResults):
    #@classmethod
    #def setUp(cls):
        #cls.res1, cls.res2 = setup_fun('rec', 'chamberlain')

#class TestRectangleHsheather(CheckModelResults):
    #@classmethod
    #def setUp(cls):
        #cls.res1, cls.res2 = setup_fun('rec', 'hsheather')
