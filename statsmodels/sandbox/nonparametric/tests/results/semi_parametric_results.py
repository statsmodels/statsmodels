"""
Test Results for Semi-Parametric regression. Validated against R where possible.

References
----------
[1] R 'np' vignette : http://cran.r-project.org/web/packages/np/vignettes/np.pdf

"""
import os
import numpy as np
import pandas as pd
from os.path import dirname, abspath

cur_dir = os.path.abspath(os.path.dirname(__file__))
filepath = dirname(abspath(__file__))

class Wage1(object):

    """ Results in this class are based on the example given in the example of
        section 8 in [1]. Currently results are not validated against R output
        as not all analogous options are avaliable. """

    def __init__(self):
        self.nobs = 526
        self.fit_wage = pd.read_csv(filepath + '/wage_semiparametric_fit_results.csv', delimiter=",")

    def semilinear(self):
        self.b = np.array([ 0.29242067, -0.0326906 ,  0.07931927,  0.01637583])
        self.bw = np.array([ 3.95951858])
        self.mean = self.fit_wage['wage1_semilinear_default_mean']
        self.mfx = self.fit_wage['wage1_semilinear_default_mfx']
        self.r_squared = 0.44763372545524732

class BirthWt(object):

    """ Results in this class are based on the example given in the example of
        section 9 in [1]. Currently results are not validated against R output
        as not all analogous options are avaliable. """


    def __init__(self):
        self.nobs = 189
        self.fit_birthwt = pd.read_csv(filepath + '/birthwt_semiparametric_fit_results.csv', delimiter=",")

    def singleindexmodel(self):
        self.b = np.array([ -239955.72735867,    10402.99668618,   325314.27425821,
       -1173334.6503826 ,  -770838.63996199,   329940.04038314,
          85409.76410862])
        self.bw = np.array([  4.46624513e-06])
        self.mean = self.fit_birthwt['birthwt_singleindex_default_mean']
        self.mfx = self.fit_birthwt['birthwt_singleindex_default_mfx']
        self.r_squared = 0.99999999999829958


