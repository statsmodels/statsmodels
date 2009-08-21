"""
extract example from test_wls
"""

import numpy as np
from numpy.random import standard_normal
from numpy.testing import *
from scipy.linalg import toeplitz
from models.tools import add_constant
from models.regression import OLS, AR, WLS, GLS, yule_walker
import models
from models import tools
from scipy.stats import t
from rmodelwrap import RModel
from rpy import r

class Dummy(object):
    pass

self = Dummy()

##def test_wls(:
##    '''
##    GLM results are an implicit test of WLS
##    '''
##    def __init__(self):
from models.datasets.ccard.data import load
data = load()
data.exog = add_constant(data.exog)
weights = 1/data.exog[:,2]**2
self.res1 = WLS(data.endog, data.exog, weights=weights).fit()
self.res2 = RModel(data.endog, data.exog, r.lm,
        weights=weights)
self.res2.wresid = self.res2.rsum['residuals']
self.res2.scale = self.res2.scale**2 # R has sigma not sigma**2
#FIXME: triaged results
self.res1.ess = self.res1.uncentered_tss - self.res1.ssr
self.res1.rsquared = self.res1.ess/self.res1.uncentered_tss
self.res1.mse_model = self.res1.ess/(self.res1.df_model + 1)
self.res1.fvalue = self.res1.mse_model/self.res1.mse_resid
self.res1.rsquared_adj = 1 -(self.res1.nobs)/(self.res1.df_resid)*\
        (1-self.res1.rsquared)

#assert_almost_equal(conf1, conf2, DECIMAL)

print self.res1.rsquared
print self.res2.rsquared
print data.exog.shape
print data.exog[:5,:]
print self.res1.params
print self.res2.params
print self.res1.bse
print self.res2.bse
print self.res1.fvalue
print self.res2.fvalue

print 'GLS llf:           ', self.res1.llf
print 'R   llf:           ', self.res2.llf

print 'GLS llf corrected: ' # which one
print -np.log(np.linalg.det(np.diag(1/weights)))/2. + self.res1.llf
print  np.log(np.linalg.det(np.diag(weights)))/2. + self.res1.llf
