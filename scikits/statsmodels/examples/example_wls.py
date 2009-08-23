"""
extract example from test_wls
"""

import numpy as np
from numpy.random import standard_normal
from numpy.testing import *
from scipy.linalg import toeplitz
from scikits.statsmodels.tools import add_constant
from scikits.statsmodels.regression import OLS, GLSAR, WLS, GLS, yule_walker
import scikits.statsmodels
from scikits.statsmodels import tools
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
from scikits.statsmodels.datasets.ccard.data import load
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

# llf is correct now
##print 'GLS llf corrected: ' # which one
##print -np.log(np.linalg.det(np.diag(1/weights)))/2. + self.res1.llf
##print  np.log(np.linalg.det(np.diag(weights)))/2. + self.res1.llf


# comparison with anova on the model in R
'''
>>> r.anova(self.res2.robj)
{'Df': [1, 1, 1, 1, 1, 67], 'Sum Sq': [33023.458414240784, 6425.0947815597365, 656.2454976748644, 797.81634550895842, 1.3428271834340058, 29269.692912443163], 'F value': [75.59258378189277, 14.707409184381831, 1.5021834522057456, 1.8262472144480861, 0.0030738081728158648, 1.#QNAN], 'Mean Sq': [33023.458414240784, 6425.0947815597365, 656.2454976748644, 797.81634550895842, 1.3428271834340058, 436.86108824542032], 'Pr(>F)': [1.3518684874757728e-012, 0.00028008727120000537, 0.22462655207459495, 0.1811169042369537, 0.95595138678960656, 1.#QNAN]}
>>> self.res1.mse_model
8180.7915732335559
>>> self.res1.fvalue
18.726299488220249
>>> self.res1.mse_resid
436.86108824542038
>>> self.res1.ess, self.res1.uncentered_tss, self.res1.ssr
(40903.957866167781, 70173.650778610943, 29269.692912443166)
>>> ran = r.anova(self.res2.robj)
>>> np.sum(ran['Sum Sq'][:-1])
40903.957866167781
>>> np.sum(ran['Sum Sq'])
70173.650778610943
>>> np.sum(ran['Sum Sq'][:-1]), np.sum(ran['Sum Sq']), np.sum(ran['Sum Sq'][-1])
(40903.957866167781, 70173.650778610943, 29269.692912443163)
>>> self.res1.ess, self.res1.uncentered_tss, self.res1.ssr
(40903.957866167781, 70173.650778610943, 29269.692912443166)
>>> np.sum(ran['Sum Sq'][:-1]) - self.res1.ess
0.0
>>> np.sum(ran['Sum Sq']) - self.res1.uncentered_tss
0.0
>>> self.res1.ssr - np.sum(ran['Sum Sq'][-1])
3.637978807091713e-012
>>>
'''

