'''Example for Generalized Linear Model

* load data
* convert string category to dummy or indicator variable
* initialize model
* call fit() and obtain estimation results
* look at results

'''

import numpy as np

import scikits.statsmodels
from scikits.statsmodels import glm
from scikits.statsmodels.tools import xi, add_constant

from exampledata import lbw

X = lbw()
X = xi(X, col='race', drop=True)
des = np.column_stack((X['age'],X['lwt'],X['black'],X['other'],X['smoke'], X['ptl'], X['ht'], X['ui']))
des = add_constant(des)
model = glm.GLM(X.low, des, family=scikits.statsmodels.family.Binomial())
results = model.fit()
print '\n Parameter Estimates'
print 'our results'
print results.params
stata_lbw_beta = (-.0271003, -.0151508, 1.262647,
                   .8620792,  .9233448,  .5418366,
                  1.832518,   .7585135,  .4612239)
# R results from Rpy_example.py
R_lbw_beta = (-0.0271003108, -0.0151508184,  1.2626472825,
               0.8620791601,  0.9233448207,  0.5418365644,
               1.8325178019,  0.7585134798,  0.4612238834)
print 'stata results'
print ' '.join(['%11.7f']*9) % stata_lbw_beta
print 'R/rpy results'
print ' '.join(['%11.8f']*9) % R_lbw_beta

print '\nstandard errors of parameters'
print results.bse
print '\nt-statistics for parameter estimates'
print results.t()


