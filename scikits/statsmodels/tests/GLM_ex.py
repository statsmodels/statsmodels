# this is old and not adjusted, superseded by examples/glm_example


import numpy as np
from exampledata import lbw
from nipy.fixes.scipy.stats.models.functions import xi, add_constant
from nipy.fixes.scipy.stats.models.glm import GLMBinomial

X=lbw()
X=xi(X, col='race', drop=True)
des = np.vstack((X['age'],X['lwt'],X['black'],X['other'],X['smoke'], X['ptl'], X['ht'], X['ui'])).T
des = add_constant(des)
model = GLMBinomial(X.low, des)
results = model.fit()
print 'Estimation results'
print results.theta
print 'Verified results'
stata_lbw_beta = (-.0271003, -.0151508, 1.262647,
                        .8620792, .9233448, .5418366, 1.832518,
                        .7585135, .4612239)
print stata_lbw_beta

