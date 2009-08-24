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
print results.params
stata_lbw_beta = (-.0271003, -.0151508, 1.262647,
                        .8620792, .9233448, .5418366, 1.832518,
                        .7585135, .4612239)
print ' '.join(['%11.7f']*9) % stata_lbw_beta
