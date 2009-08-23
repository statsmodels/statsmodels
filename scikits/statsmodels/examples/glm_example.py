import numpy as np

import scikits.statsmodels.glm as glm
import scikits.statsmodels

from exampledata import lbw
from scikits.statsmodels.tools import xi, add_constant

X=lbw()
X=xi(X, col='race', drop=True)
des = np.column_stack((X['age'],X['lwt'],X['black'],X['other'],X['smoke'], X['ptl'], X['ht'], X['ui']))
des = add_constant(des)
model = glm.GLM(X.low, des, family=scikits.statsmodels.family.Binomial())
results = model.fit()
print results.params
