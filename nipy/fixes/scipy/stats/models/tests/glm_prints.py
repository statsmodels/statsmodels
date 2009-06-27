import numpy as np
import nipy.fixes.scipy.stats.models.glm as glm
import nipy.fixes.scipy.stats.models as SSM

from exampledata import lbw
from nipy.fixes.scipy.stats.models.functions import xi, add_constant

X=lbw()
X=xi(X, col='race', drop=True)
des = np.column_stack((X['age'],X['lwt'],X['black'],X['other'],X['smoke'], X['ptl'], X['ht'], X['ui']))
des = add_constant(des)
model = glm(X.low, des, family=SSM.family.Binomial())
results = model.fit()
