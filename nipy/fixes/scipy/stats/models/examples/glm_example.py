import numpy as np
import models.glm as glm
import models

from exampledata import lbw
from models.tools import xi, add_constant

X=lbw()
X=xi(X, col='race', drop=True)
des = np.column_stack((X['age'],X['lwt'],X['black'],X['other'],X['smoke'], X['ptl'], X['ht'], X['ui']))
des = add_constant(des)
model = glm.GLM(X.low, des, family=models.family.Binomial())
results = model.fit()
print results.params
