import numpy as np
import nipy.fixes.scipy.stats.models.glm as glm
import nipy.fixes.scipy.stats.models as SSM

from exampledata import lbw
from nipy.fixes.scipy.stats.models.regression import xi

X=lbw()
X=xi(X, col='race', drop=True)
des = np.vstack((X['age'],X['lwt'],X['black'],X['other'],X['smoke'], X['ptl'], X['ht'], X['ui'])).T
model = glm(design=des, hascons=False, family=SSM.family.Binomial())
results = model.fit(X['low'])
