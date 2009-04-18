
from numpy.testing import *

from exampledata import x, y
import nipy.fixes.scipy.stats.models as SSM

def test_armodel():
    for order in range(1,4):
        model = SSM.regression.ARModel(x, order)
        for _ in range(20):
            results = model.fit(y)
            rho, sigma = SSM.regression.yule_walker(y - results.predict,order)
            model = SSM.regression.ARModel(model.design, rho)
        print "AR coefficients:", model.rho

