
from numpy.testing import *

from exampledata import x, y
import neuroimaging.fixes.scipy.stats.models as SSM

# FIXME: AttributeError: 'ARModel' object has no attribute 'yule_walker'
#@dec.knownfailureif(True)
def test_armodel():
    for i in range(1,4):
        model = SSM.regression.ARModel(x, i)
        for ii in range(20):
            results = model.fit(y)
            rho, sigma = SSM.regression.yule_walker(y - results.predict,i)
            model = SSM.regression.ARModel(model.design, rho)
        print "AR coefficients:", model.rho

