import numpy.testing as npt
import numpy as np
from scikits.statsmodels.regression.censored_model import Tobit
from scikits.statsmodels.datasets.fair import load as load_fair

class CheckTobit(object):
    pass

class TestTobit(CheckTobit):
    @classmethod
    def setupClass(cls):
        data = load_fair()
        data.exog = sm.add_constant(data.exog)
        cls.res1 = Tobit(data.endog, data.exog, left=0).fit()
