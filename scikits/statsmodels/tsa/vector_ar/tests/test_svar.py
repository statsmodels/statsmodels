"""
Test SVAR estimation
"""

import scikits.statsmodels.api as sm
from scikits.statsmodels.tsa.vector_ar.var_model import SVAR
from numpy.testing import assert_almost_equal, assert_equal
from results import results_svar
import numpy as np
import numpy.testing as npt

DECIMAL_6 = 6
DECIMAL_5 = 5
DECIMAL_4 = 4

class TestSVAR(object):
    @classmethod
    def setupClass(cls):
        mdata = sm.datasets.macrodata.load().data
        mdata = mdata[['realgdp','realcons','realinv']]
        names = mdata.dtype.names
        data = mdata.view((float,3))
        data = np.diff(np.log(data), axis=0)
        A = np.asarray([[1, 0, 0],['E', 1, 0],['E', 'E', 1]])
        B = np.asarray([['E', 0, 0], [0, 'E', 0], [0, 0, 'E']])
        results = SVAR(data, svar_type='AB', A=A, B=B).fit(maxlags=3)
        cls.res1 = results
        cls.res2 = results_svar.SVARdataResults()
    def test_A(self):
        assert_almost_equal(self.res1.A, self.res2.A)
    def test_B(self):
        assert_almost_equal(self.res1.B, self.res2.B)



