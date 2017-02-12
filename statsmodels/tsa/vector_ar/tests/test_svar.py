"""
Test SVAR estimation
"""

import statsmodels.api as sm
from statsmodels.tsa.vector_ar.svar_model import SVAR
from numpy.testing import assert_almost_equal, assert_equal, assert_allclose
from .results import results_svar
import numpy as np
import numpy.testing as npt

DECIMAL_6 = 6
DECIMAL_5 = 5
DECIMAL_4 = 4

class TestSVAR(object):
    @classmethod
    def setupClass(cls):
        mdata = sm.datasets.macrodata.load_pandas().data
        mdata = mdata[['realgdp','realcons','realinv']]
        data = mdata.values
        data = np.diff(np.log(data), axis=0)
        A = np.asarray([[1, 0, 0],['E', 1, 0],['E', 'E', 1]])
        B = np.asarray([['E', 0, 0], [0, 'E', 0], [0, 0, 'E']])
        results = SVAR(data, svar_type='AB', A=A, B=B).fit(maxlags=3)
        cls.res1 = results
        #cls.res2 = results_svar.SVARdataResults()
        from .results import results_svar_st
        cls.res2 = results_svar_st.results_svar1_small


    def _reformat(self, x):
        return x[[1, 4, 7, 2, 5, 8, 3, 6, 9, 0], :].ravel("F")


    def test_A(self):
        assert_almost_equal(self.res1.A, self.res2.A, DECIMAL_4)


    def test_B(self):
        # see issue #3148, adding np.abs to make solution positive
        # general case will need positive sqrt of covariance matrix
        assert_almost_equal(np.abs(self.res1.B), self.res2.B, DECIMAL_4)


    def test_basic(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(self._reformat(res1.params), res2.b_var, atol=1e-12)
        bse_st = np.sqrt(np.diag(res2.V_var))
        assert_allclose(self._reformat(res1.bse), bse_st, atol=1e-12)


    def test_llf_ic(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.llf, res2.ll_var, atol=1e-12)
        # different definition, missing constant term ?
        corr_const = -8.51363119922803
        assert_allclose(res1.fpe, res2.fpe_var, atol=1e-12)
        assert_allclose(res1.aic - corr_const, res2.aic_var, atol=1e-12)
        assert_allclose(res1.bic - corr_const, res2.sbic_var, atol=1e-12)
        assert_allclose(res1.hqic - corr_const, res2.hqic_var, atol=1e-12)
