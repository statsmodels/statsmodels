import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from statsmodels.datasets import heart
from statsmodels.emplike.aft_el import emplikeAFT
from statsmodels.regression.linear_model import WLS
from statsmodels.tools import add_constant

from .results.el_results import AFTRes


class GenRes:
    @classmethod
    def setup_class(cls):
        data = heart.load()
        data.endog = np.asarray(data.endog)
        data.exog = np.asarray(data.exog)
        endog = np.log10(data.endog)
        exog = add_constant(data.exog)
        cls.mod1 = emplikeAFT(endog, exog, data.censors)
        cls.res1 = cls.mod1.fit()
        cls.res2 = AFTRes()


class Test_AFTModel(GenRes):

    def test_params(self):
        assert_almost_equal(self.res1.params(), self.res2.test_params, decimal=4)

    def test_predict(self):
        mod = self.mod1
        params = self.res1.params()

        # documented formula: the linear predictor is exog @ params
        pred = mod.predict(params)
        assert_almost_equal(pred, np.dot(mod.exog, params), decimal=10)

        # explicit exog argument, including out-of-sample-shaped input
        new_exog = mod.exog[:5]
        pred_new = mod.predict(params, exog=new_exog)
        assert_almost_equal(pred_new, np.dot(new_exog, params), decimal=10)
        assert_almost_equal(pred_new, pred[:5], decimal=10)

        # independent cross-check: params() is exactly the coefficients of
        # a Kaplan-Meier weighted WLS fit of endog on exog, so the AFT
        # linear predictor exog @ params must equal that WLS fit's own
        # fittedvalues.
        modif_censors = np.copy(mod.censors)
        modif_censors[-1] = 1
        wts = mod._make_km(mod.endog, modif_censors)
        wls_res = WLS(mod.endog, mod.exog, wts).fit()
        assert_almost_equal(pred, wls_res.fittedvalues, decimal=8)

    @pytest.mark.thread_unsafe("Reuses the same emplikeAFT result object")
    def test_beta0(self):
        assert_almost_equal(
            self.res1.test_beta([4], [0]), self.res2.test_beta0, decimal=4
        )

    @pytest.mark.thread_unsafe("Reuses the same emplikeAFT result object")
    def test_beta1(self):
        assert_almost_equal(
            self.res1.test_beta([-0.04], [1]), self.res2.test_beta1, decimal=4
        )

    @pytest.mark.thread_unsafe("Reuses the same emplikeAFT result object")
    def test_beta_vect(self):
        assert_almost_equal(
            self.res1.test_beta([3.5, -0.035], [0, 1]), self.res2.test_joint, decimal=4
        )

    @pytest.mark.slow
    def test_betaci(self):
        ci = self.res1.ci_beta(1, -0.06, 0)
        ll = ci[0]
        ul = ci[1]
        ll_pval = self.res1.test_beta([ll], [1])[1]
        ul_pval = self.res1.test_beta([ul], [1])[1]
        assert_almost_equal(ul_pval, 0.050000, decimal=4)
        assert_almost_equal(ll_pval, 0.05000, decimal=4)
