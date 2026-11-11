"""
Tests corresponding to sandbox.regression.predstd

wls_prediction_std's correctness against a manually-computed prediction
variance formula, for both OLS and WLS, is already covered by
statsmodels.regression.tests.test_predict. This file adds coverage for
atleast_2dcol (untested anywhere -- its own docstring says "not tested
because not used") and predstd's error-handling branches.
"""
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import pytest

from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.sandbox.regression.predstd import atleast_2dcol, wls_prediction_std
from statsmodels.tools.tools import add_constant


class TestAtleast2dcol:
    def test_1d_becomes_column(self):
        result = atleast_2dcol(np.array([1.0, 2.0, 3.0]))
        assert result.shape == (3, 1)
        assert_allclose(result[:, 0], [1.0, 2.0, 3.0])

    def test_0d_becomes_2d(self):
        result = atleast_2dcol(np.array(5.0))
        assert result.shape == (1, 1)
        assert_allclose(result, [[5.0]])

    def test_2d_input_raises(self):
        # documented scope is converting *from* 1d or 0d; already-2d (or
        # higher) input is rejected rather than passed through
        with pytest.raises(ValueError, match="too many dimensions"):
            atleast_2dcol(np.ones((2, 2)))

    def test_accepts_plain_list(self):
        result = atleast_2dcol([1, 2, 3])
        assert result.shape == (3, 1)


class TestWlsPredictionStd:
    @classmethod
    def setup_class(cls):
        rs = np.random.RandomState(0)
        n = 50
        x = rs.normal(size=(n, 2))
        cls.exog = add_constant(x)
        beta = np.array([1.0, 2.0, -1.0])
        cls.endog = cls.exog @ beta + rs.normal(size=n)
        cls.res_ols = OLS(cls.endog, cls.exog).fit()
        weights = rs.uniform(0.5, 2.0, size=n)
        cls.res_wls = WLS(cls.endog, cls.exog, weights=weights).fit()

    def test_default_exog_matches_nobs(self):
        predstd, lo, hi = wls_prediction_std(self.res_ols)
        assert predstd.shape == (50,)
        assert_array_less(lo, self.res_ols.fittedvalues)
        assert_array_less(self.res_ols.fittedvalues, hi)

    def test_explicit_exog_matches_default_subset(self):
        predstd_all, _, _ = wls_prediction_std(self.res_ols)
        predstd_sub, lo_sub, hi_sub = wls_prediction_std(
            self.res_ols, exog=self.exog[:5]
        )
        assert_allclose(predstd_sub, predstd_all[:5])

    def test_wls_default_exog_matches_nobs(self):
        predstd, lo, hi = wls_prediction_std(self.res_wls)
        assert predstd.shape == (50,)
        assert_array_less(lo, self.res_wls.fittedvalues)
        assert_array_less(self.res_wls.fittedvalues, hi)

    def test_narrower_alpha_gives_wider_interval(self):
        # alpha=0.10 is a lower confidence level than alpha=0.05, so its
        # interval should be narrower
        _, lo_05, hi_05 = wls_prediction_std(self.res_ols, alpha=0.05)
        _, lo_10, hi_10 = wls_prediction_std(self.res_ols, alpha=0.10)
        assert np.all((hi_10 - lo_10) < (hi_05 - lo_05))

    def test_wrong_exog_shape_raises(self):
        with pytest.raises(ValueError, match="wrong shape of exog"):
            wls_prediction_std(self.res_ols, exog=np.ones((5, 2)))

    def test_weights_shape_mismatch_raises(self):
        with pytest.raises(
            ValueError, match="weights and exog do not have matching shape"
        ):
            wls_prediction_std(
                self.res_ols, exog=self.exog[:5], weights=np.ones(3)
            )

    def test_explicit_weights_with_explicit_exog(self):
        weights = np.full(5, 3.0)
        predstd, lo, hi = wls_prediction_std(
            self.res_wls, exog=self.exog[:5], weights=weights
        )
        assert predstd.shape == (5,)
        # larger weight (smaller variance contribution) should give a
        # tighter interval than a smaller weight, all else equal
        predstd_small_w, _, _ = wls_prediction_std(
            self.res_wls, exog=self.exog[:5], weights=np.full(5, 0.5)
        )
        assert np.all(predstd < predstd_small_w)

    def test_scalar_weight_with_explicit_exog(self):
        predstd, lo, hi = wls_prediction_std(
            self.res_wls, exog=self.exog[:5], weights=2.0
        )
        assert predstd.shape == (5,)
