"""
Created on Fri Nov 04 10:51:39 2011

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pytest

from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.sandbox.nonparametric import kernels, smoothers


class CheckSmoother:

    def test_predict(self):
        assert_almost_equal(
            self.res_ps.predict(self.x), self.res2.fittedvalues, decimal=13
        )
        assert_almost_equal(
            self.res_ps.predict(self.x[:10]), self.res2.fittedvalues[:10], decimal=13
        )

    def test_coef(self):
        # TODO: check dim of coef
        assert_almost_equal(self.res_ps.coef.ravel(), self.res2.params, decimal=14)

    def test_df(self):
        # TODO: make into attributes
        assert_equal(self.res_ps.df_model(), self.res2.df_model + 1)  # with const
        assert_equal(self.res_ps.df_fit(), self.res2.df_model + 1)  # alias
        assert_equal(self.res_ps.df_resid(), self.res2.df_resid)


class BasePolySmoother:

    @classmethod
    def setup_class(cls):
        # DGP: simple polynomial
        order = 3
        sigma_noise = 0.5
        nobs = 100
        lb, ub = -1, 2
        cls.x = x = np.linspace(lb, ub, nobs)
        cls.exog = exog = x[:, None] ** np.arange(order + 1)
        y_true = exog.sum(1)
        rs = np.random.RandomState(987567)
        cls.y = y_true + sigma_noise * rs.randn(nobs)


class TestPolySmoother1(BasePolySmoother, CheckSmoother):

    @classmethod
    def setup_class(cls):
        super().setup_class()  # initialize DGP

        y, x, exog = cls.y, cls.x, cls.exog

        # use order = 2 in regression
        pmod = smoothers.PolySmoother(2, x)
        pmod.fit(y)  # no return

        cls.res_ps = pmod
        cls.res2 = OLS(y, exog[:, : 2 + 1]).fit()


class TestPolySmoother2(BasePolySmoother, CheckSmoother):

    @classmethod
    def setup_class(cls):
        super().setup_class()  # initialize DGP

        y, x, exog = cls.y, cls.x, cls.exog

        # use order = 3 in regression
        pmod = smoothers.PolySmoother(3, x)
        # pmod.fit(y)  # no return
        pmod.smooth(y)  # no return, use alias for fit

        cls.res_ps = pmod
        cls.res2 = OLS(y, exog[:, : 3 + 1]).fit()


class TestPolySmoother3(BasePolySmoother, CheckSmoother):

    @classmethod
    def setup_class(cls):
        super().setup_class()  # initialize DGP

        y, x, exog = cls.y, cls.x, cls.exog
        nobs = y.shape[0]
        weights = np.ones(nobs)
        weights[: nobs // 3] = 0.1
        weights[-nobs // 5 :] = 2

        # use order = 2 in regression
        pmod = smoothers.PolySmoother(2, x)
        pmod.fit(y, weights=weights)  # no return

        cls.res_ps = pmod
        cls.res2 = WLS(y, exog[:, : 2 + 1], weights=weights).fit()


def test_polysmoother_gram_is_noop():
    ps = smoothers.PolySmoother(2, np.linspace(-1, 1, 5))
    assert ps.gram() is None
    assert ps.gram(3) is None


def test_polysmoother_fit_requires_x_if_never_set():
    ps = smoothers.PolySmoother(2)
    with pytest.raises(ValueError, match="x needed to fit PolySmoother"):
        ps.fit(np.arange(5.0))


def test_polysmoother_2d_x_in_init_uses_first_row(capsys):
    x = np.linspace(-1, 1, 5)
    x2d = np.tile(x, (2, 1))
    ps_2d = smoothers.PolySmoother(2, x2d)
    ps_1d = smoothers.PolySmoother(2, x)
    assert_allclose(ps_2d.X, ps_1d.X)
    captured = capsys.readouterr()
    assert "Warning" in captured.out


def test_polysmoother_predict_with_no_argument_uses_stored_x():
    rs = np.random.RandomState(0)
    x = np.linspace(-1, 1, 20)
    y = 1 + 2 * x + x**2 + 0.01 * rs.randn(20)
    ps = smoothers.PolySmoother(2, x)
    ps.fit(y)
    assert_allclose(ps.predict(), ps.predict(x))


def test_polysmoother_call_is_alias_for_predict():
    x = np.linspace(-1, 1, 20)
    y = 1 + 2 * x + x**2
    ps = smoothers.PolySmoother(2, x)
    ps.fit(y)
    assert_allclose(ps(), ps.predict())
    assert_allclose(ps(x), ps.predict(x))


def test_polysmoother_predict_2d_x_uses_first_column(capsys):
    x = np.linspace(-1, 1, 20)
    y = 1 + 2 * x + x**2
    ps = smoothers.PolySmoother(2, x)
    ps.fit(y)
    x2d = np.tile(x, (2, 1)).T  # shape (20, 2); column 0 is x
    result = ps.predict(x2d)
    assert_allclose(result, ps.predict(x))
    captured = capsys.readouterr()
    assert "Warning" in captured.out


def test_polysmoother_fit_weights_none_matches_all_nan():
    # `if weights is None or np.isnan(weights).all(): weights = 1` treats
    # an all-nan weights array the same as no weights at all
    x = np.linspace(-1, 1, 20)
    y = 1 + 2 * x + x**2
    ps_none = smoothers.PolySmoother(2, x)
    ps_none.fit(y, weights=None)
    ps_nan = smoothers.PolySmoother(2, x)
    ps_nan.fit(y, weights=np.full_like(y, np.nan))
    assert_allclose(ps_none.coef, ps_nan.coef)


@pytest.mark.xfail(
    reason=(
        "BUG: PolySmoother.fit()'s 2d-x handling only prints a warning "
        "but never actually reduces x to 1d (the fix is present but "
        "commented out: `# x=x[0,:] # TODO: check orientation, row or "
        "col`), unlike the equivalent branches in __init__ and predict() "
        "which do perform the reduction (x = x[0, :] / x[:, 0] "
        "respectively). Calling fit() with 2d x therefore builds a "
        "malformed higher-dimensional self.X and raises LinAlgError from "
        "np.linalg.lstsq instead of fitting against one column of x."
    ),
    raises=np.linalg.LinAlgError,
    strict=True,
)
def test_polysmoother_fit_2d_x_is_broken():
    x = np.linspace(-1, 1, 20)
    y = 1 + 2 * x + x**2
    x2d = np.tile(x, (2, 1)).T
    ps = smoothers.PolySmoother(2)
    ps.fit(y, x=x2d)


class TestKernelSmoother:
    @classmethod
    def setup_class(cls):
        rs = np.random.RandomState(12345)
        cls.x = np.linspace(-2, 2, 200)
        cls.y = cls.x**2 + rs.normal(scale=0.2, size=200)
        cls.ks = smoothers.KernelSmoother(cls.x, cls.y)

    def test_default_kernel_is_gaussian(self):
        assert isinstance(self.ks.Kernel, kernels.Gaussian)

    def test_fit_is_a_noop(self):
        assert self.ks.fit() is None

    def test_predict_scalar_matches_kernel_smooth(self):
        result = self.ks.predict(0.5)
        expected = self.ks.Kernel.smooth(self.x, self.y, 0.5)
        assert_allclose(result, expected)

    def test_predict_array_matches_elementwise_scalar_predict(self):
        xg = np.array([-1.0, 0.0, 0.5, 1.0])
        result = self.ks.predict(xg)
        expected = np.array([self.ks.predict(xx) for xx in xg])
        assert_allclose(result, expected)

    def test_call_matches_predict(self):
        xg = np.array([-1.0, 0.0, 0.5, 1.0])
        assert_allclose(self.ks(xg), self.ks.predict(xg))

    def test_predicted_curve_recovers_quadratic_shape(self):
        # smoothed values near the parabola's minimum should be smaller
        # than smoothed values further out
        assert self.ks.predict(0.0) < self.ks.predict(1.5)

    def test_std_is_sqrt_of_var(self):
        xg = np.array([-1.0, 0.0, 1.0])
        var = self.ks.var(xg)
        std = self.ks.std(xg)
        assert_allclose(std, np.sqrt(var))
        assert np.all(var >= 0)

    def test_conf_with_array(self):
        xg = np.array([-1.0, 0.0, 1.0])
        result = self.ks.conf(xg)
        assert result.shape == (3, 3)

    def test_conf_with_int_subsamples_sorted_x(self):
        confx, conffit = self.ks.conf(20)
        assert_allclose(confx, np.sort(self.x)[::20])
        assert conffit.shape == (len(confx), 3)
        # subsampled call should give the same fit as calling conf directly
        # on those points
        assert_allclose(conffit, self.ks.conf(confx))

    @pytest.mark.xfail(
        reason=(
            "BUG (in sandbox.nonparametric.kernels, not smoothers): "
            "KernelSmoother documents accepting any Kernel object via its "
            "`Kernel` parameter, but non-Gaussian kernels (e.g., "
            "Epanechnikov, Uniform) raise "
            "`TypeError: unsupported operand type(s) for -: 'tuple' and "
            "'float'` inside Kernel.smooth() even with default "
            "construction (no unusual arguments). Only the default "
            "Gaussian kernel currently works. See "
            "sandbox.nonparametric.tests.test_kernels for the root-cause "
            "investigation."
        ),
        raises=TypeError,
        strict=True,
    )
    def test_predict_with_non_gaussian_kernel(self):
        ks = smoothers.KernelSmoother(self.x, self.y, Kernel=kernels.Epanechnikov())
        ks.predict(0.5)
