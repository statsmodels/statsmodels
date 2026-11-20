"""
Tests for the deprecated statsmodels.tsa.mlemodel.TSMLEModel

TSMLEModel will be removed after statsmodels 0.16. It is a base class, so
it is tested through a conditional Gaussian AR(1) subclass whose maximum
likelihood estimates have a closed form: OLS of y_t on (1, y_{t-1}) for the
mean parameters and RSS / (nobs - 1) for the variance.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.mlemodel import TSMLEModel

DEPRECATED = "TSMLEModel is deprecated"

try:
    import numdifftools  # noqa: F401

    HAS_NUMDIFFTOOLS = True
except ImportError:
    HAS_NUMDIFFTOOLS = False


class GaussianAR1(TSMLEModel):
    """
    y_t = c + phi * y_{t-1} + e_t, e_t ~ N(0, sigma2), params (c, phi, sigma2)

    The log-likelihood conditions on the first observation.
    """

    def _resid_and_design(self, params):
        y = self.endog
        x = np.column_stack([np.ones(y.shape[0] - 1), y[:-1]])
        return y[1:] - x @ params[:2], x

    def loglike(self, params):
        e, _ = self._resid_and_design(params)
        s2 = params[2]
        return -0.5 * e.shape[0] * np.log(2 * np.pi * s2) - e @ e / (2 * s2)


class GaussianAR1Analytic(GaussianAR1):
    """
    Adds the analytic score and Hessian

    TSMLEModel's own score and hessian need numdifftools, which statsmodels
    does not depend on.
    """

    _start_params = np.array([0.0, 0.0, 1.0])

    def score(self, params):
        e, x = self._resid_and_design(params)
        s2 = params[2]
        m = e.shape[0]
        return np.r_[x.T @ e / s2, -m / (2 * s2) + e @ e / (2 * s2**2)]

    def hessian(self, params):
        e, x = self._resid_and_design(params)
        s2 = params[2]
        m = e.shape[0]
        hess = np.empty((3, 3))
        hess[:2, :2] = -x.T @ x / s2
        hess[:2, 2] = hess[2, :2] = -x.T @ e / s2**2
        hess[2, 2] = m / (2 * s2**2) - e @ e / s2**3
        return hess


def make(cls, endog):
    with pytest.warns(FutureWarning, match=DEPRECATED):
        return cls(endog)


@pytest.fixture
def endog():
    rng = np.random.default_rng(20261119)
    eps = 1.5 * rng.standard_normal(600)
    y = np.zeros(600)
    for t in range(1, 600):
        y[t] = 0.5 + 0.6 * y[t - 1] + eps[t]
    return y[100:]


@pytest.fixture
def closed_form(endog):
    """Closed-form MLE, its standard errors and the maximised log-likelihood"""
    m = endog.shape[0] - 1
    ols = OLS(endog[1:], np.column_stack([np.ones(m), endog[:-1]])).fit()
    sigma2 = ols.ssr / m
    params = np.r_[ols.params, sigma2]
    # OLS scales its covariance by RSS / (m - 2); the MLE uses RSS / m
    bse = np.r_[ols.bse * np.sqrt((m - 2) / m), np.sqrt(2 * sigma2**2 / m)]
    llf = -0.5 * m * (np.log(2 * np.pi * sigma2) + 1)
    return params, bse, llf


@pytest.mark.parametrize("start_params", [None, [0.2, 0.2, 2.0]])
def test_fit_matches_closed_form(endog, closed_form, start_params):
    params, bse, llf = closed_form
    mod = make(GaussianAR1Analytic, endog)
    res = mod.fit(start_params=start_params, method="newton")

    assert_allclose(res.params, params, rtol=1e-8)
    assert_allclose(res.bse, bse, rtol=1e-8)
    assert_allclose(res.llf, llf, rtol=1e-10)
    # start_params=None falls back to the model's _start_params
    start = mod._start_params if start_params is None else start_params
    assert_allclose(res.mle_settings["start_params"], start, rtol=1e-12)
    assert res.mle_settings["optimizer"] == "newton"
    assert res.mle_settings["maxiter"] == 5000
    assert_allclose(res.mle_settings["tol"], 1e-8, rtol=1e-12)


def test_fit_forwards_maxiter_and_tol(endog, closed_form):
    params, _, _ = closed_form
    mod = make(GaussianAR1Analytic, endog)
    res = mod.fit(maxiter=50, method="newton", tol=1e-10)

    assert_allclose(res.params, params, rtol=1e-8)
    assert res.mle_settings["maxiter"] == 50
    assert_allclose(res.mle_settings["tol"], 1e-10, rtol=1e-12)


@pytest.mark.filterwarnings("error:Keyword arguments have been passed:FutureWarning")
@pytest.mark.xfail(
    strict=True,
    raises=FutureWarning,
    reason="TSMLEModel.fit always passes tol to the optimizer, and only "
    'method="newton" accepts it',
)
def test_fit_other_optimizer(endog, closed_form):
    params, _, _ = closed_form
    res = make(GaussianAR1Analytic, endog).fit(method="nm")
    # Nelder-Mead stops at an absolute xtol of 1e-4, which TSMLEModel.fit
    # cannot change
    assert_allclose(res.params, params, rtol=1e-3)


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason='TSMLEModel.fit defaults to method="fmin", which '
    "LikelihoodModel.fit does not accept",
)
def test_fit_default_method(endog, closed_form):
    params, _, _ = closed_form
    res = make(GaussianAR1Analytic, endog).fit()
    assert_allclose(res.params, params, rtol=1e-3)


@pytest.mark.parametrize(
    "container",
    ["list", "tuple", "series", "frame", "column"],
)
def test_endog_array_like(endog, closed_form, container):
    params, _, llf = closed_form
    index = pd.date_range("2000-01-01", periods=endog.shape[0], freq="D")
    data = {
        "list": endog.tolist(),
        "tuple": tuple(endog),
        "series": pd.Series(endog, index=index),
        "frame": pd.DataFrame({"y": endog}, index=index),
        "column": endog[:, None],
    }[container]
    mod = make(GaussianAR1, data)
    assert_allclose(mod.loglike(params), llf, rtol=1e-12)


@pytest.mark.parametrize("method", ["loglike", "geterrors"])
def test_base_methods_not_implemented(endog, closed_form, method):
    mod = make(TSMLEModel, endog)
    with pytest.raises(NotImplementedError):
        getattr(mod, method)(closed_form[0])


@pytest.mark.skipif(HAS_NUMDIFFTOOLS, reason="checks the behavior without numdifftools")
@pytest.mark.xfail(
    strict=True,
    raises=NameError,
    reason="TSMLEModel.score and hessian use numdifftools, which statsmodels "
    "does not depend on; without it the name ndt is undefined",
)
@pytest.mark.parametrize("method", ["score", "hessian"])
def test_numerical_derivatives(endog, method):
    # Away from the maximum, so that the score is not zero
    params = np.array([0.4, 0.5, 2.0])
    analytic = make(GaussianAR1Analytic, endog)
    numerical = make(GaussianAR1, endog)
    expected = getattr(analytic, method)(params)
    assert_allclose(getattr(numerical, method)(params), expected, rtol=1e-6)
