"""
Tests for statsmodels.tsa.varma_process.VarmaPoly and _Var

VarmaPoly is documented in tsa.rst but had no test coverage at all and no
references anywhere else in the repository. These tests independently
verify its stationarity/invertibility checks against textbook AR/MA root
conditions rather than merely asserting the methods run.

_Var is deprecated and will be removed after statsmodels 0.16. Until then
it is checked against statsmodels.tsa.api.VAR without a trend.

varfilter is checked against explicit loops over its documented formula.
"""

from statsmodels.compat.python import PYTHON_IMPL_WASM

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
import pytest

from statsmodels.tsa.api import VAR
from statsmodels.tsa.varma_process import VarmaPoly, _Var, varfilter

VAR_DEPRECATED = "_Var is deprecated"

ar23 = np.array(
    [
        [[1.0, 0.0], [0.0, 1.0]],
        [[-0.6, 0.0], [0.2, -0.6]],
        [[-0.1, 0.0], [0.1, -0.1]],
    ]
)
ma22 = np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.4, 0.0], [0.2, 0.3]]])


def test_init_defaults():
    vp = VarmaPoly(ar23, ma22)
    assert vp.nlags == 3
    assert vp.nvarall == 2
    assert vp.nvars == 2
    assert vp.hasexog is False
    assert_array_equal(vp.arm1, -ar23[1:])

    # Without an explicit ma, an identity is substituted and marked
    # independent.
    vp_no_ma = VarmaPoly(ar23)
    assert vp_no_ma.isindependent is True
    assert_array_equal(vp_no_ma.ma, np.eye(2)[None, ...])


def test_vstack_hstack_are_reshapes_of_the_source_array():
    vp = VarmaPoly(ar23, ma22)
    assert_array_equal(vp.vstack(), ar23.reshape(-1, 2))
    assert_array_equal(vp.hstack(name="ma"), ma22.transpose(1, 0, 2).reshape(2, -1))

    # An explicit `a` is used as-is; `name` is only consulted when `a` is
    # None, so an invalid `name` alongside an explicit `a` must not raise.
    assert_array_equal(vp.vstack(a=ma22, name="invalid"), ma22.reshape(-1, 2))


def test_vstack_hstack_stacksquare_invalid_name_raises():
    import pytest

    vp = VarmaPoly(ar23, ma22)
    with pytest.raises(ValueError, match="name"):
        vp.vstack(name="invalid")
    with pytest.raises(ValueError, match="name"):
        vp.hstack(name="invalid")
    with pytest.raises(ValueError, match="name"):
        vp.stacksquare(name="invalid")


def test_stacksquare_is_companion_form():
    # stacksquare builds a companion matrix: the first `nvars` columns hold
    # the full vertically-stacked lag polynomial, and the remaining columns
    # are an identity shifted by `nvars` (the sub-diagonal blocks that shift
    # lagged states forward by one period).
    vp = VarmaPoly(ar23, ma22)
    lenpk = ar23.shape[0] * vp.nvars
    sq = vp.stacksquare()

    assert sq.shape == (lenpk, lenpk)
    assert_array_equal(sq[:, : vp.nvars], vp.vstack())
    assert_array_equal(sq[:, vp.nvars :], np.eye(lenpk, k=vp.nvars)[:, vp.nvars :])


def test_getisstationary_matches_ar1_boundary():
    # y_t = phi * y_{t-1} + e_t is stationary iff |phi| < 1.
    for phi, expected in [(0.5, True), (0.9, True), (1.01, False), (1.5, False)]:
        ar = np.array([[[1.0]], [[-phi]]])
        vp = VarmaPoly(ar)
        assert bool(vp.getisstationary()) == expected
        assert_allclose(vp.areigenvalues, [phi])


def test_getisstationary_matches_independent_root_computation():
    # For a univariate AR(2), the reciprocals of the companion-matrix
    # eigenvalues used by getisstationary must equal the roots of the
    # characteristic polynomial 1 - phi_1 L - phi_2 L^2, computed via an
    # entirely independent method (numpy.roots).
    phi1, phi2 = 0.5, 0.3
    roots = np.roots([-phi2, -phi1, 1.0])
    ar = np.array([[[1.0]], [[-phi1]], [[-phi2]]])
    vp = VarmaPoly(ar)
    is_stationary = vp.getisstationary()

    assert is_stationary == bool((np.abs(roots) > 1).all())
    assert_allclose(np.sort(1 / vp.areigenvalues), np.sort(roots), rtol=1e-10)


def test_getisinvertible_matches_ma1_boundary():
    # y_t = e_t + theta * e_{t-1} is invertible iff |theta| < 1.
    for theta, expected in [(0.5, True), (0.9, True), (1.01, False), (1.5, False)]:
        ma = np.array([[[1.0]], [[theta]]])
        vp = VarmaPoly(np.array([[[1.0]]]), ma)
        assert bool(vp.getisinvertible()) == expected
        assert_allclose(vp.maeigenvalues, [theta])


def test_vstackarma_minus1_and_hstackarma_minus1_reshape_lags():
    # Both methods concatenate the ar and ma blocks excluding lag 0 (shape
    # (nlags-1 + malags-1, nvarall, nvars)) and reshape it to 2d.
    # vstackarma_minus1 flattens the lag and row axes together; verify
    # against a hand-built array following that documented layout.
    vp = VarmaPoly(ar23, ma22)

    expected_v = np.array(
        [
            [-0.6, 0.0],
            [0.2, -0.6],
            [-0.1, 0.0],
            [0.1, -0.1],
            [0.4, 0.0],
            [0.2, 0.3],
        ]
    )
    assert_array_equal(vp.vstackarma_minus1(), expected_v)

    # hstackarma_minus1 additionally transposes each (nvarall, nvars) block
    # before flattening (the "Kalman filter representation").
    expected_h = np.array(
        [
            [-0.6, 0.2],
            [0.0, -0.6],
            [-0.1, 0.1],
            [0.0, -0.1],
            [0.4, 0.2],
            [0.0, 0.3],
        ]
    )
    assert_array_equal(vp.hstackarma_minus1(), expected_h)


def test_reduceform_normalizes_lag_zero_to_identity():
    # reduceform left-multiplies every lag by inv(apoly[0]); check against
    # that definition directly for a non-trivial (non-identity) lag-zero
    # block, and confirm the reduced lag-zero block becomes the identity,
    # as required for a "reduced form" representation.
    vp = VarmaPoly(ar23, ma22)
    apoly = np.array([[[2.0, 0.0], [0.0, 1.0]], [[1.0, 2.0], [3.0, 4.0]]])

    reduced = vp.reduceform(apoly)

    a0inv = np.linalg.inv(apoly[0])
    expected = np.stack([a0inv @ apoly[0], a0inv @ apoly[1]])
    assert_allclose(reduced, expected)
    assert_allclose(reduced[0], np.eye(2))


def test_reduceform_errors():
    vp = VarmaPoly(ar23, ma22)
    with pytest.raises(ValueError, match="apoly needs to be 3d"):
        vp.reduceform(np.eye(2))


@pytest.mark.skipif(PYTHON_IMPL_WASM, reason="linalg error not raised on WASM")
def test_reduceform_linalg_error():
    vp = VarmaPoly(ar23, ma22)

    singular = np.array([[[0.0, 0.0], [0.0, 0.0]], [[1.0, 2.0], [3.0, 4.0]]])
    with pytest.raises(ValueError, match="matrix not invertible"):
        vp.reduceform(singular)


def _varfilter_loop(x, a):
    # The filter documented in varfilter, y[t] = a(L) x[t] for
    # t = nlags - 1, ..., nobs - 1, by explicit loops
    x = np.asarray(x, dtype=float)
    a = np.asarray(a, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    nobs, nvars = x.shape
    nlags = a.shape[0]
    if a.ndim == 3:
        # Case 3: y[t, i] = sum_p sum_j a[p, j, i] x[t - p, j]
        out = np.zeros((nobs - nlags + 1, a.shape[2]))
        for t in range(nlags - 1, nobs):
            for i in range(a.shape[2]):
                for p in range(nlags):
                    for j in range(nvars):
                        out[t - nlags + 1, i] += a[p, j, i] * x[t - p, j]
        return out
    # Case 2: y[t, j] = sum_p a[p, j] x[t - p, j]. In case 1, a 1-d or
    # (nlags, 1) a is the same lag polynomial for every variable.
    coef = np.broadcast_to(a.reshape(nlags, -1), (nlags, nvars))
    out = np.zeros((nobs - nlags + 1, nvars))
    for t in range(nlags - 1, nobs):
        for j in range(nvars):
            for p in range(nlags):
                out[t - nlags + 1, j] += coef[p, j] * x[t - p, j]
    return out


def test_varfilter_3d_example():
    # y[t] = x[t] + 0.5 * x[t - 1, 0] e_0 for t = 1, ..., 4
    x = np.arange(10.0).reshape(5, 2)
    a = [[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.0], [0.0, 0.0]]]
    expected = x[1:] + np.c_[0.5 * x[:-1, 0], np.zeros(4)]
    assert_allclose(varfilter(x, a), expected, rtol=1e-12)


@pytest.mark.parametrize("nvars", [1, 2, 3])
@pytest.mark.parametrize("nlags", [1, 2, 3])
@pytest.mark.parametrize("case", ["1d", "2d_column", "2d", "3d", "3d_npoly"])
def test_varfilter(case, nlags, nvars):
    # case "2d" with nlags == 1 and nvars > 1: a 2-d a of shape (1, nvars)
    # used to be taken as one lag polynomial (case 1) and gave one column
    rng = np.random.default_rng(20261121)
    nobs = 20
    x = rng.standard_normal((nobs, nvars))
    shape = {
        "1d": (nlags,),
        "2d_column": (nlags, 1),
        "2d": (nlags, nvars),
        "3d": (nlags, nvars, nvars),
        "3d_npoly": (nlags, nvars, nvars + 1),
    }[case]
    a = rng.standard_normal(shape)
    expected = _varfilter_loop(x, a)
    y = varfilter(x, a)
    assert y.shape == expected.shape
    assert_allclose(y, expected, rtol=1e-10)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_varfilter_lists_and_1d_x(ndim):
    # x and a are array_like, and a 1-d x is a single variable
    rng = np.random.default_rng(20261122)
    x = rng.standard_normal(15)
    a = rng.standard_normal((3,) + (1,) * (ndim - 1))
    expected = _varfilter_loop(x, a)
    y = varfilter(x.tolist(), a.tolist())
    assert y.shape == (13, 1)
    assert_allclose(y, expected, rtol=1e-10)


@pytest.fixture
def var_data():
    # Stationary VAR(2): y_t = A1 y_{t-1} + A2 y_{t-2} + e_t
    rng = np.random.default_rng(20261119)
    a1 = np.array([[0.5, 0.1], [0.2, 0.4]])
    a2 = np.array([[-0.2, 0.0], [0.1, -0.1]])
    eps = rng.standard_normal((400, 2))
    y = np.zeros((400, 2))
    for t in range(2, 400):
        y[t] = a1 @ y[t - 1] + a2 @ y[t - 2] + eps[t]
    return y[100:]


def make_var(y):
    with pytest.warns(FutureWarning, match=VAR_DEPRECATED):
        return _Var(y)


@pytest.mark.parametrize("as_frame", [False, True])
@pytest.mark.parametrize("nlags", [1, 2, 3])
def test_var_fit(var_data, nlags, as_frame):
    nobs, nvars = var_data.shape
    y = var_data
    if as_frame:
        index = pd.date_range("2000-01-01", periods=nobs, freq="D")
        y = pd.DataFrame(var_data, index=index, columns=["a", "b"])
    v = make_var(y)
    assert v.fit(nlags) is None
    res = VAR(var_data).fit(nlags, trend="n")

    assert (v.nobs, v.nvars, v.nlags) == (nobs, nvars, nlags)
    # arlhs[i] is the transpose of the coefficient matrix of lag i + 1
    coefs_t = res.coefs.transpose(0, 2, 1)
    assert_allclose(v.arlhs, coefs_t, rtol=1e-10)
    assert_array_equal(v.arhat[0], np.eye(nvars))
    assert_allclose(v.arhat[1:], -coefs_t, rtol=1e-10)
    assert_allclose(v.rss, (np.asarray(res.resid) ** 2).sum(0), rtol=1e-10)
    assert v.xredrank == nlags * nvars
    # The regression of y_t on (y_{t-1}, ..., y_{t-nlags})
    assert_allclose(v.yred, var_data[nlags:], rtol=1e-12)
    lags = [var_data[nlags - i : nobs - i] for i in range(1, nlags + 1)]
    assert_allclose(v.xred, np.hstack(lags), rtol=1e-12)


@pytest.mark.parametrize("horiz", [None, 5])
@pytest.mark.parametrize("nlags", [1, 2, 3])
def test_var_forecast(var_data, nlags, horiz):
    nobs, nvars = var_data.shape
    v = make_var(var_data)
    v.fit(nlags)
    res = VAR(var_data).fit(nlags, trend="n")
    if horiz is None:
        fcast, horiz = v.forecast(), 1
    else:
        fcast = v.forecast(horiz)

    assert fcast.shape == (nobs + horiz, nvars)
    assert_allclose(fcast[:nobs], var_data, rtol=1e-12)
    expected = res.forecast(var_data[-nlags:], horiz)
    assert_allclose(fcast[nobs:], expected, rtol=1e-10)


def test_var_forecast_with_errors(var_data):
    nobs, nvars = var_data.shape
    v = make_var(var_data)
    v.fit(2)
    res = VAR(var_data).fit(2, trend="n")
    u = np.random.default_rng(20261120).standard_normal((6, nvars))
    # horiz is ignored when u is given; the horizon is len(u)
    fcast = v.forecast(horiz=1, u=u)

    assert fcast.shape == (nobs + 6, nvars)
    assert_allclose(fcast[:nobs], var_data, rtol=1e-12)
    # The errors enter through the MA representation of the VAR
    psi = res.ma_rep(5)
    shocks = [sum(psi[k] @ u[h - k] for k in range(h + 1)) for h in range(6)]
    expected = res.forecast(var_data[-2:], 6) + np.array(shocks)
    assert_allclose(fcast[nobs:], expected, rtol=1e-10)


@pytest.mark.parametrize("as_frame", [False, True])
@pytest.mark.parametrize("nlags", [1, 2, 3])
def test_var_predict(var_data, nlags, as_frame):
    nobs, nvars = var_data.shape
    y = var_data
    if as_frame:
        index = pd.date_range("2000-01-01", periods=nobs, freq="D")
        y = pd.DataFrame(var_data, index=index, columns=["a", "b"])
    v = make_var(y)
    v.fit(nlags)
    res = VAR(var_data).fit(nlags, trend="n")
    yhat = v.predict()
    assert v.predict() is yhat
    assert v.yhat is yhat
    assert isinstance(yhat, np.ndarray)
    assert yhat.shape == (nobs - nlags, nvars)
    assert_allclose(yhat, res.fittedvalues, rtol=1e-10)


def test_var_predict_after_refit(var_data):
    # predict caches its result, and a new fit must replace it
    v = make_var(var_data)
    v.fit(1)
    v.predict()
    v.fit(2)
    res = VAR(var_data).fit(2, trend="n")
    assert_allclose(v.predict(), res.fittedvalues, rtol=1e-10)


@pytest.mark.parametrize("nlags", [1, 2, 3])
def test_var_covmat(var_data, nlags):
    nvars = var_data.shape[1]
    v = make_var(var_data)
    v.fit(nlags)
    res = VAR(var_data).fit(nlags, trend="n")
    assert v.covmat() is None
    k = nlags * nvars
    assert v.paramcov.shape == (k, k, nvars)
    # VAR orders its parameters regressor by regressor, equation within
    cov = np.asarray(res.cov_params())
    for eq in range(nvars):
        assert_allclose(v.paramcov[:, :, eq], cov[eq::nvars, eq::nvars], rtol=1e-8)
