import numpy as np
from numpy.testing import assert_allclose
import pytest

from statsmodels.tools._lstsq import QRSVD, default_rcond, numerical_rank

EPS = np.finfo(float).eps


@pytest.fixture(scope="module")
def full_rank():
    rng = np.random.default_rng(20261108)
    x = rng.standard_normal((200, 5))
    y = x @ np.arange(1.0, 6.0) + rng.standard_normal(200)
    return x, y


def duplicated_constant(nobs, rng):
    x = np.column_stack([np.ones(nobs), rng.standard_normal((nobs, 3))])
    x = np.column_stack([x, x[:, 0]])
    y = x[:, :4] @ np.array([1.0, 2.0, -1.0, 0.5]) + rng.standard_normal(nobs)
    return x, y


def test_default_rcond():
    assert default_rcond(100, 5) == 100 * EPS
    assert default_rcond(3, 5) == 5 * EPS


@pytest.mark.parametrize("shape", [(50, 4), (1000, 6), (4, 4), (3, 6)])
@pytest.mark.parametrize("deficient", [False, True])
def test_numerical_rank_matches_numpy(shape, deficient):
    rng = np.random.default_rng(1234)
    x = rng.standard_normal(shape)
    if deficient:
        x[:, -1] = x[:, 0] + x[:, 1]
    s = np.linalg.svd(x, compute_uv=False)
    assert numerical_rank(s, shape) == np.linalg.matrix_rank(x)


def test_numerical_rank_explicit_rcond_and_empty():
    s = [10.0, 1.0, 1e-3]
    assert numerical_rank(s, (10, 3)) == 3
    assert numerical_rank(s, (10, 3), rcond=1e-2) == 2
    assert numerical_rank([], (0, 0)) == 0
    assert numerical_rank(np.zeros(3), (10, 3)) == 0


def test_qrsvd_full_rank(full_rank):
    x, y = full_rank
    fact = QRSVD(x)
    assert fact.shape == x.shape
    assert fact.rank() == 5
    assert_allclose(fact.singular_values, np.linalg.svd(x, compute_uv=False))
    params = fact.solve(y)
    assert_allclose(params, np.linalg.lstsq(x, y, rcond=None)[0])
    assert_allclose(fact.pinv(), np.linalg.pinv(x), rtol=1e-10, atol=1e-12)
    assert_allclose(fact.cov_unscaled(), np.linalg.inv(x.T @ x))
    # effects are the coordinates of the fitted values in an orthonormal basis
    effects = fact.effects(y)
    assert effects.shape == (5,)
    assert_allclose(effects @ effects, np.sum((x @ params) ** 2))


def test_qrsvd_two_dimensional_data(full_rank):
    x, y = full_rank
    rng = np.random.default_rng(99)
    ys = np.column_stack([y, rng.standard_normal(y.shape[0])])
    fact = QRSVD(x)
    params = fact.solve(ys)
    assert params.shape == (5, 2)
    for j in range(2):
        assert_allclose(params[:, j], fact.solve(ys[:, j]))
    assert fact.effects(ys).shape == (5, 2)


@pytest.mark.parametrize("nobs", [50, 1000, 10000])
def test_qrsvd_rank_deficient(nobs):
    rng = np.random.default_rng(20261108 + nobs)
    x, y = duplicated_constant(nobs, rng)
    fact = QRSVD(x)
    assert fact.rank() == 4
    params = fact.solve(y)
    # the minimum-norm solution splits the constant between its two copies
    assert_allclose(params[0], params[-1])
    assert_allclose(params, np.linalg.lstsq(x, y, rcond=None)[0], atol=1e-8)
    # Moore-Penrose conditions and the covariance as pinv @ pinv.T
    pinv = fact.pinv()
    assert_allclose(x @ pinv @ x, x, atol=1e-9)
    assert_allclose(pinv @ x @ pinv, pinv, atol=1e-9)
    assert_allclose(pinv @ y, params, atol=1e-9)
    assert_allclose(fact.cov_unscaled(), pinv @ pinv.T, atol=1e-10)


def test_qrsvd_more_columns_than_rows():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((3, 6))
    y = rng.standard_normal(3)
    fact = QRSVD(x)
    assert fact.rank() == 3
    params = fact.solve(y)
    assert params.shape == (6,)
    assert_allclose(x @ params, y, atol=1e-12)
    assert_allclose(params, np.linalg.lstsq(x, y, rcond=None)[0])
    assert fact.pinv().shape == (6, 3)
    assert fact.cov_unscaled().shape == (6, 6)


def test_qrsvd_explicit_rcond():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((100, 3))
    x[:, 2] *= 1e-4
    fact = QRSVD(x)
    assert fact.rank() == 3
    assert fact.rank(rcond=1e-2) == 2
    y = rng.standard_normal(100)
    # the cutoff has the meaning of the rcond argument of numpy.linalg.lstsq
    assert_allclose(
        fact.solve(y, rcond=1e-2), np.linalg.lstsq(x, y, rcond=1e-2)[0], atol=1e-10
    )
    assert_allclose(fact.pinv(rcond=1e-2), np.linalg.pinv(x, rcond=1e-2), atol=1e-10)


def test_qrsvd_ill_conditioned_covariance():
    # The covariance is computed from the factors, not from x'x. The latter
    # squares the condition number and loses about twice as many digits.
    rng = np.random.default_rng(2)
    nobs, k = 500, 5
    u, _ = np.linalg.qr(rng.standard_normal((nobs, k)))
    v, _ = np.linalg.qr(rng.standard_normal((k, k)))
    sv = np.geomspace(1.0, 1e-6, k)
    x = (u * sv) @ v.T
    expected = (v / sv**2) @ v.T
    cov = QRSVD(x).cov_unscaled()
    assert np.max(np.abs(cov - expected)) / np.max(np.abs(expected)) < 1e-8


def test_qrsvd_invalid_input():
    with pytest.raises(ValueError, match="2-dimensional"):
        QRSVD(np.ones(5))
