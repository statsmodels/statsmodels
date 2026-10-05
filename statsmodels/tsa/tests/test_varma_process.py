"""
Tests for statsmodels.tsa.varma_process.VarmaPoly

VarmaPoly is documented in tsa.rst but had no test coverage at all and no
references anywhere else in the repository. These tests independently
verify its stationarity/invertibility checks against textbook AR/MA root
conditions rather than merely asserting the methods run.
"""

from statsmodels.compat.python import PYTHON_IMPL_WASM

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from statsmodels.tsa.varma_process import VarmaPoly

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
