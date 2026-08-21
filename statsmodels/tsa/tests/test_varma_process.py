"""
Tests for statsmodels.tsa.varma_process.VarmaPoly

VarmaPoly is documented in tsa.rst but had no test coverage at all and no
references anywhere else in the repository. These tests independently
verify its stationarity/invertibility checks against textbook AR/MA root
conditions rather than merely asserting the methods run.
"""
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

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
    assert_array_equal(sq[:, vp.nvars:], np.eye(lenpk, k=vp.nvars)[:, vp.nvars:])


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
    assert_allclose(
        np.sort(1 / vp.areigenvalues), np.sort(roots), rtol=1e-10
    )


def test_getisinvertible_matches_ma1_boundary():
    # y_t = e_t + theta * e_{t-1} is invertible iff |theta| < 1.
    for theta, expected in [(0.5, True), (0.9, True), (1.01, False), (1.5, False)]:
        ma = np.array([[[1.0]], [[theta]]])
        vp = VarmaPoly(np.array([[[1.0]]]), ma)
        assert bool(vp.getisinvertible()) == expected
        assert_allclose(vp.maeigenvalues, [theta])
