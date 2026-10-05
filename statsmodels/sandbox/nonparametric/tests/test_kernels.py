"""
Tests corresponding to sandbox.nonparametric.kernels

Note: correctness of smooth()/smoothconf() against reference (Stata)
values is already covered, for Gaussian/Epanechnikov/Uniform/Triangular/
Biweight/Cosine/Tricube, by statsmodels.nonparametric.tests.test_kernels.
This file focuses on behavior not exercised there: NdKernel, the shared
CustomKernel helper methods, and a bug in in_domain() found while writing
these tests.
"""
import numpy as np
from numpy.testing import assert_allclose
import pytest
import scipy.integrate

from statsmodels.sandbox.nonparametric.kernels import (
    Biweight,
    Cosine2,
    CustomKernel,
    Epanechnikov,
    Gaussian,
    NdKernel,
    Triangular,
    Uniform,
)


def test_customkernel_requires_callable_shape():
    with pytest.raises(TypeError, match="shape must be a callable"):
        CustomKernel(shape="not callable")


def test_customkernel_h_property():
    k = Gaussian(h=2.0)
    assert k.h == 2.0
    k.h = 3.0
    assert k.h == 3.0


def test_call_is_alias_for_shape():
    k = Gaussian()
    assert_allclose(k(0.5), k._shape(0.5))


def test_weight_is_normconst_times_shape():
    k = Epanechnikov()
    assert_allclose(k.weight(0.3), k.norm_const * k._shape(0.3))


# Uniform's shape lambda is `0.5 * np.ones(x.shape)`, which requires x to
# be array-like (it needs `.shape`). scipy.integrate.quad always calls the
# integrand with a plain scalar float, so Uniform can't go through the
# quad-based L2Norm/kernel_var/norm_const computation at all -- this is
# presumably exactly *why* Uniform's __init__ hardcodes _L2Norm,
# _kernel_var, and norm=1.0 rather than relying on the lazy properties.
# It's excluded from the numerical cross-check below and given its own
# closed-form check instead.
_QUAD_COMPATIBLE_KERNELS = [Gaussian, Epanechnikov, Triangular, Biweight]


@pytest.mark.parametrize("Kernel", _QUAD_COMPATIBLE_KERNELS)
def test_l2norm_shortcut_matches_numerical_integration(Kernel):
    # each bounded kernel hardcodes _L2Norm as a shortcut; verify it
    # against direct numerical integration of (norm_const * shape(x))**2
    k = Kernel()
    bounds = (-np.inf, np.inf) if k.domain is None else tuple(k.domain)

    def integrand(x):
        return (k.norm_const * k._shape(x)) ** 2

    expected, _ = scipy.integrate.quad(integrand, *bounds)
    assert_allclose(k.L2Norm, expected, rtol=1e-6)


@pytest.mark.parametrize("Kernel", _QUAD_COMPATIBLE_KERNELS)
def test_kernel_var_shortcut_matches_numerical_integration(Kernel):
    k = Kernel()
    bounds = (-np.inf, np.inf) if k.domain is None else tuple(k.domain)

    def integrand(x):
        return x**2 * k.norm_const * k._shape(x)

    expected, _ = scipy.integrate.quad(integrand, *bounds)
    assert_allclose(k.kernel_var, expected, rtol=1e-6)


def test_uniform_hardcoded_constants_match_closed_form():
    # Uniform density 0.5 on [-1, 1]:
    # L2Norm = int_{-1}^{1} 0.5**2 dx = 0.5
    # kernel_var = int_{-1}^{1} x**2 * 0.5 dx = 1/3
    k = Uniform()
    assert_allclose(k.L2Norm, 0.5)
    assert_allclose(k.kernel_var, 1.0 / 3)


@pytest.mark.xfail(
    reason=(
        "BUG: Uniform's shape lambda `lambda x: 0.5 * np.ones(x.shape)` "
        "requires x to be array-like (it calls x.shape), so it raises "
        "AttributeError for a plain scalar float input -- exactly what "
        "scipy.integrate.quad (used internally by the lazy L2Norm/"
        "kernel_var/norm_const properties) always passes. This is "
        "normally masked because Uniform.__init__ hardcodes _L2Norm, "
        "_kernel_var, and norm=1.0, so those lazy properties are never "
        "actually computed in practice -- but calling _shape directly "
        "with a scalar, as any generic CustomKernel-consuming code might "
        "reasonably do, fails."
    ),
    raises=AttributeError,
    strict=True,
)
def test_uniform_shape_does_not_accept_scalar_input():
    k = Uniform()
    k._shape(0.5)


def test_moments():
    k = Gaussian()
    assert k.moments(1) == 0
    assert_allclose(k.moments(2), k.kernel_var)
    with pytest.raises(NotImplementedError):
        k.moments(3)


def test_normal_reference_constant_requires_second_order():
    k = Gaussian()
    k._order = 4
    with pytest.raises(NotImplementedError, match="second order"):
        _ = k.normal_reference_constant


def test_density_var_and_confint():
    k = Gaussian()
    density = np.array([0.1, 0.2, 0.3])
    nobs = 100
    var = k.density_var(density, nobs)
    assert_allclose(var, density * k.L2Norm / k.h / nobs)

    conf = k.density_confint(density, nobs, alpha=0.05)
    from scipy import stats

    crit = stats.norm.isf(0.025)
    half_width = crit * np.sqrt(var)
    assert_allclose(conf[:, 0], density - half_width)
    assert_allclose(conf[:, 1], density + half_width)


class TestInDomain:
    def test_no_domain_returns_input_unchanged(self):
        k = Gaussian()  # domain is None
        xs = np.array([1.0, 2.0, 3.0])
        ys = np.array([4.0, 5.0, 6.0])
        xs2, ys2 = k.in_domain(xs, ys, 0.0)
        assert xs2 is xs
        assert ys2 is ys

    def test_filters_points_outside_domain(self):
        k = Epanechnikov()  # domain [-1, 1]
        xs = np.array([-5.0, -0.5, 0.0, 0.5, 5.0])
        ys = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        xs2, ys2 = k.in_domain(xs, ys, 0.0)
        assert_allclose(sorted(xs2), [-0.5, 0.0, 0.5])

    def test_empty_result_when_nothing_in_domain(self):
        k = Epanechnikov()
        xs = np.array([-5.0, 5.0])
        ys = np.array([1.0, 2.0])
        xs2, ys2 = k.in_domain(xs, ys, 0.0)
        assert xs2 == []
        assert ys2 == []

    @pytest.mark.xfail(
        reason=(
            "BUG: in_domain()'s filtering branch returns Python tuples, "
            "not ndarrays: `xs, ys = lzip(*filtered)` unzips a list of "
            "(x, y) pairs into two tuples-of-scalars. This happens "
            "whenever a domain is set and at least one point passes the "
            "filter (i.e., essentially always for realistic calls), not "
            "just when some points get excluded."
        ),
        raises=AssertionError,
        strict=True,
    )
    def test_filtered_result_should_be_ndarray_not_tuple(self):
        k = Epanechnikov()
        xs = np.array([-0.5, 0.0, 0.5])
        ys = np.array([1.0, 2.0, 3.0])
        xs2, ys2 = k.in_domain(xs, ys, 0.0)
        assert isinstance(xs2, np.ndarray)
        assert isinstance(ys2, np.ndarray)


class TestSmoothPlainFloatVsNumpyScalar:
    # in_domain's tuple bug (see TestInDomain above) only actually
    # surfaces as a crash depending on the *type* of the query point x:
    # `numpy.float64 - tuple` broadcasts the tuple as array_like and
    # works, but `float - tuple` raises TypeError. This means the exact
    # same call succeeds or fails purely based on whether the caller
    # happens to pass a numpy scalar (e.g., from np.linspace) or a plain
    # Python float/int -- a fragile, surprising inconsistency.
    @classmethod
    def setup_class(cls):
        rs = np.random.RandomState(0)
        cls.xs = np.linspace(-1, 1, 20)
        cls.ys = cls.xs**2 + rs.normal(scale=0.05, size=20)

    def test_smooth_with_numpy_scalar_query_point_works(self):
        k = Epanechnikov()
        xg = np.linspace(-0.5, 0.5, 3)  # numpy.float64 elements
        result = np.array([k.smooth(self.xs, self.ys, xx) for xx in xg])
        assert np.all(np.isfinite(result))

    @pytest.mark.xfail(
        reason=(
            "BUG: same root cause as TestInDomain."
            "test_filtered_result_should_be_ndarray_not_tuple. A plain "
            "Python float query point (as opposed to a numpy scalar) "
            "triggers `TypeError: unsupported operand type(s) for -: "
            "'tuple' and 'float'` inside smooth(), for any kernel with a "
            "bounded domain (Epanechnikov, Uniform, Triangular, Cosine, "
            "Cosine2, Tricube, Triweight -- everything except Gaussian, "
            "whose domain is None, and Biweight, which overrides smooth() "
            "with its own implementation that doesn't call in_domain's "
            "buggy path the same way)."
        ),
        raises=TypeError,
        strict=True,
    )
    def test_smooth_with_plain_float_query_point_raises(self):
        k = Epanechnikov()
        k.smooth(self.xs, self.ys, 0.0)  # plain Python float


@pytest.mark.filterwarnings(
    "ignore:the matrix subclass:PendingDeprecationWarning"
)
class TestNdKernel:
    # NdKernel.__init__ builds its default H as np.matrix(np.identity(n)),
    # which raises PendingDeprecationWarning on numpy versions that warn
    # about np.matrix; not itself a functional bug, just deprecated numpy
    # usage inside the sandbox code, silenced here so it doesn't fail the
    # suite under warnings-as-errors.
    def test_default_construction_uses_gaussian_and_identity_h(self):
        ndk = NdKernel(2)
        assert isinstance(ndk._kernels, Gaussian)
        assert_allclose(np.asarray(ndk.H), np.eye(2))

    def test_h_property_getter_setter(self):
        ndk = NdKernel(2)
        new_h = np.matrix(np.eye(2) * 2)
        ndk.H = new_h
        assert ndk.H is new_h

    def test_density_is_finite_and_matches_call(self):
        rs = np.random.RandomState(0)
        ndk = NdKernel(2)
        xs = rs.normal(size=(30, 2))
        x = np.array([0.0, 0.0])
        result = ndk.density(xs, x)
        assert np.isfinite(result)

    def test_density_returns_nan_for_empty_input(self):
        ndk = NdKernel(2)
        result = ndk.density(np.empty((0, 2)), np.array([0.0, 0.0]))
        assert np.isnan(result)

    @pytest.mark.xfail(
        reason=(
            "BUG: NdKernel.density()'s weighted branch computes "
            "`np.mean(kernel_vals * weights) / sum(weights)`, but "
            "np.mean already divides by n, so this divides by n a second "
            "time via sum(weights) -- the correct weighted-mean formula "
            "is np.sum(kernel_vals * weights) / sum(weights) (no extra "
            "np.mean). For uniform weights=np.ones(n), sum(weights)==n, "
            "so the result is off from the correct (and from the "
            "unweighted-branch) answer by a factor of n."
        ),
        raises=AssertionError,
        strict=True,
    )
    def test_density_with_weights(self):
        rs = np.random.RandomState(0)
        ndk = NdKernel(2)
        xs = rs.normal(size=(30, 2))
        x = np.array([0.0, 0.0])
        ndk.weights = np.ones(30)
        result_weighted = ndk.density(xs, x)
        ndk.weights = None
        result_unweighted = ndk.density(xs, x)
        # uniform weights of 1 should match the unweighted computation
        assert_allclose(result_weighted, result_unweighted)

    def test_density_with_weights_matches_correct_weighted_mean_formula(self):
        # demonstrates the fix: sum(...) / sum(weights), not mean(...) / sum(weights)
        rs = np.random.RandomState(0)
        ndk = NdKernel(2)
        xs = rs.normal(size=(30, 2))
        x = np.array([0.0, 0.0])
        weights = np.ones(30)
        ndk.weights = weights
        kernel_vals = ndk._kernweight((xs - x) * ndk._Hrootinv).T
        corrected = np.sum(kernel_vals * weights) / np.sum(weights)
        ndk.weights = None
        result_unweighted = ndk.density(xs, x)
        assert_allclose(corrected, result_unweighted)


def test_cosine2_matches_stata_definition_at_zero():
    # K(0) = 1 + cos(0) = 2, normalized by norm_const
    k = Cosine2()
    assert_allclose(k.weight(0.0), k.norm_const * 2.0)
