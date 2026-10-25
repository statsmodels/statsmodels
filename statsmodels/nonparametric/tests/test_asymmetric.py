"""
Created on Mon Mar  8 16:18:21 2021

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import pandas as pd
import pytest
from scipy import stats

import statsmodels.nonparametric.kernels_asymmetric as kern

kernels_rplus = [
    ("gamma", 0.1),
    ("gamma2", 0.1),
    ("invgamma", 0.02),
    ("invgauss", 0.01),
    ("recipinvgauss", 0.1),
    ("bs", 0.1),
    ("lognorm", 0.01),
    ("weibull", 0.1),
]

kernels_unit = [
    ("beta", 0.005),
    ("beta2", 0.005),
]


class CheckKernels:

    def test_kernels(self, case):
        name, bw = case

        rvs = self.rvs
        x_plot = self.x_plot

        kde = []
        kce = []
        for xi in x_plot:
            kde.append(kern.pdf_kernel_asym(xi, rvs, bw, name))
            kce.append(kern.cdf_kernel_asym(xi, rvs, bw, name))

        kde = np.asarray(kde)
        kce = np.asarray(kce)

        # average mean squared error
        amse = ((kde - self.pdf_dgp) ** 2).mean()
        assert_array_less(amse, self.amse_pdf)
        amse = ((kce - self.cdf_dgp) ** 2).mean()
        assert_array_less(amse, self.amse_cdf)

    def test_kernels_vectorized(self, case):
        name, bw = case

        rvs = self.rvs
        x_plot = self.x_plot

        kde = []
        kce = []
        for xi in x_plot:
            kde.append(kern.pdf_kernel_asym(xi, rvs, bw, name))
            kce.append(kern.cdf_kernel_asym(xi, rvs, bw, name))

        kde = np.asarray(kde)
        kce = np.asarray(kce)

        kde1 = kern.pdf_kernel_asym(x_plot, rvs, bw, name)
        kce1 = kern.cdf_kernel_asym(x_plot, rvs, bw, name)

        assert_allclose(kde1, kde, rtol=1e-12)
        assert_allclose(kce1, kce, rtol=1e-12)

    def test_kernels_weights(self, case):
        name, bw = case
        rvs = self.rvs
        x = self.x_plot

        kde2 = kern.pdf_kernel_asym(x, rvs, bw, name)
        kce2 = kern.cdf_kernel_asym(x, rvs, bw, name)

        n = len(rvs)
        w = np.ones(n) / n
        kde1 = kern.pdf_kernel_asym(x, rvs, bw, name, weights=w)
        kce1 = kern.cdf_kernel_asym(x, rvs, bw, name, weights=w)

        assert_allclose(kde1, kde2, rtol=1e-12)
        assert_allclose(kce1, kce2, rtol=1e-12)

        # weights that do not add to 1 are valid, but do not produce pdf, cdf
        n = len(rvs)
        w = np.ones(n) / n * 2
        kde1 = kern.pdf_kernel_asym(x, rvs, bw, name, weights=w)
        kce1 = kern.cdf_kernel_asym(x, rvs, bw, name, weights=w)

        assert_allclose(kde1, kde2 * 2, rtol=1e-12)
        assert_allclose(kce1, kce2 * 2, rtol=1e-12)


class TestKernelsRplus(CheckKernels):

    @classmethod
    def setup_class(cls):
        b = 2
        scale = 1.5
        rs = np.random.RandomState(1)
        nobs = 1000
        distr0 = stats.gamma(b, scale=scale)
        rvs = distr0.rvs(size=nobs, random_state=rs)
        x_plot = np.linspace(0.5, 16, 51) + 1e-13

        cls.rvs = rvs
        cls.x_plot = x_plot
        cls.pdf_dgp = distr0.pdf(x_plot)
        cls.cdf_dgp = distr0.cdf(x_plot)
        cls.amse_pdf = 1e-4  # tol for average mean squared error
        cls.amse_cdf = 5e-4

    @pytest.mark.parametrize("case", kernels_rplus)
    def test_kernels(self, case):
        super().test_kernels(case)

    @pytest.mark.parametrize("case", kernels_rplus)
    def test_kernels_vectorized(self, case):
        super().test_kernels_vectorized(case)

    @pytest.mark.parametrize("case", kernels_rplus)
    def test_kernels_weights(self, case):
        super().test_kernels_weights(case)


class TestKernelsUnit(CheckKernels):

    @classmethod
    def setup_class(cls):
        rs = np.random.RandomState(43784319)
        nobs = 1000
        distr0 = stats.beta(2, 3)
        rvs = distr0.rvs(size=nobs, random_state=rs)
        # Runtime warning if x_plot includes 0
        x_plot = np.linspace(1e-10, 1, 51)

        cls.rvs = rvs
        cls.x_plot = x_plot
        cls.pdf_dgp = distr0.pdf(x_plot)
        cls.cdf_dgp = distr0.cdf(x_plot)
        cls.amse_pdf = 0.02
        cls.amse_cdf = 5e-3

    @pytest.mark.parametrize("case", kernels_unit)
    def test_kernels(self, case):
        super().test_kernels(case)

    @pytest.mark.parametrize("case", kernels_unit)
    def test_kernels_vectorized(self, case):
        super().test_kernels_vectorized(case)

    @pytest.mark.parametrize("case", kernels_unit)
    def test_kernels_weights(self, case):
        super().test_kernels_weights(case)


def _asym_estimate(kind, x, sample, bw, kernel, **kwargs):
    return getattr(kern, f"{kind}_kernel_asym")(x, sample, bw, kernel, **kwargs)


def _asym_data(kernel, nobs, seed=0):
    rng = np.random.default_rng(seed)
    if kernel in ("beta", "beta2"):
        sample = rng.beta(2, 5, size=nobs)
    else:
        sample = rng.gamma(2.0, scale=1.5, size=nobs)
    return sample


@pytest.mark.parametrize("kind", ["pdf", "cdf"])
@pytest.mark.parametrize("kernel, bw", [("beta", 0.05), ("gamma", 0.1)])
def test_batch_loop_sample_larger_than_batch_limit(kind, kernel, bw):
    # The default batch_size=10 allows 10_000 elements per kernel array.
    # With 10_001 sample points the old loop computed k = 10_000 // 10_001 = 0.
    sample = _asym_data(kernel, 10001)
    x = np.array([0.1, 0.4, 0.8]) if kernel == "beta" else np.array([0.5, 2.0, 4.0])
    expected = _asym_estimate(kind, x, sample, bw, kernel, batch_size=1000)
    result = _asym_estimate(kind, x, sample, bw, kernel)
    assert_allclose(result, expected, rtol=1e-12)

    # scalar, 0-dim and length-1 inputs also failed in the batch loop
    for x_scalar in (x[0], np.float64(x[0]), np.array(x[0]), np.array([x[0]])):
        expected = _asym_estimate(
            kind, x_scalar, sample, bw, kernel, batch_size=1000
        )
        result = _asym_estimate(kind, x_scalar, sample, bw, kernel)
        assert np.ndim(result) == 0
        assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.parametrize("kind", ["pdf", "cdf"])
@pytest.mark.parametrize("kernel, bw", [("beta", 0.05), ("gamma", 0.1)])
def test_batch_loop_batch_size_smaller_than_sample(kind, kernel, bw):
    # Asking for a smaller batch must not require a sample smaller than the
    # batch limit; the sample itself is split into blocks as well.
    sample = _asym_data(kernel, 2000)
    x = np.array([0.1, 0.4, 0.8]) if kernel == "beta" else np.array([0.5, 2.0, 4.0])
    expected = _asym_estimate(kind, x, sample, bw, kernel, batch_size=100)
    result = _asym_estimate(kind, x, sample, bw, kernel, batch_size=1)
    assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.parametrize("kind", ["pdf", "cdf"])
def test_batch_loop_partial_last_block(kind):
    # 300 sample points and a 1000 element limit: three evaluation points per
    # block, with a final block containing a single point.
    sample = _asym_data("beta", 300)
    x = np.linspace(0.1, 0.9, 10)
    expected = _asym_estimate(kind, x, sample, 0.05, "beta", batch_size=100)
    result = _asym_estimate(kind, x, sample, 0.05, "beta", batch_size=1)
    assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.parametrize("kind", ["pdf", "cdf"])
def test_batch_loop_weights(kind):
    sample = _asym_data("beta", 2000)
    x = np.array([0.2, 0.5, 0.8])
    weights = np.linspace(0.5, 1.5, len(sample))
    expected = _asym_estimate(
        kind, x, sample, 0.05, "beta", weights=weights, batch_size=100
    )
    result = _asym_estimate(
        kind, x, sample, 0.05, "beta", weights=weights, batch_size=1
    )
    assert_allclose(result, expected, rtol=1e-12)
    # weights are summed, not normalized, also in the batch loop
    doubled = _asym_estimate(
        kind, x, sample, 0.05, "beta", weights=2 * weights, batch_size=1
    )
    assert_allclose(doubled, 2 * result, rtol=1e-12)


@pytest.mark.parametrize("kind", ["pdf", "cdf"])
@pytest.mark.parametrize("case", kernels_unit + kernels_rplus)
def test_batch_loop_all_kernels(kind, case):
    kernel, bw = case
    sample = _asym_data(kernel, 1200)
    if kernel in ("beta", "beta2"):
        x = np.array([0.1, 0.4, 0.8])
    else:
        x = np.array([0.5, 2.0, 4.0])
    expected = _asym_estimate(kind, x, sample, bw, kernel, batch_size=100)
    result = _asym_estimate(kind, x, sample, bw, kernel, batch_size=1)
    assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.parametrize("kind", ["pdf", "cdf"])
def test_batch_loop_intermediate_array_size(kind):
    # Documented bound: batch_size * 1000 elements per kernel array.
    shapes = []
    kernel_func = getattr(kern, f"kernel_{kind}_beta")
    estimate = getattr(kern, f"{kind}_kernel_asym")

    def recording_kernel(x, sample, bw):
        shapes.append((np.size(x), np.size(sample)))
        return kernel_func(x, sample, bw)

    sample = _asym_data("beta", 300)
    x = np.linspace(0.1, 0.9, 25)
    estimate(x, sample, 0.05, recording_kernel, batch_size=1)
    assert max(rows * cols for rows, cols in shapes) <= 1000
    # the sample is only split when it alone exceeds the limit
    assert {cols for _, cols in shapes} == {300}

    # More sample points than the limit: the sample must be split as well.
    sample = _asym_data("beta", 2500)
    shapes.clear()
    estimate(np.array([0.5]), sample, 0.05, recording_kernel, batch_size=1)
    assert max(rows * cols for rows, cols in shapes) <= 1000
    assert {rows for rows, _ in shapes} == {1}


@pytest.mark.parametrize("kind", ["pdf", "cdf"])
def test_batch_loop_scipy_reference(kind):
    sample = _asym_data("beta", 10001, seed=2024)
    x = np.array([0.1, 0.5, 0.9])
    result = _asym_estimate(kind, x, sample, 0.05, "beta")
    a = x / 0.05 + 1
    b = (1 - x) / 0.05 + 1
    values = (stats.beta.pdf if kind == "pdf" else stats.beta.sf)(
        sample[None, :], a[:, None], b[:, None]
    )
    assert_allclose(result, values.mean(-1), rtol=1e-13)


@pytest.mark.parametrize("kind", ["pdf", "cdf"])
@pytest.mark.parametrize("nobs", [9999, 10000, 10001])
def test_batch_loop_threshold(kind, nobs):
    sample = _asym_data("beta", nobs, seed=11)
    for x in (0.5, np.array([0.5]), np.array([0.2, 0.5, 0.8])):
        expected = _asym_estimate(
            kind, x, sample, 0.05, "beta", batch_size=1000
        )
        result = _asym_estimate(kind, x, sample, 0.05, "beta")
        assert np.shape(result) == np.shape(expected)
        assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.parametrize("kind", ["pdf", "cdf"])
def test_batch_loop_one_element_list(kind):
    sample = _asym_data("beta", 100)
    expected = _asym_estimate(kind, np.array([0.5]), sample, 0.05, "beta")
    result = _asym_estimate(kind, [0.5], sample, 0.05, "beta")
    assert np.shape(result) == np.shape(expected)
    assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.parametrize("kind", ["pdf", "cdf"])
@pytest.mark.parametrize(
    "kernel, bw, x",
    [
        ("beta", 0.05, [0.0, 1e-8, 0.05, 0.5, 1 - 1e-8, 1.0]),
        ("beta2", 0.05, [0.0, 1e-8, 0.05, 0.5, 1 - 1e-8, 1.0]),
        ("gamma", 0.1, [0.0, 1e-8, 0.1, 1.0, 3.0]),
        ("gamma2", 0.1, [0.0, 1e-8, 0.1, 1.0, 3.0]),
    ],
)
def test_batch_loop_support_endpoints(kind, kernel, bw, x):
    # endpoints and points inside the boundary-corrected region (x < 2*bw)
    sample = _asym_data(kernel, 10001, seed=5)
    x = np.array(x)
    kernel_func = getattr(kern, f"kernel_dict_{kind}")[kernel]
    expected = kernel_func(x[:, None], sample, bw).mean(-1)
    result = _asym_estimate(kind, x, sample, bw, kernel)
    assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.parametrize("kind", ["pdf", "cdf"])
def test_batch_loop_invalid_batch_size(kind):
    sample = _asym_data("beta", 100)
    with pytest.raises(ValueError, match="batch_size"):
        _asym_estimate(
            kind, np.array([0.5]), sample, 0.05, "beta", batch_size=0
        )


@pytest.mark.parametrize("kind", ["pdf", "cdf"])
def test_batch_loop_non_integer_batch_size(kind):
    # numeric values that are not Python ints were accepted before and must
    # keep working instead of raising inside the block loop
    sample = _asym_data("beta", 5000)
    x = np.array([0.2, 0.5, 0.8])
    expected = _asym_estimate(kind, x, sample, 0.05, "beta", batch_size=10)
    for batch_size in (10.0, np.float64(10.0), np.int64(10), float("inf")):
        result = _asym_estimate(
            kind, x, sample, 0.05, "beta", batch_size=batch_size
        )
        assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.parametrize("kind", ["pdf", "cdf"])
def test_batch_loop_weights_length_mismatch(kind):
    # splitting the sample must not silently drop weights beyond len(sample)
    sample = _asym_data("beta", 2000)
    weights = np.ones(len(sample) + 1) / len(sample)
    with pytest.raises(ValueError, match="weights"):
        _asym_estimate(
            kind,
            np.array([0.5]),
            sample,
            0.05,
            "beta",
            weights=weights,
            batch_size=1,
        )


@pytest.mark.parametrize("kind", ["pdf", "cdf"])
def test_callable_kernel_receives_scalar_x(kind):
    # A callable may rely on x staying a Python/NumPy scalar, as it did before
    # the batching fix; converting it to a 0-d array broke that contract.
    def scalar_kernel(x, sample, bw):
        assert np.isscalar(x)
        return np.ones(len(sample))

    estimate = getattr(kern, f"{kind}_kernel_asym")
    sample = np.linspace(0.1, 0.9, 100)
    for x in (0.5, np.float64(0.5)):
        result = estimate(x, sample, 0.05, scalar_kernel)
        assert np.ndim(result) == 0
        assert_allclose(result, 1.0)


@pytest.mark.parametrize("kind", ["pdf", "cdf"])
def test_callable_kernel_receives_sample_container(kind):
    # The batched path used to materialize the sample before calling the
    # kernel; a callable that relies on the caller's container must still see it.

    def series_kernel(x, sample, bw):
        assert isinstance(sample, pd.Series)
        return np.ones((len(x), len(sample)))

    estimate = getattr(kern, f"{kind}_kernel_asym")
    for index in (None, np.arange(1000) * 7 + 3):
        sample = pd.Series(np.linspace(0.1, 0.9, 1000), index=index)
        for m in (3, 20):  # unbatched and batched evaluation
            x = np.linspace(0.1, 0.9, m)
            result = estimate(x, sample, 0.05, series_kernel)
            assert_allclose(result, 1.0)


@pytest.mark.parametrize("kind", ["pdf", "cdf"])
def test_series_sample_matches_array(kind):
    estimate = getattr(kern, f"{kind}_kernel_asym")
    sample = np.random.default_rng(3).beta(2, 5, size=5000)
    x = np.array([0.2, 0.5, 0.8])
    expected = estimate(x, sample, 0.05, "beta")
    assert_allclose(
        estimate(x, pd.Series(sample), 0.05, "beta"), expected, rtol=1e-12
    )
    indexed = pd.Series(sample, index=np.arange(5000) * 3)
    assert_allclose(estimate(x, indexed, 0.05, "beta"), expected, rtol=1e-12)


@pytest.mark.parametrize(
    "batch_size, outcome",
    [
        pytest.param(0, "ValueError", id="zero"),
        pytest.param(-5, "ValueError", id="negative"),
        pytest.param(0.0001, "ValueError", id="budget-below-one"),
        pytest.param(0.5, "ok", id="half"),
        pytest.param(1, "ok", id="one"),
        pytest.param(2.5, "ok", id="two-point-five"),
        pytest.param(10.0, "ok", id="float"),
        pytest.param(np.float64(10.0), "ok", id="numpy-float"),
        pytest.param(np.int64(10), "ok", id="numpy-int"),
        pytest.param(float("inf"), "ok", id="inf"),
        pytest.param(float("nan"), "ValueError", id="nan"),
        pytest.param(True, "TypeError", id="bool"),
        pytest.param(None, "TypeError", id="none"),
        pytest.param("10", "TypeError", id="string"),
    ],
)
@pytest.mark.parametrize("kind", ["pdf", "cdf"])
def test_batch_size_budget_semantics(kind, batch_size, outcome):
    # the effective budget is batch_size * 1000 elements; it can be any number
    # and must allow at least one element
    sample = np.linspace(0.02, 0.98, 5000)
    x = np.array([0.2, 0.5, 0.8])
    if outcome == "ok":
        expected = _asym_estimate(kind, x, sample, 0.05, "beta", batch_size=10)
        result = _asym_estimate(
            kind, x, sample, 0.05, "beta", batch_size=batch_size
        )
        assert_allclose(result, expected, rtol=1e-12)
    else:
        error = {"ValueError": ValueError, "TypeError": TypeError}[outcome]
        with pytest.raises(error, match="batch_size"):
            _asym_estimate(kind, x, sample, 0.05, "beta", batch_size=batch_size)


@pytest.mark.parametrize("kind", ["pdf", "cdf"])
@pytest.mark.parametrize("kernel", ["beta", "beta2", "gamma", "gamma2"])
@pytest.mark.parametrize("batch_size", [10, 0.01], ids=["whole", "batched"])
def test_one_element_x_containers(kind, kernel, batch_size):
    # A single-element x in any container returns a scalar. A one-element
    # Series used to fail in the kernels that branch on x, if not batched.
    sample = _asym_data(kernel, 100)
    bw = 0.1
    x0 = 0.3 if kernel.startswith("beta") else 1.5
    expected = _asym_estimate(kind, x0, sample, bw, kernel)
    for x in ([x0], (x0,), np.array([x0]), pd.Series([x0])):
        result = _asym_estimate(
            kind, x, sample, bw, kernel, batch_size=batch_size
        )
        assert np.ndim(result) == 0
        assert_allclose(result, expected, rtol=1e-12)

    # two elements are not reduced to a scalar
    x2 = pd.Series([x0, 1.1 * x0], index=[3, 7])
    expected = np.array(
        [_asym_estimate(kind, v, sample, bw, kernel) for v in x2]
    )
    result = _asym_estimate(kind, x2, sample, bw, kernel, batch_size=batch_size)
    assert result.shape == (2,)
    assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.parametrize("kind", ["pdf", "cdf"])
def test_callable_kernel_receives_zero_dim_array(kind):
    # A 0-d array is not a scalar; main passed it through as an array, so a
    # callable can rely on receiving a non-scalar 0-d value.
    def array_kernel(x, sample, bw):
        assert not np.isscalar(x)
        assert np.ndim(x) == 0
        return np.ones(len(sample))

    estimate = getattr(kern, f"{kind}_kernel_asym")
    sample = np.linspace(0.1, 0.9, 100)
    result = estimate(np.array(0.5), sample, 0.05, array_kernel)
    assert np.ndim(result) == 0
    assert_allclose(result, 1.0)


@pytest.mark.parametrize("kind", ["pdf", "cdf"])
def test_callable_kernel_receives_list_sample(kind):
    # A list sample must reach the kernel as a list when the sample is not split.
    def list_kernel(x, sample, bw):
        assert isinstance(sample, list)
        return np.ones((len(x), len(sample)))

    estimate = getattr(kern, f"{kind}_kernel_asym")
    sample = list(np.linspace(0.1, 0.9, 1000))
    x = np.linspace(0.1, 0.9, 20)  # 2 * 10**4 >= 10**4 -> batched, n <= budget
    result = estimate(x, sample, 0.05, list_kernel)
    assert_allclose(result, 1.0)


@pytest.mark.parametrize("kind", ["pdf", "cdf"])
def test_batch_loop_two_column_weights(kind):
    # main accepted (n, 2) weights and returned (m, 2); the batched path keeps it.
    estimate = getattr(kern, f"{kind}_kernel_asym")
    sample = np.linspace(0.02, 0.98, 5000)
    x = np.array([0.2, 0.5, 0.8])
    weights = np.column_stack(
        [np.linspace(0.5, 1.5, 5000), np.linspace(1.5, 0.5, 5000)]
    )
    result = estimate(x, sample, 0.05, "beta", weights=weights)
    a = x[:, None] / 0.05 + 1
    b = (1 - x[:, None]) / 0.05 + 1
    values = (stats.beta.pdf if kind == "pdf" else stats.beta.sf)(
        sample[None, :], a, b
    )
    assert result.shape == (3, 2)
    assert_allclose(result, values @ weights, rtol=1e-12)


@pytest.mark.parametrize("kind", ["pdf", "cdf"])
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weights"])
# 100 elements per kernel array split the evaluation points only, 10 elements
# split the sample into several blocks as well
@pytest.mark.parametrize("batch_size", [0.1, 0.01], ids=["split-x", "split-sample"])
@pytest.mark.parametrize("kernel", sorted(kern.kernel_dict_pdf))
def test_batch_loop_per_point_reference(kind, kernel, batch_size, weighted):
    # Independent reference: one call of the kernel function for each point
    # with scalar x, summed in extended precision, without any batching.
    kernel_func = getattr(kern, f"kernel_dict_{kind}")[kernel]
    bw = 0.05 if kernel.startswith("beta") else dict(kernels_rplus)[kernel]
    nobs = 30
    sample = _asym_data(kernel, nobs, seed=17)
    if kernel.startswith("beta"):
        # includes points inside the boundary region x < 2 * bw
        x = np.array([0.02, 0.08, 0.3, 0.5, 0.7, 0.93, 0.99])
    else:
        x = np.array([0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 6.0])

    rng = np.random.default_rng(23)
    weights = rng.uniform(0.5, 1.5, size=nobs) / nobs if weighted else None
    w = weights if weighted else np.full(nobs, 1 / nobs)
    w = w.astype(np.longdouble)

    expected = np.array(
        [
            np.sum(np.asarray(kernel_func(xi, sample, bw), np.longdouble) * w)
            for xi in x
        ],
        dtype=float,
    )
    result = _asym_estimate(
        kind, x, sample, bw, kernel, weights=weights, batch_size=batch_size
    )
    assert result.shape == x.shape
    assert_allclose(result, expected, rtol=1e-12)
