"""Independent mixture integrals for KDE probability and clipping behavior."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy import integrate, stats

from statsmodels.nonparametric.kde import KDEUnivariate, kernel_switch

KERNELS = tuple(kernel_switch)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("weighted", [False, True])
def test_probabilities_against_integrated_kernel(kernel, weighted):
    # Integrate each original kernel separately, not the mixture algorithm.
    data = np.array([-2.0, -0.2, 0.4, 1.8])
    weights = np.array([0.0, 1.0, 3.0, 2.0]) if weighted else None
    model = KDEUnivariate(data).fit(
        kernel=kernel, bw=0.65, fft=False, weights=weights, gridsize=37, cut=1.5
    )
    kern = kernel_switch[kernel]()
    lower, upper = (-np.inf, np.inf) if kern.domain is None else kern.domain
    normalized = (
        np.ones(len(data)) / len(data) if weights is None else weights / weights.sum()
    )

    def integral(a, b):
        points = [0.0] if np.isfinite(a) and np.isfinite(b) and a < 0 < b else None
        return integrate.quad(
            lambda value: float(kern(np.asarray(value))),
            a,
            b,
            epsabs=2e-13,
            epsrel=2e-13,
            points=points,
        )[0]

    cdf, sf = [], []
    for point in model.support:
        z = (point - data) / model.bw
        left, right = [], []
        for value in z:
            cut = min(max(value, lower), upper)
            left.append(integral(lower, cut))
            right.append(integral(cut, upper))
        cdf.append(np.dot(normalized, left))
        sf.append(np.dot(normalized, right))
    # Quadrature's absolute error controls probabilities near zero.
    assert_allclose(model.cdf, cdf, rtol=2e-11, atol=2e-13)
    assert_allclose(model.sf, sf, rtol=2e-11, atol=2e-13)
    assert_allclose(model.cdf + model.sf, 1.0, rtol=0, atol=4e-15)
    assert np.all(np.diff(model.cdf) >= -4e-15)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("weighted", [False, True])
def test_clipped_fit_is_the_retained_sample(kernel, weighted):
    data = np.array([-30.0, -2.0, -0.4, 0.2, 1.0, 2.0, 40.0])
    weights = np.array([40.0, 11.0, 1.0, 3.0, 2.0, 17.0, 50.0]) if weighted else None
    keep = (data > -2.0) & (data < 2.0)
    fit_kw = dict(kernel=kernel, bw=0.55, fft=False, gridsize=25)
    clipped = KDEUnivariate(data).fit(clip=(-2.0, 2.0), weights=weights, **fit_kw)
    retained = KDEUnivariate(data[keep]).fit(
        weights=None if weights is None else weights[keep], **fit_kw
    )
    assert_array_equal(clipped.endog, data)
    assert_allclose(clipped.density, retained.density, rtol=2e-14, atol=0)
    assert_allclose(
        clipped.evaluate(clipped.support),
        retained.evaluate(retained.support),
        rtol=2e-14,
        atol=0,
    )
    assert_allclose(clipped.cdf, retained.cdf, rtol=2e-14, atol=0)
    assert_allclose(clipped.sf, retained.sf, rtol=2e-14, atol=0)
    assert_allclose(clipped.icdf, retained.icdf, rtol=2e-14, atol=0)


def test_gaussian_upper_tail_and_shifted_grid():
    # Closed-form single-component normal probabilities at >8 sigma.
    for location in [0.0, 1e6]:
        model = KDEUnivariate([location, location]).fit(
            bw=0.25, fft=False, gridsize=9, cut=12
        )
        z = (model.support - location) / model.bw
        assert_allclose(model.cdf, stats.norm.cdf(z), rtol=3e-15, atol=0)
        assert_allclose(model.sf, stats.norm.sf(z), rtol=3e-15, atol=0)
        assert model.sf[-1] > 0
        assert_allclose(model.cumhazard, -stats.norm.logsf(z), rtol=3e-15, atol=2e-16)


def test_refit_and_fft_use_current_sample():
    data = np.array([-20.0, -1.0, 0.0, 0.5, 1.5, 30.0])
    model = KDEUnivariate(data).fit(bw=0.6, fft=True, gridsize=64)
    old_cdf = model.cdf.copy()
    model.fit(bw=0.3, fft=True, gridsize=64, clip=(-2, 2))
    expected = KDEUnivariate(data[1:-1]).fit(bw=0.3, fft=True, gridsize=64)
    assert not np.array_equal(old_cdf, model.cdf)
    assert_allclose(model.cdf, expected.cdf, rtol=2e-14, atol=0)
    assert_allclose(model.sf, expected.sf, rtol=2e-14, atol=0)
    assert_allclose(model.entropy, expected.entropy, rtol=2e-14, atol=0)


def test_repeated_observations_preserve_probability_mass():
    data = np.repeat([-1.5, 0.3, 2.0], [90000, 100000, 90003])
    model = KDEUnivariate(data).fit(bw=0.7, fft=True, gridsize=16)
    weights = np.array([90000, 100000, 90003]) / len(data)
    expected = (
        stats.norm.cdf((model.support[:, None] - [-1.5, 0.3, 2.0]) / 0.7) @ weights
    )
    expected_sf = (
        stats.norm.sf((model.support[:, None] - [-1.5, 0.3, 2.0]) / 0.7) @ weights
    )
    assert_allclose(model.cdf, expected, rtol=4e-14, atol=0)
    assert_allclose(model.sf, expected_sf, rtol=4e-14, atol=0)


def test_probability_blocks_cover_large_distinct_observation_axis():
    # Distinct observations retain the blocked path after duplicate aggregation.
    data = np.linspace(-2.0, 2.0, 270003)
    model = KDEUnivariate(data).fit(bw=0.7, fft=True, gridsize=16)
    expected = [stats.norm.cdf((point - data) / 0.7).mean() for point in model.support]
    expected_sf = [
        stats.norm.sf((point - data) / 0.7).mean() for point in model.support
    ]
    assert_allclose(model.cdf, expected, rtol=4e-14, atol=0)
    assert_allclose(model.sf, expected_sf, rtol=4e-14, atol=0)


def test_numeric_object_weights():
    data = [-1.0, 0.0, 2.0]
    weights = np.array([1.0, 2.0, 3.0])
    plain = KDEUnivariate(data).fit(weights=weights, fft=False, bw=0.5, gridsize=8)
    objects = KDEUnivariate(data).fit(
        weights=weights.astype(object), fft=False, bw=0.5, gridsize=8
    )
    assert_allclose(objects.cdf, plain.cdf, rtol=2e-14, atol=0)
    assert_allclose(objects.sf, plain.sf, rtol=2e-14, atol=0)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("weighted", [False, True])
def test_repeated_clipped_sample_matches_explicit_three_component_mixture(
    kernel, weighted
):
    data = np.tile([-8.0, -2.0, -0.4, 0.2, 0.2, 1.0, -0.4, 2.0, 0.2, 8.0, 1.0, 1.0], 64)
    weights = None
    if weighted:
        weights = np.tile(
            np.array(
                [100.0, 11.0, 1.0, 3.0, 0.0, 5.0, 2.0, 17.0, 0.5, 100.0, 0.25, 4.0],
                dtype=object,
            ),
            64,
        )
    fit_kw = dict(kernel=kernel, bw=0.55, fft=False, gridsize=31)
    actual = KDEUnivariate(data).fit(weights=weights, clip=(-2.0, 2.0), **fit_kw)
    # Explicit masses from the three retained values in the original 12 rows.
    masses = [3.0, 3.5, 9.25] if weighted else [2.0, 3.0, 3.0]
    reference = KDEUnivariate([-0.4, 0.2, 1.0]).fit(weights=masses, **fit_kw)
    assert_allclose(actual.cdf, reference.cdf, rtol=3e-15, atol=0)
    assert_allclose(actual.sf, reference.sf, rtol=3e-15, atol=0)


def test_aggregated_weights_preserve_small_tail_mass():
    model = KDEUnivariate(np.zeros(100000)).fit(
        weights=np.ones(100000), bw=1.0, fft=False, gridsize=3, cut=12
    )
    assert_allclose(model.cdf, stats.norm.cdf(model.support), rtol=3e-15, atol=0)
    assert_allclose(model.sf, stats.norm.sf(model.support), rtol=3e-15, atol=0)


def test_legacy_fitted_object_and_custom_kernel_fallback():
    from statsmodels.sandbox.nonparametric.kernels import Gaussian

    class GaussianSubclass(Gaussian):
        pass

    data = np.array([-1.0, 0.0, 2.0])
    model = KDEUnivariate(data).fit(fft=False, bw=0.5, gridsize=8)
    del model._fit_endog  # Older fitted objects have only the original endog.
    model.kernel = GaussianSubclass(h=model.bw)
    z = (model.support[:, None] - data) / model.bw
    assert_allclose(model.cdf, stats.norm.cdf(z).mean(axis=1), rtol=2e-12, atol=2e-14)
    assert_allclose(model.sf, stats.norm.sf(z).mean(axis=1), rtol=2e-12, atol=2e-14)
