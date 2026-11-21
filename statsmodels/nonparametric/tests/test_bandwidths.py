"""

Tests for bandwidth selection and calculation.

Author: Padarn Wilson
"""

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import optimize, stats

from statsmodels.datasets import engel, nile, sunspots
from statsmodels.distributions.mixture_rvs import mixture_rvs
from statsmodels.nonparametric.bandwidths import (
    _bw_isj,
    _isj_fixed_point,
    _select_sigma,
    bw_isj,
    bw_normal_reference,
    select_bandwidth,
)
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.sandbox.nonparametric import kernels

# setup test data
RANDOM_STATE = np.random.RandomState(12345)
Xi = mixture_rvs(
    [0.25, 0.75],
    size=200,
    dist=[stats.norm, stats.norm],
    kwargs=(dict(loc=-1, scale=0.5), dict(loc=1, scale=0.5)),
    rng=RANDOM_STATE,
)


class TestBandwidthCalculation:

    def test_calculate_bandwidth_gaussian(self):

        bw_expected = [0.29774853596742024, 0.25304408155871411, 0.29781147113698891]

        kern = kernels.Gaussian()

        bw_calc = [0, 0, 0]
        for ii, bw in enumerate(["scott", "silverman", "normal_reference"]):
            bw_calc[ii] = select_bandwidth(Xi, bw, kern)

        assert_allclose(bw_expected, bw_calc)

    def test_calculate_normal_reference_bandwidth(self):
        # Should be the same as the Gaussian Kernel
        bw_expected = 0.29781147113698891
        bw = bw_normal_reference(Xi)
        assert_allclose(bw, bw_expected)


@pytest.mark.parametrize("percentile", [40, 45])
def test_select_sigma_percentile(percentile):
    # the inter-percentile range must be normalized by the same range of the
    # standard normal, otherwise a non-default percentile badly understates
    # the dispersion of normal data
    rs = np.random.RandomState(12345)
    x = rs.standard_normal(50000)
    assert_allclose(
        _select_sigma(x, percentile=percentile), np.std(x, ddof=1), rtol=0.02
    )


def test_select_sigma_default_normalization():
    # the default keeps Silverman's rounded constant, so bandwidths that
    # depend on it are unchanged
    rs = np.random.RandomState(12345)
    x = rs.standard_normal(500)
    iqr = stats.scoreatpercentile(x, 75) - stats.scoreatpercentile(x, 25)
    expected = np.minimum(np.std(x, ddof=1), iqr / 1.349)
    assert_allclose(_select_sigma(x), expected, rtol=0, atol=0)


class CheckNormalReferenceConstant:

    def test_calculate_normal_reference_constant(self):
        const = self.constant
        kern = self.kern
        assert_allclose(const, kern.normal_reference_constant, 1e-2)


class TestEpanechnikov(CheckNormalReferenceConstant):

    kern = kernels.Epanechnikov()
    constant = 2.34


class TestGaussian(CheckNormalReferenceConstant):

    kern = kernels.Gaussian()
    constant = 1.06


class TestBiweight(CheckNormalReferenceConstant):

    kern = kernels.Biweight()
    constant = 2.78


class TestTriweight(CheckNormalReferenceConstant):

    kern = kernels.Triweight()
    constant = 3.15


class BandwidthZero:

    def test_bandwidth_zero(self):

        kern = kernels.Gaussian()
        for bw in ["scott", "silverman", "normal_reference"]:
            with pytest.raises(RuntimeError, match="Selected KDE bandwidth is 0"):
                select_bandwidth(self.xx, bw, kern)


class TestAllBandwidthZero(BandwidthZero):

    xx = np.ones((100, 3))


class TestAnyBandwidthZero(BandwidthZero):
    rs = np.random.RandomState(328193821)
    xx = rs.normal(size=(100, 3))
    xx[:, 0] = 1.0


def _isj_data(name):
    if name == "xi":
        fname = Path(__file__).parent / "Xi_test_data.csv"
        return np.loadtxt(fname, delimiter=",")[:, 0]
    if name == "nile":
        return nile.load_pandas().data["volume"].to_numpy(dtype=float)
    if name == "sunspots":
        return sunspots.load_pandas().data["SUNACTIVITY"].to_numpy(dtype=float)
    return engel.load_pandas().data[name].to_numpy(dtype=float)


# Reference values are from botev() in the R package provenance 4.4, which
# uses 512 bins. All data except for xi have ties.
#
# library(provenance)
# xi <- read.csv("nonparametric/tests/Xi_test_data.csv", header=FALSE)[, 1]
# nile <- read.csv("datasets/nile/nile.csv")$volume
# engel <- read.csv("datasets/engel/engel.csv")
# sunspots <- read.csv("datasets/sunspots/sunspots.csv")$SUNACTIVITY
# for (x in list(xi, nile, engel$income, engel$foodexp, sunspots)) {
#     cat(sprintf("%.15g", botev(x)), "\n")
# }
ISJ_R = {
    "xi": 0.212842429629364,
    "nile": 78.2881681228895,
    "income": 97.2809635698703,
    "foodexp": 78.1933644270126,
    "sunspots": 4.91338904146496,
}


class TestBandwidthISJ:
    @pytest.mark.parametrize("name", list(ISJ_R))
    def test_r_provenance(self, name):
        x = _isj_data(name)
        assert_allclose(_bw_isj(x, gridsize=512), ISJ_R[name], rtol=1e-10)

    @pytest.mark.parametrize("name", list(ISJ_R))
    def test_unbinned(self, name):
        # the default grid is fine enough to agree with the solution that
        # uses the cosine series coefficients of the data without binning
        x = _isj_data(name)
        grid_range = 1.2 * np.ptp(x)
        u = (x - x.min()) / grid_range + 1 / 12
        k = np.arange(1, 2**12)
        a2 = np.cos(np.pi * np.outer(k, u)).mean(1) ** 2
        args = (len(np.unique(x)), k.astype(float) ** 2, a2)
        t_star = optimize.brentq(_isj_fixed_point, 0, 0.1, args=args, xtol=1e-200)
        assert_allclose(bw_isj(x), np.sqrt(t_star) * grid_range, rtol=5e-4)
        # 512 bins are too coarse for the skewed income data
        if name == "income":
            assert abs(ISJ_R[name] / bw_isj(x) - 1) > 0.03

    def test_equivariance(self):
        x = _isj_data("xi")
        bw = bw_isj(x)
        assert_allclose(bw_isj(10 * x + 3), 10 * bw, rtol=1e-10)
        assert_allclose(bw_isj(-0.25 * x), 0.25 * bw, rtol=1e-10)

    def test_select_bandwidth(self):
        x = _isj_data("xi")
        bw = bw_isj(x)
        kern = kernels.Gaussian()
        assert_allclose(bw_isj(x, kern), bw, rtol=1e-13)
        assert_allclose(select_bandwidth(x, "isj", kern), bw, rtol=1e-13)
        assert_allclose(select_bandwidth(x[:, None], "ISJ", kern), bw, rtol=1e-13)
        # bimodal data, the normal reference bandwidth is larger
        assert bw < 0.8 * select_bandwidth(x, "normal_reference", kern)

    @pytest.mark.parametrize("fft", [True, False])
    def test_kde_univariate(self, fft):
        x = _isj_data("xi")
        bw = bw_isj(x)
        kde = KDEUnivariate(x).fit(bw="isj", fft=fft)
        assert kde.bw_method == "isj"
        assert_allclose(kde.bw, bw, rtol=1e-13)
        kde2 = KDEUnivariate(x).fit(bw=bw, fft=fft)
        assert_allclose(kde.density, kde2.density, rtol=1e-13)
        kde3 = KDEUnivariate(x).fit(bw="isj", fft=fft, adjust=2)
        assert_allclose(kde3.bw, 2 * bw, rtol=1e-13)

    @pytest.mark.parametrize(
        "kern", [kernels.Epanechnikov(), kernels.Biweight(), kernels.Uniform()]
    )
    def test_kernel_not_gaussian(self, kern):
        x = _isj_data("xi")
        with pytest.raises(ValueError, match="only available for the Gaussian"):
            bw_isj(x, kern)
        with pytest.raises(ValueError, match="only available for the Gaussian"):
            select_bandwidth(x, "isj", kern)

    def test_kde_univariate_kernel_not_gaussian(self):
        x = _isj_data("xi")
        with pytest.raises(ValueError, match="only available for the Gaussian"):
            KDEUnivariate(x).fit(kernel="epa", bw="isj", fft=False)

    @pytest.mark.parametrize(
        "x",
        [
            np.array([1.0, 2.0]),
            np.arange(10.0),
            np.linspace(0, 1, 50),
            np.repeat([1.0, 2.0, 3.0, 4.0], 30),
        ],
    )
    def test_no_solution(self, x):
        with pytest.raises(RuntimeError, match="fixed point equation has no"):
            bw_isj(x)
        with pytest.raises(RuntimeError, match="fixed point equation has no"):
            KDEUnivariate(x).fit(bw="isj")

    @pytest.mark.parametrize("x", [np.ones(1), np.ones(20)])
    def test_bandwidth_zero(self, x):
        assert bw_isj(x) == 0
        with pytest.raises(RuntimeError, match="Selected KDE bandwidth is 0"):
            select_bandwidth(x, "isj", kernels.Gaussian())

    def test_2d(self):
        x = _isj_data("xi")
        with pytest.raises(ValueError, match="x is required to have ndim 1"):
            bw_isj(np.column_stack((x, x)))
