"""
Tests for the deprecated statsmodels.tsa.descriptivestats.TsaDescriptive

TsaDescriptive will be removed after statsmodels 0.16. Until then, each
method is checked against the public function it wraps and, where one
exists, against a direct computation.
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import linalg, signal

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.descriptivestats import TsaDescriptive
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.tsatools import detrend

DEPRECATED = "TsaDescriptive is deprecated"


def make(data, label=None, name=""):
    with pytest.warns(FutureWarning, match=DEPRECATED):
        return TsaDescriptive(data, label=label, name=name)


@pytest.fixture
def data():
    # ARMA(1, 1) with phi = 0.5 and theta = 0.3
    rng = np.random.default_rng(20261119)
    return arma_generate_sample(
        [1, -0.5], [1, 0.3], 300, distrvs=rng.standard_normal, burnin=100
    )


def test_init(data):
    td = make(data, label="lbl", name="x")
    assert td.data is data
    assert td.label == "lbl"
    assert td.name == "x"
    assert td.mod is None
    assert td.res is None


def _first_difference(x):
    # lfilter starts from zero initial conditions, so out[0] = x[0]
    return np.r_[x[0], np.diff(x)]


def _ar1_recursion(x):
    out = np.empty_like(x)
    out[0] = x[0]
    for t in range(1, x.shape[0]):
        out[t] = x[t] + 0.5 * out[t - 1]
    return out


@pytest.mark.parametrize(
    "num, den, reference",
    [
        ([1.0, -1.0], [1.0], _first_difference),
        ([1.0], [1.0, -0.5], _ar1_recursion),
    ],
    ids=["first_difference", "ar1_recursion"],
)
@pytest.mark.parametrize("name", ["", "x"])
def test_filter(data, num, den, reference, name):
    td = make(data, label="lbl", name=name)
    with pytest.warns(FutureWarning, match=DEPRECATED):
        out = td.filter(num, den)
    assert type(out) is TsaDescriptive
    assert out.label == "lbl"
    assert out.name == name + "_filtered"
    assert_allclose(out.data, signal.lfilter(num, den, data), rtol=1e-12)
    assert_allclose(out.data, reference(data), rtol=1e-10)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_detrend(data, order):
    t = np.arange(data.shape[0], dtype=float)
    y = data + 2.0 + 0.05 * t - 1e-4 * t**2
    td = make(y, label="lbl", name="x")
    with pytest.warns(FutureWarning, match=DEPRECATED):
        out = td.detrend(order=order)
    assert type(out) is TsaDescriptive
    assert out.label == "lbl"
    assert out.name == "x_detrended"
    # Residuals of a least-squares fit of a polynomial trend of the given
    # order. They are O(1), and atol covers the few that are close to zero.
    trend = np.vander(t, order + 1)
    beta = np.linalg.lstsq(trend, y, rcond=None)[0]
    assert_allclose(out.data, y - trend @ beta, rtol=1e-8, atol=1e-10)
    assert_allclose(out.data, detrend(y, order=order), rtol=1e-12)


def test_detrend_default_order(data):
    td = make(data)
    with pytest.warns(FutureWarning, match=DEPRECATED):
        out = td.detrend()
    assert_allclose(out.data, detrend(data, order=1), rtol=1e-12)


def test_fit_default_order(data):
    td = make(data)
    res = td.fit()
    expected = ARIMA(data, order=(1, 0, 1)).fit()
    assert isinstance(td.mod, ARIMA)
    assert td.mod.order == (1, 0, 1)
    assert td.res is res
    assert_allclose(res.params, expected.params, rtol=1e-10)
    assert_allclose(res.llf, expected.llf, rtol=1e-10)


def test_fit_order_and_kwds(data):
    td = make(data)
    res = td.fit(order=(2, 0, 0), method="burg")
    expected = ARIMA(data, order=(2, 0, 0)).fit(method="burg")
    assert td.mod.order == (2, 0, 0)
    assert td.res is res
    assert res.fit_details["arma_estimator"] == "burg"
    assert_allclose(res.params, expected.params, rtol=1e-10)


def _sample_acf(x, nlags):
    d = x - x.mean()
    denom = d @ d
    return np.array([d[k:] @ d[: d.shape[0] - k] / denom for k in range(nlags + 1)])


@pytest.mark.parametrize("nlags", [None, 10])
def test_acf(data, nlags):
    td = make(data)
    if nlags is None:
        res, nlags = td.acf(), 40
    else:
        res = td.acf(nlags)
    assert_allclose(res, acf(data, nlags=nlags), rtol=1e-12)
    assert_allclose(res, _sample_acf(data, nlags), rtol=1e-10)


def _yule_walker_pacf(x, nlags):
    # The partial autocorrelation at lag k is the last coefficient of the
    # Yule-Walker AR(k) fit, here with autocovariances divided by n - k
    n = x.shape[0]
    d = x - x.mean()
    acov = np.array([d[k:] @ d[: n - k] / (n - k) for k in range(nlags + 1)])
    out = [1.0]
    for k in range(1, nlags + 1):
        out.append(linalg.solve_toeplitz(acov[:k], acov[1 : k + 1])[-1])
    return np.array(out)


@pytest.mark.parametrize("nlags", [None, 10])
def test_pacf(data, nlags):
    td = make(data)
    if nlags is None:
        res, nlags = td.pacf(), 40
    else:
        res = td.pacf(nlags)
    assert_allclose(res, pacf(data, nlags=nlags), rtol=1e-12)
    assert_allclose(res, _yule_walker_pacf(data, nlags), rtol=1e-8)


def _periodogram(x):
    # The definition of statsmodels.tsa.stattools.periodogram, removed in
    # statsmodels 0.12
    pgram = np.abs(np.fft.fft(x)) ** 2 / x.shape[0]
    pgram[0] = 0.0
    return pgram


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="TsaDescriptive.periodogram calls statsmodels.tsa.stattools."
    "periodogram, which was removed in statsmodels 0.12",
)
def test_periodogram(data):
    td = make(data)
    assert_allclose(td.periodogram(), _periodogram(data), rtol=1e-12)


class RestoredPeriodogram(TsaDescriptive):
    # TsaDescriptive.periodogram is broken (see test_periodogram), and plot4
    # calls it before drawing anything. Restore the removed function so
    # that the rest of plot4 can be checked.
    def periodogram(self):
        return _periodogram(np.asarray(self.data))


@pytest.mark.thread_unsafe(reason="Uses matplotlib")
@pytest.mark.matplotlib
@pytest.mark.parametrize("new_fig", [True, False])
@pytest.mark.parametrize("name", ["", "x"])
def test_plot4(data, close_figures, new_fig, name):
    import matplotlib.pyplot as plt

    with pytest.warns(FutureWarning, match=DEPRECATED):
        td = RestoredPeriodogram(data, name=name)
    if new_fig:
        fig = td.plot4(nacf=15, nfreq=50)
    else:
        given = plt.figure()
        fig = td.plot4(fig=given, nacf=15, nfreq=50)
        assert fig is given
    suffix = f" for {name}" if name else ""
    expected = [
        ("Time series", data),
        ("Autocorrelation", acf(data, nlags=15)),
        ("Power Spectrum", _periodogram(data)[:50]),
        ("Partial Autocorrelation", pacf(data, nlags=15)),
    ]
    assert len(fig.axes) == 4
    for ax, (title, values) in zip(fig.axes, expected, strict=True):
        assert ax.get_title() == title + suffix
        (line,) = ax.get_lines()
        assert_allclose(line.get_ydata(), values, rtol=1e-12)
