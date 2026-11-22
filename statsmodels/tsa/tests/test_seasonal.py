from statsmodels.compat.pandas import MONTH_END, QUARTER_END, YEAR_END

import warnings

import numpy as np
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_equal,
)
import pandas as pd
import pytest

from statsmodels.tsa.seasonal import seasonal_decompose, seasonal_mean
from statsmodels.tsa.tests._contract import SERIES_FORMS, series_forms

# Verification values for tests
SEASONAL = [
    62.46,
    86.17,
    -88.38,
    -60.25,
    62.46,
    86.17,
    -88.38,
    -60.25,
    62.46,
    86.17,
    -88.38,
    -60.25,
    62.46,
    86.17,
    -88.38,
    -60.25,
    62.46,
    86.17,
    -88.38,
    -60.25,
    62.46,
    86.17,
    -88.38,
    -60.25,
    62.46,
    86.17,
    -88.38,
    -60.25,
    62.46,
    86.17,
    -88.38,
    -60.25,
]
TREND = [
    np.nan,
    np.nan,
    159.12,
    204.00,
    221.25,
    245.12,
    319.75,
    451.50,
    561.12,
    619.25,
    615.62,
    548.00,
    462.12,
    381.12,
    316.62,
    264.00,
    228.38,
    210.75,
    188.38,
    199.00,
    207.12,
    191.00,
    166.88,
    72.00,
    -9.25,
    -33.12,
    -36.75,
    36.25,
    103.00,
    131.62,
    np.nan,
    np.nan,
]
RANDOM = [
    np.nan,
    np.nan,
    78.254,
    70.254,
    -36.710,
    -94.299,
    -6.371,
    -62.246,
    105.415,
    103.576,
    2.754,
    1.254,
    15.415,
    -10.299,
    -33.246,
    -27.746,
    46.165,
    -57.924,
    28.004,
    -36.746,
    -37.585,
    151.826,
    -75.496,
    86.254,
    -10.210,
    -194.049,
    48.129,
    11.004,
    -40.460,
    143.201,
    np.nan,
    np.nan,
]

MULT_SEASONAL = [
    1.0815,
    1.5538,
    0.6716,
    0.6931,
    1.0815,
    1.5538,
    0.6716,
    0.6931,
    1.0815,
    1.5538,
    0.6716,
    0.6931,
    1.0815,
    1.5538,
    0.6716,
    0.6931,
    1.0815,
    1.5538,
    0.6716,
    0.6931,
    1.0815,
    1.5538,
    0.6716,
    0.6931,
    1.0815,
    1.5538,
    0.6716,
    0.6931,
    1.0815,
    1.5538,
    0.6716,
    0.6931,
]
MULT_TREND = [
    np.nan,
    np.nan,
    171.62,
    204.00,
    221.25,
    245.12,
    319.75,
    451.50,
    561.12,
    619.25,
    615.62,
    548.00,
    462.12,
    381.12,
    316.62,
    264.00,
    228.38,
    210.75,
    188.38,
    199.00,
    207.12,
    191.00,
    166.88,
    107.25,
    80.50,
    79.12,
    78.75,
    116.50,
    140.00,
    157.38,
    np.nan,
    np.nan,
]
MULT_RANDOM = [
    np.nan,
    np.nan,
    1.29263,
    1.51360,
    1.03223,
    0.62226,
    1.04771,
    1.05139,
    1.20124,
    0.84080,
    1.28182,
    1.28752,
    1.08043,
    0.77172,
    0.91697,
    0.96191,
    1.36441,
    0.72986,
    1.01171,
    0.73956,
    1.03566,
    1.44556,
    0.02677,
    1.31843,
    0.49390,
    1.14688,
    1.45582,
    0.16101,
    0.82555,
    1.47633,
    np.nan,
    np.nan,
]


class TestDecompose:
    @classmethod
    def setup_class(cls):
        # even
        data = [
            -50,
            175,
            149,
            214,
            247,
            237,
            225,
            329,
            729,
            809,
            530,
            489,
            540,
            457,
            195,
            176,
            337,
            239,
            128,
            102,
            232,
            429,
            3,
            98,
            43,
            -141,
            -77,
            -13,
            125,
            361,
            -45,
            184,
        ]
        index = pd.date_range(start="1/1/1951", periods=len(data), freq=QUARTER_END)
        df = pd.DataFrame(data, index=index)
        cls.data = df
        df = df.copy()
        df.index = pd.period_range(start="1/1/1951", periods=len(data), freq="Q")
        cls.data_period_index = df

    def test_ndarray(self):
        res_add = seasonal_decompose(self.data.values, period=4)
        assert_almost_equal(res_add.seasonal, SEASONAL, 2)
        assert_almost_equal(res_add.trend, TREND, 2)
        assert_almost_equal(res_add.resid, RANDOM, 3)

        res_mult = seasonal_decompose(np.abs(self.data.values), "mul", period=4)

        assert_almost_equal(res_mult.seasonal, MULT_SEASONAL, 4)
        assert_almost_equal(res_mult.trend, MULT_TREND, 2)
        assert_almost_equal(res_mult.resid, MULT_RANDOM, 4)

        # test odd
        res_add = seasonal_decompose(self.data.values[:-1], period=4)
        seasonal = [
            68.18,
            69.02,
            -82.66,
            -54.54,
            68.18,
            69.02,
            -82.66,
            -54.54,
            68.18,
            69.02,
            -82.66,
            -54.54,
            68.18,
            69.02,
            -82.66,
            -54.54,
            68.18,
            69.02,
            -82.66,
            -54.54,
            68.18,
            69.02,
            -82.66,
            -54.54,
            68.18,
            69.02,
            -82.66,
            -54.54,
            68.18,
            69.02,
            -82.66,
        ]
        trend = [
            np.nan,
            np.nan,
            159.12,
            204.00,
            221.25,
            245.12,
            319.75,
            451.50,
            561.12,
            619.25,
            615.62,
            548.00,
            462.12,
            381.12,
            316.62,
            264.00,
            228.38,
            210.75,
            188.38,
            199.00,
            207.12,
            191.00,
            166.88,
            72.00,
            -9.25,
            -33.12,
            -36.75,
            36.25,
            103.00,
            np.nan,
            np.nan,
        ]
        random = [
            np.nan,
            np.nan,
            72.538,
            64.538,
            -42.426,
            -77.150,
            -12.087,
            -67.962,
            99.699,
            120.725,
            -2.962,
            -4.462,
            9.699,
            6.850,
            -38.962,
            -33.462,
            40.449,
            -40.775,
            22.288,
            -42.462,
            -43.301,
            168.975,
            -81.212,
            80.538,
            -15.926,
            -176.900,
            42.413,
            5.288,
            -46.176,
            np.nan,
            np.nan,
        ]
        assert_almost_equal(res_add.seasonal, seasonal, 2)
        assert_almost_equal(res_add.trend, trend, 2)
        assert_almost_equal(res_add.resid, random, 3)

    def test_pandas(self):
        res_add = seasonal_decompose(self.data, period=4)
        freq_override_data = self.data.copy()
        freq_override_data.index = pd.date_range(
            start="1/1/1951", periods=len(freq_override_data), freq=YEAR_END
        )
        res_add_override = seasonal_decompose(freq_override_data, period=4)

        assert_almost_equal(res_add.seasonal.values.squeeze(), SEASONAL, 2)
        assert_almost_equal(res_add.trend.values.squeeze(), TREND, 2)
        assert_almost_equal(res_add.resid.values.squeeze(), RANDOM, 3)
        assert_almost_equal(res_add_override.seasonal.values.squeeze(), SEASONAL, 2)
        assert_almost_equal(res_add_override.trend.values.squeeze(), TREND, 2)
        assert_almost_equal(res_add_override.resid.values.squeeze(), RANDOM, 3)
        assert_equal(res_add.seasonal.index.values.squeeze(), self.data.index.values)

        res_mult = seasonal_decompose(np.abs(self.data), "mul", period=4)
        res_mult_override = seasonal_decompose(
            np.abs(freq_override_data), "mul", period=4
        )
        assert_almost_equal(res_mult.seasonal.values.squeeze(), MULT_SEASONAL, 4)
        assert_almost_equal(res_mult.trend.values.squeeze(), MULT_TREND, 2)
        assert_almost_equal(res_mult.resid.values.squeeze(), MULT_RANDOM, 4)
        assert_almost_equal(
            res_mult_override.seasonal.values.squeeze(), MULT_SEASONAL, 4
        )
        assert_almost_equal(res_mult_override.trend.values.squeeze(), MULT_TREND, 2)
        assert_almost_equal(res_mult_override.resid.values.squeeze(), MULT_RANDOM, 4)
        assert_equal(res_mult.seasonal.index.values.squeeze(), self.data.index.values)

        res_mult_pi = seasonal_decompose(np.abs(self.data_period_index), "mul", period=4)
        assert_almost_equal(
            res_mult.seasonal.values.squeeze(), res_mult_pi.seasonal.values.squeeze(), 4
        )

    def test_pandas_nofreq(self):
        # issue #3503
        rs = np.random.RandomState(32132100)
        nobs = 100
        dta = pd.Series([x % 3 for x in range(nobs)] + rs.randn(nobs))
        res_np = seasonal_decompose(dta.values, period=3)
        res = seasonal_decompose(dta, period=3)

        atol = 1e-8
        rtol = 1e-10
        assert_allclose(
            res.seasonal.values.squeeze(), res_np.seasonal, atol=atol, rtol=rtol
        )
        assert_allclose(res.trend.values.squeeze(), res_np.trend, atol=atol, rtol=rtol)
        assert_allclose(res.resid.values.squeeze(), res_np.resid, atol=atol, rtol=rtol)

    def test_filt(self):
        filt = np.array([1 / 8.0, 1 / 4.0, 1.0 / 4, 1 / 4.0, 1 / 8.0])
        res_add = seasonal_decompose(self.data.values, filt=filt, period=4)
        assert_almost_equal(res_add.seasonal, SEASONAL, 2)
        assert_almost_equal(res_add.trend, TREND, 2)
        assert_almost_equal(res_add.resid, RANDOM, 3)

    def test_one_sided_moving_average_in_stl_decompose(self):
        res_add = seasonal_decompose(self.data.values, period=4, two_sided=False)

        seasonal = np.array(
            [
                76.76,
                90.03,
                -114.4,
                -52.4,
                76.76,
                90.03,
                -114.4,
                -52.4,
                76.76,
                90.03,
                -114.4,
                -52.4,
                76.76,
                90.03,
                -114.4,
                -52.4,
                76.76,
                90.03,
                -114.4,
                -52.4,
                76.76,
                90.03,
                -114.4,
                -52.4,
                76.76,
                90.03,
                -114.4,
                -52.4,
                76.76,
                90.03,
                -114.4,
                -52.4,
            ]
        )

        trend = np.array(
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                159.12,
                204.0,
                221.25,
                245.12,
                319.75,
                451.5,
                561.12,
                619.25,
                615.62,
                548.0,
                462.12,
                381.12,
                316.62,
                264.0,
                228.38,
                210.75,
                188.38,
                199.0,
                207.12,
                191.0,
                166.88,
                72.0,
                -9.25,
                -33.12,
                -36.75,
                36.25,
                103.0,
                131.62,
            ]
        )

        resid = np.array(
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                11.112,
                -57.031,
                118.147,
                136.272,
                332.487,
                267.469,
                83.272,
                -77.853,
                -152.388,
                -181.031,
                -152.728,
                -152.728,
                -56.388,
                -115.031,
                14.022,
                -56.353,
                -33.138,
                139.969,
                -89.728,
                -40.603,
                -200.638,
                -303.031,
                46.647,
                72.522,
                84.987,
                234.719,
                -33.603,
                104.772,
            ]
        )

        assert_almost_equal(res_add.seasonal, seasonal, 2)
        assert_almost_equal(res_add.trend, trend, 2)
        assert_almost_equal(res_add.resid, resid, 3)

        res_mult = seasonal_decompose(
            np.abs(self.data.values), "mul", period=4, two_sided=False
        )

        seasonal = np.array(
            [
                1.1985,
                1.5449,
                0.5811,
                0.6755,
                1.1985,
                1.5449,
                0.5811,
                0.6755,
                1.1985,
                1.5449,
                0.5811,
                0.6755,
                1.1985,
                1.5449,
                0.5811,
                0.6755,
                1.1985,
                1.5449,
                0.5811,
                0.6755,
                1.1985,
                1.5449,
                0.5811,
                0.6755,
                1.1985,
                1.5449,
                0.5811,
                0.6755,
                1.1985,
                1.5449,
                0.5811,
                0.6755,
            ]
        )

        trend = np.array(
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                171.625,
                204.0,
                221.25,
                245.125,
                319.75,
                451.5,
                561.125,
                619.25,
                615.625,
                548.0,
                462.125,
                381.125,
                316.625,
                264.0,
                228.375,
                210.75,
                188.375,
                199.0,
                207.125,
                191.0,
                166.875,
                107.25,
                80.5,
                79.125,
                78.75,
                116.5,
                140.0,
                157.375,
            ]
        )

        resid = np.array(
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                1.2008,
                0.752,
                1.75,
                1.987,
                1.9023,
                1.1598,
                1.6253,
                1.169,
                0.7319,
                0.5398,
                0.7261,
                0.6837,
                0.888,
                0.586,
                0.9645,
                0.7165,
                1.0276,
                1.3954,
                0.0249,
                0.7596,
                0.215,
                0.851,
                1.646,
                0.2432,
                1.3244,
                2.0058,
                0.5531,
                1.7309,
            ]
        )

        assert_almost_equal(res_mult.seasonal, seasonal, 4)
        assert_almost_equal(res_mult.trend, trend, 2)
        assert_almost_equal(res_mult.resid, resid, 4)

        # test odd
        res_add = seasonal_decompose(self.data.values[:-1], period=4, two_sided=False)
        seasonal = np.array(
            [
                81.21,
                94.48,
                -109.95,
                -65.74,
                81.21,
                94.48,
                -109.95,
                -65.74,
                81.21,
                94.48,
                -109.95,
                -65.74,
                81.21,
                94.48,
                -109.95,
                -65.74,
                81.21,
                94.48,
                -109.95,
                -65.74,
                81.21,
                94.48,
                -109.95,
                -65.74,
                81.21,
                94.48,
                -109.95,
                -65.74,
                81.21,
                94.48,
                -109.95,
            ]
        )

        trend = [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            159.12,
            204.0,
            221.25,
            245.12,
            319.75,
            451.5,
            561.12,
            619.25,
            615.62,
            548.0,
            462.12,
            381.12,
            316.62,
            264.0,
            228.38,
            210.75,
            188.38,
            199.0,
            207.12,
            191.0,
            166.88,
            72.0,
            -9.25,
            -33.12,
            -36.75,
            36.25,
            103.0,
        ]

        random = [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            6.663,
            -61.48,
            113.699,
            149.618,
            328.038,
            263.02,
            78.824,
            -64.507,
            -156.837,
            -185.48,
            -157.176,
            -139.382,
            -60.837,
            -119.48,
            9.574,
            -43.007,
            -37.587,
            135.52,
            -94.176,
            -27.257,
            -205.087,
            -307.48,
            42.199,
            85.868,
            80.538,
            230.27,
            -38.051,
        ]

        assert_almost_equal(res_add.seasonal, seasonal, 2)
        assert_almost_equal(res_add.trend, trend, 2)
        assert_almost_equal(res_add.resid, random, 3)

    def test_2d(self):
        x = np.tile(np.arange(6), (2, 1)).T
        trend = seasonal_decompose(x, period=2).trend
        expected = np.tile(np.arange(6, dtype=float), (2, 1)).T
        expected[0] = expected[-1] = np.nan
        assert_equal(trend, expected)

    def test_interpolate_trend(self):
        x = np.arange(12)
        period = 4
        trend = seasonal_decompose(x, period=period).trend
        assert_equal(trend[0], np.nan)

        trend = seasonal_decompose(x, period=period, extrapolate_trend=5).trend
        assert_almost_equal(trend, x)

        trend = seasonal_decompose(x, period=period, extrapolate_trend="period").trend
        assert_almost_equal(trend, x)

        trend = seasonal_decompose(x[:, None], period=period, extrapolate_trend=5).trend
        assert_almost_equal(trend, x)

        # 2d case
        x = np.tile(np.arange(12), (2, 1)).T
        trend = seasonal_decompose(x, period=period, extrapolate_trend=1).trend
        assert_almost_equal(trend, x)

        trend = seasonal_decompose(x, period=period, extrapolate_trend="period").trend
        assert_almost_equal(trend, x)

    def test_extrapolate_trend_freq_deprecated(self):
        x = np.arange(12)
        period = 4
        with pytest.warns(FutureWarning, match="extrapolate_trend='freq'"):
            trend_freq = seasonal_decompose(
                x, period=period, extrapolate_trend="freq"
            ).trend
        trend_period = seasonal_decompose(
            x, period=period, extrapolate_trend="period"
        ).trend
        assert_almost_equal(trend_freq, trend_period)

    def test_extrapolate_trend_invalid_string_raises(self):
        x = np.arange(12)
        with pytest.raises(ValueError, match="extrapolate_trend"):
            seasonal_decompose(x, period=4, extrapolate_trend="not-a-value")

    def test_raises(self):
        with pytest.raises(ValueError):
            seasonal_decompose(self.data.values)
        with pytest.raises(ValueError):
            seasonal_decompose(self.data, "mul", period=4)
        x = self.data.astype(float).copy()
        x.iloc[2] = np.nan
        with pytest.raises(ValueError):
            seasonal_decompose(x)


def test_seasonal_decompose_too_short():
    rs = np.random.RandomState(43437241)
    dates = pd.date_range("2000-01-31", periods=4, freq=QUARTER_END)
    y = np.sin(np.arange(4) / 4 * 2 * np.pi)
    y += rs.standard_normal(y.size)
    y = pd.Series(y, name="y", index=dates)
    with pytest.raises(ValueError):
        seasonal_decompose(y)

    dates = pd.date_range("2000-01-31", periods=12, freq=MONTH_END)
    y = np.sin(np.arange(12) / 12 * 2 * np.pi)
    y += rs.standard_normal(y.size)
    y = pd.Series(y, name="y", index=dates)
    with pytest.raises(ValueError):
        seasonal_decompose(y)
    with pytest.raises(ValueError):
        seasonal_decompose(y.values, period=12)


@pytest.mark.smoke
def test_seasonal_decompose_smoke():
    x = np.array(
        [
            -50,
            175,
            149,
            214,
            247,
            237,
            225,
            329,
            729,
            809,
            530,
            489,
            540,
            457,
            195,
            176,
            337,
            239,
            128,
            102,
            232,
            429,
            3,
            98,
            43,
            -141,
            -77,
            -13,
            125,
            361,
            -45,
            184,
        ]
    )
    seasonal_decompose(x, period=4)

    data = pd.DataFrame(
        x, pd.date_range(start="1/1/1951", periods=len(x), freq=QUARTER_END)
    )

    seasonal_decompose(data)


def test_seasonal_decompose_multiple():
    x = np.array(
        [
            -50,
            175,
            149,
            214,
            247,
            237,
            225,
            329,
            729,
            809,
            530,
            489,
            540,
            457,
            195,
            176,
            337,
            239,
            128,
            102,
            232,
            429,
            3,
            98,
            43,
            -141,
            -77,
            -13,
            125,
            361,
            -45,
            184,
        ]
    )
    x = np.c_[x, x]
    res = seasonal_decompose(x, period=4)
    assert_allclose(res.trend[:, 0], res.trend[:, 1])
    assert_allclose(res.seasonal[:, 0], res.seasonal[:, 1])
    assert_allclose(res.resid[:, 0], res.resid[:, 1])


def _dataframe_to_decompose(ncols):
    rng = np.random.default_rng(20261130)
    index = pd.date_range("2000-01-01", periods=48, freq="MS")
    t = np.arange(48)[:, None]
    x = 10 + 0.1 * t + np.sin(2 * np.pi * t / 12) + rng.standard_normal((48, ncols))
    columns = ["a", "b", "c"][:ncols]
    return pd.DataFrame(x, index=index, columns=columns)


@pytest.mark.parametrize("extrapolate_trend", [0, "period"])
@pytest.mark.parametrize("model", ["additive", "multiplicative"])
@pytest.mark.parametrize("ncols", [1, 2, 3])
def test_seasonal_decompose_dataframe(ncols, model, extrapolate_trend):
    # A DataFrame with more than one column raised a TypeError when the
    # components were wrapped. Each column is decomposed on its own, as for a
    # 2-d ndarray, and the components keep the index and columns of x. A
    # one-column DataFrame still gives Series named after the component.
    df = _dataframe_to_decompose(ncols)
    kwargs = {"model": model, "extrapolate_trend": extrapolate_trend}
    res = seasonal_decompose(df, **kwargs)
    names = {
        "observed": None,
        "seasonal": "seasonal",
        "trend": "trend",
        "resid": "resid",
    }
    for attr, name in names.items():
        component = getattr(res, attr)
        pd.testing.assert_index_equal(component.index, df.index)
        if ncols == 1:
            assert isinstance(component, pd.Series)
            assert component.name == name
        else:
            assert isinstance(component, pd.DataFrame)
            pd.testing.assert_index_equal(component.columns, df.columns)
        values = np.asarray(component).reshape(48, ncols)
        for i, col in enumerate(df.columns):
            # the period, 12, comes from the monthly index
            single = seasonal_decompose(df[col].to_numpy(), period=12, **kwargs)
            # atol: seasonal and resid are differences of values of order
            # ten, so values near zero carry absolute rounding
            expected = getattr(single, attr)
            assert_allclose(values[:, i], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.thread_unsafe(reason="Uses matplotlib")
@pytest.mark.matplotlib
def test_seasonal_decompose_dataframe_plot(close_figures):
    # a DataFrame result has one seasonal panel per column
    df = _dataframe_to_decompose(2)
    res = seasonal_decompose(df)
    fig = res.plot()
    assert len(fig.axes) == 5
    for ax, col in zip(fig.axes[2:4], df.columns, strict=True):
        ydata = np.asarray(ax.lines[0].get_ydata(), dtype=float)
        assert_allclose(ydata, res.seasonal[col].to_numpy(), rtol=1e-12)


@pytest.mark.thread_unsafe(reason="Uses matplotlib")
@pytest.mark.matplotlib
@pytest.mark.parametrize("model", ["additive", "multiplicative"])
@pytest.mark.parametrize("period", [4, 12])
@pytest.mark.parametrize("two_sided", [True, False])
@pytest.mark.parametrize("extrapolate_trend", [True, False])
def test_seasonal_decompose_plot(
    model, period, two_sided, extrapolate_trend, close_figures
):
    x = np.array(
        [
            -50,
            175,
            149,
            214,
            247,
            237,
            225,
            329,
            729,
            809,
            530,
            489,
            540,
            457,
            195,
            176,
            337,
            239,
            128,
            102,
            232,
            429,
            3,
            98,
            43,
            -141,
            -77,
            -13,
            125,
            361,
            -45,
            184,
        ]
    )
    # a minimum of 1, since the multiplicative model needs positive data
    x -= x.min() - 1
    x2 = np.r_[x[12:], x[:12]]
    x = np.c_[x, x2]
    res = seasonal_decompose(
        x,
        model=model,
        period=period,
        two_sided=two_sided,
        extrapolate_trend=extrapolate_trend,
    )
    # the seasonal component of a cycle averages to 0 for the additive and
    # to 1 for the multiplicative model, in each column. atol since the
    # additive average is zero up to rounding of values of order 100.
    expected = 0.0 if model == "additive" else 1.0
    assert_allclose(
        res.seasonal[:period].mean(axis=0), [expected, expected], atol=1e-10
    )
    fig = res.plot()
    # observed, trend, one seasonal panel per column and residual
    assert len(fig.axes) == 5

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_seasonal_decompose_period_validation():
    # period <= 0 used to crash with ZeroDivisionError when building the
    # filter and negative periods silently produced garbage
    x = np.arange(48.0) + 10 * np.sin(np.arange(48.0) / 3)
    for period in (0, -3, 1):
        with pytest.raises(ValueError, match=f"period must be >= 2, got {period}"):
            seasonal_decompose(x, period=period)
    # a float period with an integral value is still tolerated
    res = seasonal_decompose(x, period=12.0)
    assert_equal(res.trend.shape[0], 48)


def test_period_from_inferred_freq():
    # A DatetimeIndex without a freq gives the period of its inferred freq,
    # without the warning pandas 3.1 gives when inferred_freq is read
    rng = np.random.default_rng(20261121)
    dates = pd.date_range("2000-01-01", periods=48, freq="QS")
    x = pd.Series(10 + rng.standard_normal(48), index=list(dates))
    assert x.index.freq is None
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        res = seasonal_decompose(x)
    assert [str(w.message) for w in record] == []
    expected = seasonal_decompose(x.to_numpy(), period=4)
    assert_allclose(res.seasonal, expected.seasonal, rtol=1e-12)


def _seasonal_series(nobs, period, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(nobs)
    season = np.sin(2 * np.pi * t / period)
    return 10 + 0.1 * t + season + 0.2 * rng.standard_normal(nobs)


@pytest.mark.parametrize("freq, period", [("M", 12), ("Q", 4)])
def test_seasonal_decompose_period_from_period_index(freq, period):
    # without period, a PeriodIndex gives the period of its frequency: 12 for
    # monthly and 4 for quarterly data
    x = _seasonal_series(4 * period, period, 20261125)
    index = pd.period_range("2000-01", periods=x.shape[0], freq=freq)
    res = seasonal_decompose(pd.Series(x, index=index))
    expected = seasonal_decompose(x, period=period)
    for attr in ("seasonal", "trend", "resid"):
        actual = getattr(res, attr).to_numpy()
        assert_allclose(actual, getattr(expected, attr), rtol=1e-12)
    # the seasonal component repeats with that period
    seasonal = res.seasonal.to_numpy()
    assert_allclose(seasonal[period:], seasonal[:-period], rtol=1e-12)
    pd.testing.assert_index_equal(res.seasonal.index, index)


@pytest.mark.parametrize("model", ["additive", "multiplicative"])
@pytest.mark.parametrize("form", SERIES_FORMS + ("int",))
def test_seasonal_decompose_input_forms(form, model):
    # x is documented as array_like: every form of a 1-d series gives the
    # values of the ndarray, and pandas input gives Series with its index
    x = _seasonal_series(60, 12, 20261126)
    forms = series_forms(x)
    forms["int"] = np.round(10 * x).astype(np.int64)
    x_form = forms[form]
    x_float = np.asarray(x_form, dtype=float).squeeze()
    res = seasonal_decompose(x_form, model=model, period=12)
    expected = seasonal_decompose(x_float, model=model, period=12)
    for attr in ("observed", "seasonal", "trend", "resid"):
        actual = getattr(res, attr)
        # atol: seasonal and resid are differences of values of order ten, so
        # values near zero carry absolute rounding
        desired = getattr(expected, attr)
        assert_allclose(np.asarray(actual), desired, rtol=1e-12, atol=1e-12)
        if isinstance(x_form, (pd.Series, pd.DataFrame)):
            assert isinstance(actual, pd.Series)
            pd.testing.assert_index_equal(actual.index, x_form.index)
        else:
            assert isinstance(actual, np.ndarray)
            assert actual.shape == (60,)
    # the components make up the series
    observed = np.asarray(res.observed)
    parts = [np.asarray(res.trend), np.asarray(res.seasonal), np.asarray(res.resid)]
    valid = np.isfinite(parts[0])
    if model == "additive":
        combined = parts[0] + parts[1] + parts[2]
    else:
        combined = parts[0] * parts[1] * parts[2]
    assert_allclose(combined[valid], observed[valid], rtol=1e-12)
    assert_allclose(observed, x_float, rtol=1e-15)


@pytest.mark.parametrize("form", ["list", "tuple", "series"])
def test_seasonal_decompose_filt_forms(form):
    # filt is documented as array_like
    x = _seasonal_series(48, 4, 20261127)
    filt = np.array([1, 2, 2, 2, 1]) / 8.0
    expected = seasonal_decompose(x, filt=filt, period=4)
    res = seasonal_decompose(x, filt=series_forms(filt)[form], period=4)
    for attr in ("seasonal", "trend", "resid"):
        assert_allclose(getattr(res, attr), getattr(expected, attr), rtol=1e-12)


def test_seasonal_decompose_two_column_list():
    # 2-d x has one series per column, also as a nested list
    x = np.c_[_seasonal_series(48, 4, 1), _seasonal_series(48, 4, 2)]
    res = seasonal_decompose(x.tolist(), period=4)
    for i in range(2):
        expected = seasonal_decompose(x[:, i], period=4)
        for attr in ("seasonal", "trend", "resid"):
            actual = getattr(res, attr)[:, i]
            assert_allclose(actual, getattr(expected, attr), rtol=1e-12)


def test_seasonal_mean():
    # the mean of every period-th value, ignoring NaN, for each column of 2-d
    # input
    x = np.tile([1.0, 2.0, 3.0], 4)
    x[4] = np.nan
    assert_allclose(seasonal_mean(x, 3), [1.0, 2.0, 3.0], rtol=1e-15)
    x2 = np.c_[x, 10 * x]
    expected = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
    assert_allclose(seasonal_mean(x2, 3), expected, rtol=1e-15)
    assert_allclose(seasonal_mean(np.arange(6), 2), [2.0, 3.0], rtol=1e-15)


SEASONAL_MEAN_X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0])
# with period 3, the means of positions (0, 3, 6, 9), (1, 4, 7) and (2, 5, 8),
# without the NaN at position 5
SEASONAL_MEAN_EXPECTED = np.array([5.5, 5.0, 6.0])


@pytest.mark.parametrize(
    "form",
    [
        "ndarray",
        "list",
        "tuple",
        "series",
        "series-float-index",
        "dataframe",
        "dataframe-2",
        "ndarray-2d",
        "ndarray-3d",
    ],
)
def test_seasonal_mean_array_like(form):
    # a list, tuple, Series or DataFrame raised an AttributeError or a
    # ValueError from pandas' private nanmean. Time is in the first axis, by
    # position, also for a pandas object with a float index.
    x, expected = SEASONAL_MEAN_X, SEASONAL_MEAN_EXPECTED
    x2, expected2 = np.c_[x, 10 * x], np.c_[expected, 10 * expected]
    index = pd.date_range("2000-01-01", periods=10, freq="MS")
    forms = {
        "ndarray": (x, expected),
        "list": (x.tolist(), expected),
        "tuple": (tuple(x), expected),
        "series": (pd.Series(x, index=index), expected),
        "series-float-index": (pd.Series(x, index=np.arange(10.0)), expected),
        "dataframe": (pd.DataFrame({"a": x}, index=index), expected[:, None]),
        "dataframe-2": (pd.DataFrame(x2, index=index, columns=["a", "b"]), expected2),
        "ndarray-2d": (x2, expected2),
        "ndarray-3d": (x2[:, None, :], expected2[:, None, :]),
    }
    x_in, expected = forms[form]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = seasonal_mean(x_in, 3)
    assert isinstance(res, np.ndarray)
    assert res.shape == expected.shape
    assert_allclose(res, expected, rtol=1e-12)


def test_seasonal_mean_dtype():
    # float32 and complex x keep their dtype, and integer x has float means,
    # as before
    x, expected = SEASONAL_MEAN_X, SEASONAL_MEAN_EXPECTED
    res = seasonal_mean(x.astype(np.float32), 3)
    assert res.dtype == np.float32
    assert_allclose(res, expected, rtol=1e-6)
    res = seasonal_mean(x + 2j * x, 3)
    assert res.dtype == np.complex128
    assert_allclose(res, expected + 2j * expected, rtol=1e-12)
    # the means of positions (0, 3, 6, 9), (1, 4, 7) and (2, 5, 8)
    res = seasonal_mean(np.arange(10), 3)
    assert res.dtype == np.float64
    assert_allclose(res, [4.5, 4.0, 5.0], rtol=1e-12)


def test_seasonal_mean_all_nan_period():
    # a period without any non-NaN value has a NaN mean, without a warning
    x = np.array([1.0, np.nan, 3.0, np.nan, 5.0, np.nan])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = seasonal_mean(x, 2)
    assert_allclose(res, [3.0, np.nan], rtol=1e-12)
