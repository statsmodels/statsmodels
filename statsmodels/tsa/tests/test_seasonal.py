import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_raises
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas import DataFrame, DatetimeIndex


class TestDecompose:
    @classmethod
    def setupClass(cls):
        # even
        data = [-50, 175, 149, 214, 247, 237, 225, 329, 729, 809,
                530, 489, 540, 457, 195, 176, 337, 239, 128, 102,
                232, 429, 3, 98, 43, -141, -77, -13, 125, 361, -45, 184]
        cls.data = DataFrame(data, DatetimeIndex(start='1/1/1951',
                                                 periods=len(data),
                                                 freq='Q'))


    def test_ndarray(self):
        res_add = seasonal_decompose(self.data.values, freq=4)
        seasonal = [62.46, 86.17, -88.38, -60.25, 62.46, 86.17, -88.38,
                    -60.25, 62.46, 86.17, -88.38, -60.25, 62.46, 86.17,
                    -88.38, -60.25, 62.46, 86.17, -88.38, -60.25,
                     62.46, 86.17, -88.38, -60.25, 62.46, 86.17, -88.38,
                    -60.25, 62.46, 86.17, -88.38, -60.25]
        trend = [np.nan, np.nan, 159.12, 204.00, 221.25, 245.12, 319.75,
                 451.50, 561.12, 619.25, 615.62, 548.00, 462.12, 381.12,
                 316.62, 264.00, 228.38, 210.75, 188.38, 199.00, 207.12,
                 191.00, 166.88, 72.00, -9.25, -33.12, -36.75, 36.25,
                 103.00, 131.62, np.nan, np.nan]
        random = [np.nan, np.nan, 78.254, 70.254, -36.710, -94.299, -6.371,
                  -62.246, 105.415, 103.576, 2.754, 1.254, 15.415, -10.299,
                  -33.246, -27.746, 46.165, -57.924, 28.004, -36.746,
                  -37.585, 151.826, -75.496, 86.254, -10.210, -194.049,
                  48.129, 11.004, -40.460, 143.201, np.nan, np.nan]
        assert_almost_equal(res_add.seasonal, seasonal, 2)
        assert_almost_equal(res_add.trend, trend, 2)
        assert_almost_equal(res_add.resid, random, 3)

        res_mult = seasonal_decompose(np.abs(self.data.values), 'm', freq=4)

        seasonal = [1.0815, 1.5538, 0.6716, 0.6931, 1.0815, 1.5538, 0.6716,
                    0.6931, 1.0815, 1.5538, 0.6716, 0.6931, 1.0815, 1.5538,
                    0.6716, 0.6931, 1.0815, 1.5538, 0.6716, 0.6931, 1.0815,
                    1.5538, 0.6716, 0.6931, 1.0815, 1.5538, 0.6716, 0.6931,
                    1.0815, 1.5538, 0.6716, 0.6931]
        trend = [np.nan, np.nan, 171.62, 204.00, 221.25, 245.12, 319.75,
                 451.50, 561.12, 619.25, 615.62, 548.00, 462.12, 381.12,
                 316.62, 264.00, 228.38, 210.75, 188.38, 199.00, 207.12,
                 191.00, 166.88, 107.25, 80.50, 79.12, 78.75, 116.50,
                 140.00, 157.38, np.nan, np.nan]
        random = [np.nan, np.nan, 1.29263, 1.51360, 1.03223, 0.62226,
                  1.04771, 1.05139, 1.20124, 0.84080, 1.28182, 1.28752,
                  1.08043, 0.77172, 0.91697, 0.96191, 1.36441, 0.72986,
                  1.01171, 0.73956, 1.03566, 1.44556, 0.02677, 1.31843,
                  0.49390, 1.14688, 1.45582, 0.16101, 0.82555, 1.47633,
                  np.nan, np.nan]

        assert_almost_equal(res_mult.seasonal, seasonal, 4)
        assert_almost_equal(res_mult.trend, trend, 2)
        assert_almost_equal(res_mult.resid, random, 4)

        # test odd
        res_add = seasonal_decompose(self.data.values[:-1], freq=4)
        seasonal = [68.18, 69.02, -82.66, -54.54, 68.18, 69.02, -82.66,
                    -54.54, 68.18, 69.02, -82.66, -54.54, 68.18, 69.02,
                    -82.66, -54.54, 68.18, 69.02, -82.66, -54.54, 68.18,
                    69.02, -82.66, -54.54, 68.18, 69.02, -82.66, -54.54,
                    68.18, 69.02, -82.66]
        trend = [np.nan, np.nan, 159.12, 204.00, 221.25, 245.12, 319.75,
                 451.50, 561.12, 619.25, 615.62, 548.00, 462.12, 381.12,
                 316.62, 264.00, 228.38, 210.75, 188.38, 199.00, 207.12,
                 191.00, 166.88, 72.00, -9.25, -33.12, -36.75, 36.25,
                 103.00, np.nan, np.nan]
        random = [np.nan, np.nan, 72.538, 64.538, -42.426, -77.150,
                  -12.087, -67.962, 99.699, 120.725, -2.962, -4.462,
                  9.699, 6.850, -38.962, -33.462, 40.449, -40.775, 22.288,
                  -42.462, -43.301, 168.975, -81.212, 80.538, -15.926,
                  -176.900, 42.413, 5.288, -46.176, np.nan, np.nan]
        assert_almost_equal(res_add.seasonal, seasonal, 2)
        assert_almost_equal(res_add.trend, trend, 2)
        assert_almost_equal(res_add.resid, random, 3)

    def test_pandas(self):
        res_add = seasonal_decompose(self.data, freq=4)
        freq_override_data = self.data.copy()
        freq_override_data.index = DatetimeIndex(start='1/1/1951', periods=len(freq_override_data), freq='A')
        res_add_override = seasonal_decompose(freq_override_data, freq=4)
        seasonal = [62.46, 86.17, -88.38, -60.25, 62.46, 86.17, -88.38,
                    -60.25, 62.46, 86.17, -88.38, -60.25, 62.46, 86.17,
                    -88.38, -60.25, 62.46, 86.17, -88.38, -60.25,
                     62.46, 86.17, -88.38, -60.25, 62.46, 86.17, -88.38,
                    -60.25, 62.46, 86.17, -88.38, -60.25]
        trend = [np.nan, np.nan, 159.12, 204.00, 221.25, 245.12, 319.75,
                 451.50, 561.12, 619.25, 615.62, 548.00, 462.12, 381.12,
                 316.62, 264.00, 228.38, 210.75, 188.38, 199.00, 207.12,
                 191.00, 166.88, 72.00, -9.25, -33.12, -36.75, 36.25,
                 103.00, 131.62, np.nan, np.nan]
        random = [np.nan, np.nan, 78.254, 70.254, -36.710, -94.299, -6.371,
                  -62.246, 105.415, 103.576, 2.754, 1.254, 15.415, -10.299,
                  -33.246, -27.746, 46.165, -57.924, 28.004, -36.746,
                  -37.585, 151.826, -75.496, 86.254, -10.210, -194.049,
                  48.129, 11.004, -40.460, 143.201, np.nan, np.nan]
        assert_almost_equal(res_add.seasonal.values.squeeze(), seasonal, 2)
        assert_almost_equal(res_add.trend.values.squeeze(), trend, 2)
        assert_almost_equal(res_add.resid.values.squeeze(), random, 3)
        assert_almost_equal(res_add_override.seasonal.values.squeeze(), seasonal, 2)
        assert_almost_equal(res_add_override.trend.values.squeeze(), trend, 2)
        assert_almost_equal(res_add_override.resid.values.squeeze(), random, 3)
        assert_equal(res_add.seasonal.index.values.squeeze(),
                            self.data.index.values)

        res_mult = seasonal_decompose(np.abs(self.data), 'm', freq=4)
        res_mult_override = seasonal_decompose(np.abs(freq_override_data), 'm', freq=4)
        seasonal = [1.0815, 1.5538, 0.6716, 0.6931, 1.0815, 1.5538, 0.6716,
                    0.6931, 1.0815, 1.5538, 0.6716, 0.6931, 1.0815, 1.5538,
                    0.6716, 0.6931, 1.0815, 1.5538, 0.6716, 0.6931, 1.0815,
                    1.5538, 0.6716, 0.6931, 1.0815, 1.5538, 0.6716, 0.6931,
                    1.0815, 1.5538, 0.6716, 0.6931]
        trend = [np.nan, np.nan, 171.62, 204.00, 221.25, 245.12, 319.75,
                 451.50, 561.12, 619.25, 615.62, 548.00, 462.12, 381.12,
                 316.62, 264.00, 228.38, 210.75, 188.38, 199.00, 207.12,
                 191.00, 166.88, 107.25, 80.50, 79.12, 78.75, 116.50,
                 140.00, 157.38, np.nan, np.nan]
        random = [np.nan, np.nan, 1.29263, 1.51360, 1.03223, 0.62226,
                  1.04771, 1.05139, 1.20124, 0.84080, 1.28182, 1.28752,
                  1.08043, 0.77172, 0.91697, 0.96191, 1.36441, 0.72986,
                  1.01171, 0.73956, 1.03566, 1.44556, 0.02677, 1.31843,
                  0.49390, 1.14688, 1.45582, 0.16101, 0.82555, 1.47633,
                  np.nan, np.nan]

        assert_almost_equal(res_mult.seasonal.values.squeeze(), seasonal, 4)
        assert_almost_equal(res_mult.trend.values.squeeze(), trend, 2)
        assert_almost_equal(res_mult.resid.values.squeeze(), random, 4)
        assert_almost_equal(res_mult_override.seasonal.values.squeeze(), seasonal, 4)
        assert_almost_equal(res_mult_override.trend.values.squeeze(), trend, 2)
        assert_almost_equal(res_mult_override.resid.values.squeeze(), random, 4)
        assert_equal(res_mult.seasonal.index.values.squeeze(),
                            self.data.index.values)


    def test_filt(self):
        filt = np.array([1/8., 1/4., 1./4, 1/4., 1/8.])
        res_add = seasonal_decompose(self.data.values, filt=filt, freq=4)
        seasonal = [62.46, 86.17, -88.38, -60.25, 62.46, 86.17, -88.38,
                    -60.25, 62.46, 86.17, -88.38, -60.25, 62.46, 86.17,
                    -88.38, -60.25, 62.46, 86.17, -88.38, -60.25,
                     62.46, 86.17, -88.38, -60.25, 62.46, 86.17, -88.38,
                    -60.25, 62.46, 86.17, -88.38, -60.25]
        trend = [np.nan, np.nan, 159.12, 204.00, 221.25, 245.12, 319.75,
                 451.50, 561.12, 619.25, 615.62, 548.00, 462.12, 381.12,
                 316.62, 264.00, 228.38, 210.75, 188.38, 199.00, 207.12,
                 191.00, 166.88, 72.00, -9.25, -33.12, -36.75, 36.25,
                 103.00, 131.62, np.nan, np.nan]
        random = [np.nan, np.nan, 78.254, 70.254, -36.710, -94.299, -6.371,
                  -62.246, 105.415, 103.576, 2.754, 1.254, 15.415, -10.299,
                  -33.246, -27.746, 46.165, -57.924, 28.004, -36.746,
                  -37.585, 151.826, -75.496, 86.254, -10.210, -194.049,
                  48.129, 11.004, -40.460, 143.201, np.nan, np.nan]
        assert_almost_equal(res_add.seasonal, seasonal, 2)
        assert_almost_equal(res_add.trend, trend, 2)
        assert_almost_equal(res_add.resid, random, 3)

    def test_one_sided_moving_average_in_stl_decompose(self):
        res_add = seasonal_decompose(self.data.values, freq=4, two_sided=False)

        seasonal = np.array([76.76, 90.03, -114.4, -52.4, 76.76, 90.03, -114.4,
                             -52.4, 76.76, 90.03, -114.4, -52.4, 76.76, 90.03,
                             -114.4, -52.4, 76.76, 90.03, -114.4, -52.4, 76.76,
                             90.03, -114.4, -52.4, 76.76, 90.03, -114.4, -52.4,
                             76.76, 90.03, -114.4, -52.4])

        trend = np.array([np.nan, np.nan, np.nan, np.nan, 159.12, 204., 221.25,
                          245.12, 319.75, 451.5, 561.12, 619.25, 615.62, 548.,
                          462.12, 381.12, 316.62, 264., 228.38, 210.75, 188.38,
                          199., 207.12, 191., 166.88, 72., -9.25, -33.12,
                          -36.75, 36.25, 103., 131.62])

        resid = np.array([np.nan, np.nan, np.nan, np.nan, 11.112, -57.031,
                          118.147, 136.272, 332.487, 267.469, 83.272, -77.853,
                          -152.388, -181.031, -152.728, -152.728, -56.388, -115.031,
                          14.022, -56.353, -33.138, 139.969, -89.728, -40.603,
                          -200.638, -303.031, 46.647, 72.522, 84.987, 234.719,
                          -33.603, 104.772])

        assert_almost_equal(res_add.seasonal, seasonal, 2)
        assert_almost_equal(res_add.trend, trend, 2)
        assert_almost_equal(res_add.resid, resid, 3)

        res_mult = seasonal_decompose(np.abs(self.data.values), 'm', freq=4, two_sided=False)

        seasonal = np.array([1.1985, 1.5449, 0.5811, 0.6755, 1.1985, 1.5449, 0.5811,
                             0.6755, 1.1985, 1.5449, 0.5811, 0.6755, 1.1985, 1.5449,
                             0.5811, 0.6755, 1.1985, 1.5449, 0.5811, 0.6755, 1.1985,
                             1.5449, 0.5811, 0.6755, 1.1985, 1.5449, 0.5811, 0.6755,
                             1.1985, 1.5449, 0.5811, 0.6755])

        trend = np.array([np.nan, np.nan, np.nan, np.nan, 171.625, 204.,
                          221.25, 245.125, 319.75, 451.5, 561.125, 619.25,
                          615.625, 548., 462.125, 381.125, 316.625, 264.,
                          228.375, 210.75, 188.375, 199., 207.125, 191.,
                          166.875, 107.25, 80.5, 79.125, 78.75, 116.5,
                          140., 157.375])

        resid = np.array([np.nan, np.nan, np.nan, np.nan, 1.2008, 0.752, 1.75,
                          1.987, 1.9023, 1.1598, 1.6253, 1.169, 0.7319, 0.5398,
                          0.7261, 0.6837, 0.888, 0.586, 0.9645, 0.7165, 1.0276,
                          1.3954, 0.0249, 0.7596, 0.215, 0.851, 1.646, 0.2432,
                          1.3244, 2.0058, 0.5531, 1.7309])

        assert_almost_equal(res_mult.seasonal, seasonal, 4)
        assert_almost_equal(res_mult.trend, trend, 2)
        assert_almost_equal(res_mult.resid, resid, 4)

        # test odd
        res_add = seasonal_decompose(self.data.values[:-1], freq=4, two_sided=False)
        seasonal = np.array([81.21, 94.48, -109.95, -65.74, 81.21, 94.48, -109.95,
                             -65.74, 81.21, 94.48, -109.95, -65.74, 81.21, 94.48,
                             -109.95, -65.74, 81.21, 94.48, -109.95, -65.74, 81.21,
                             94.48, -109.95, -65.74, 81.21, 94.48, -109.95, -65.74,
                             81.21, 94.48, -109.95])

        trend = [np.nan, np.nan, np.nan, np.nan, 159.12, 204., 221.25,
                 245.12, 319.75, 451.5, 561.12, 619.25, 615.62, 548.,
                 462.12, 381.12, 316.62, 264., 228.38, 210.75, 188.38,
                 199., 207.12, 191., 166.88, 72., -9.25, -33.12,
                 -36.75, 36.25, 103.]

        random = [np.nan, np.nan, np.nan, np.nan, 6.663, -61.48,
                  113.699, 149.618, 328.038, 263.02, 78.824, -64.507,
                  -156.837, -185.48, -157.176, -139.382, -60.837, -119.48,
                  9.574, -43.007, -37.587, 135.52, -94.176, -27.257,
                  -205.087, -307.48, 42.199, 85.868, 80.538, 230.27, -38.051]

        assert_almost_equal(res_add.seasonal, seasonal, 2)
        assert_almost_equal(res_add.trend, trend, 2)
        assert_almost_equal(res_add.resid, random, 3)


    def test_raises(self):
        assert_raises(ValueError, seasonal_decompose, self.data.values)
        assert_raises(ValueError, seasonal_decompose, self.data, 'm',
                      freq=4)
        x = self.data.astype(float).copy()
        x.ix[2] = np.nan
        assert_raises(ValueError, seasonal_decompose, x)


