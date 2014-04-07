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
        assert_equal(res_add.seasonal.index.values.squeeze(),
                            self.data.index.values)

        res_mult = seasonal_decompose(np.abs(self.data), 'm', freq=4)

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

    def test_raises(self):
        assert_raises(ValueError, seasonal_decompose, self.data.values)
        assert_raises(ValueError, seasonal_decompose, self.data, 'm',
                      freq=4)
        x = self.data.astype(float).copy()
        x.ix[2] = np.nan
        assert_raises(ValueError, seasonal_decompose, x)


