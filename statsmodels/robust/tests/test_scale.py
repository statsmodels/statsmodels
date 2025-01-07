"""
Test functions for models.robust.scale
"""

import os

import numpy as np
from numpy.random import standard_normal
from numpy.testing import assert_almost_equal, assert_equal, assert_allclose
import pytest

import pandas as pd

from scipy.stats import norm as Gaussian
from scipy import stats

import statsmodels.api as sm
import statsmodels.robust.scale as scale
from statsmodels.robust.scale import mad, scale_tau
import statsmodels.robust.norms as rnorms

cur_dir = os.path.abspath(os.path.dirname(__file__))

file_name = 'hbk.csv'
file_path = os.path.join(cur_dir, 'results', file_name)
dta_hbk = pd.read_csv(file_path)


# Example from Section 5.5, Venables & Ripley (2002)

DECIMAL = 4
# TODO: Can replicate these tests using stackloss data and R if this
#  data is a problem


class TestChem:
    @classmethod
    def setup_class(cls):
        cls.chem = np.array(
            [
                2.20,
                2.20,
                2.4,
                2.4,
                2.5,
                2.7,
                2.8,
                2.9,
                3.03,
                3.03,
                3.10,
                3.37,
                3.4,
                3.4,
                3.4,
                3.5,
                3.6,
                3.7,
                3.7,
                3.7,
                3.7,
                3.77,
                5.28,
                28.95,
            ]
        )

    def test_mean(self):
        assert_almost_equal(np.mean(self.chem), 4.2804, DECIMAL)

    def test_median(self):
        assert_almost_equal(np.median(self.chem), 3.385, DECIMAL)

    def test_mad(self):
        assert_almost_equal(scale.mad(self.chem), 0.52632, DECIMAL)

    def test_iqr(self):
        assert_almost_equal(scale.iqr(self.chem), 0.68570, DECIMAL)

    def test_qn(self):
        assert_almost_equal(scale.qn_scale(self.chem), 0.73231, DECIMAL)

    def test_huber_scale(self):
        assert_almost_equal(scale.huber(self.chem)[0], 3.20549, DECIMAL)

    def test_huber_location(self):
        assert_almost_equal(scale.huber(self.chem)[1], 0.67365, DECIMAL)

    def test_huber_huberT(self):
        n = scale.norms.HuberT()
        n.t = 1.5
        h = scale.Huber(norm=n)
        assert_almost_equal(
            scale.huber(self.chem)[0], h(self.chem)[0], DECIMAL
        )
        assert_almost_equal(
            scale.huber(self.chem)[1], h(self.chem)[1], DECIMAL
        )

    def test_huber_Hampel(self):
        hh = scale.Huber(norm=scale.norms.Hampel())
        assert_almost_equal(hh(self.chem)[0], 3.17434, DECIMAL)
        assert_almost_equal(hh(self.chem)[1], 0.66782, DECIMAL)


class TestMad:
    @classmethod
    def setup_class(cls):
        np.random.seed(54321)
        cls.X = standard_normal((40, 10))

    def test_mad(self):
        m = scale.mad(self.X)
        assert_equal(m.shape, (10,))

    def test_mad_empty(self):
        empty = np.empty(0)
        assert np.isnan(scale.mad(empty))
        empty = np.empty((10, 100, 0))
        assert_equal(scale.mad(empty, axis=1), np.empty((10, 0)))
        empty = np.empty((100, 100, 0, 0))
        assert_equal(scale.mad(empty, axis=-1), np.empty((100, 100, 0)))

    def test_mad_center(self):
        n = scale.mad(self.X, center=0)
        assert_equal(n.shape, (10,))
        with pytest.raises(TypeError):
            scale.mad(self.X, center=None)
        assert_almost_equal(
            scale.mad(self.X, center=1),
            np.median(np.abs(self.X - 1), axis=0) / Gaussian.ppf(3 / 4.0),
            DECIMAL,
        )


class TestMadAxes:
    @classmethod
    def setup_class(cls):
        np.random.seed(54321)
        cls.X = standard_normal((40, 10, 30))

    def test_axis0(self):
        m = scale.mad(self.X, axis=0)
        assert_equal(m.shape, (10, 30))

    def test_axis1(self):
        m = scale.mad(self.X, axis=1)
        assert_equal(m.shape, (40, 30))

    def test_axis2(self):
        m = scale.mad(self.X, axis=2)
        assert_equal(m.shape, (40, 10))

    def test_axisneg1(self):
        m = scale.mad(self.X, axis=-1)
        assert_equal(m.shape, (40, 10))


class TestIqr:
    @classmethod
    def setup_class(cls):
        np.random.seed(54321)
        cls.X = standard_normal((40, 10))

    def test_iqr(self):
        m = scale.iqr(self.X)
        assert_equal(m.shape, (10,))

    def test_iqr_empty(self):
        empty = np.empty(0)
        assert np.isnan(scale.iqr(empty))
        empty = np.empty((10, 100, 0))
        assert_equal(scale.iqr(empty, axis=1), np.empty((10, 0)))
        empty = np.empty((100, 100, 0, 0))
        assert_equal(scale.iqr(empty, axis=-1), np.empty((100, 100, 0)))
        empty = np.empty(shape=())
        with pytest.raises(ValueError):
            scale.iqr(empty)


class TestIqrAxes:
    @classmethod
    def setup_class(cls):
        np.random.seed(54321)
        cls.X = standard_normal((40, 10, 30))

    def test_axis0(self):
        m = scale.iqr(self.X, axis=0)
        assert_equal(m.shape, (10, 30))

    def test_axis1(self):
        m = scale.iqr(self.X, axis=1)
        assert_equal(m.shape, (40, 30))

    def test_axis2(self):
        m = scale.iqr(self.X, axis=2)
        assert_equal(m.shape, (40, 10))

    def test_axisneg1(self):
        m = scale.iqr(self.X, axis=-1)
        assert_equal(m.shape, (40, 10))


class TestQn:
    @classmethod
    def setup_class(cls):
        np.random.seed(54321)
        cls.normal = standard_normal(size=40)
        cls.range = np.arange(0, 40)
        cls.exponential = np.random.exponential(size=40)
        cls.stackloss = sm.datasets.stackloss.load_pandas().data
        cls.sunspot = sm.datasets.sunspots.load_pandas().data.SUNACTIVITY

    def test_qn_naive(self):
        assert_almost_equal(
            scale.qn_scale(self.normal), scale._qn_naive(self.normal), DECIMAL
        )
        assert_almost_equal(
            scale.qn_scale(self.range), scale._qn_naive(self.range), DECIMAL
        )
        assert_almost_equal(
            scale.qn_scale(self.exponential),
            scale._qn_naive(self.exponential),
            DECIMAL,
        )

    def test_qn_robustbase(self):
        # from R's robustbase with finite.corr = FALSE
        assert_almost_equal(scale.qn_scale(self.range), 13.3148, DECIMAL)
        assert_almost_equal(
            scale.qn_scale(self.stackloss),
            np.array([8.87656, 8.87656, 2.21914, 4.43828]),
            DECIMAL,
        )
        # sunspot.year from datasets in R only goes up to 289
        assert_almost_equal(
            scale.qn_scale(self.sunspot[0:289]), 33.50901, DECIMAL
        )

    def test_qn_empty(self):
        empty = np.empty(0)
        assert np.isnan(scale.qn_scale(empty))
        empty = np.empty((10, 100, 0))
        assert_equal(scale.qn_scale(empty, axis=1), np.empty((10, 0)))
        empty = np.empty((100, 100, 0, 0))
        assert_equal(scale.qn_scale(empty, axis=-1), np.empty((100, 100, 0)))
        empty = np.empty(shape=())
        with pytest.raises(ValueError):
            scale.iqr(empty)


class TestQnAxes:
    @classmethod
    def setup_class(cls):
        np.random.seed(54321)
        cls.X = standard_normal((40, 10, 30))

    def test_axis0(self):
        m = scale.qn_scale(self.X, axis=0)
        assert_equal(m.shape, (10, 30))

    def test_axis1(self):
        m = scale.qn_scale(self.X, axis=1)
        assert_equal(m.shape, (40, 30))

    def test_axis2(self):
        m = scale.qn_scale(self.X, axis=2)
        assert_equal(m.shape, (40, 10))

    def test_axisneg1(self):
        m = scale.qn_scale(self.X, axis=-1)
        assert_equal(m.shape, (40, 10))


class TestHuber:
    @classmethod
    def setup_class(cls):
        np.random.seed(54321)
        cls.X = standard_normal((40, 10))

    def test_huber_result_shape(self):
        h = scale.Huber(maxiter=100)
        m, s = h(self.X)
        assert_equal(m.shape, (10,))


class TestHuberAxes:
    @classmethod
    def setup_class(cls):
        np.random.seed(54321)
        cls.X = standard_normal((40, 10, 30))
        cls.h = scale.Huber(maxiter=100, tol=1.0e-05)

    def test_default(self):
        m, s = self.h(self.X, axis=0)
        assert_equal(m.shape, (10, 30))

    def test_axis1(self):
        m, s = self.h(self.X, axis=1)
        assert_equal(m.shape, (40, 30))

    def test_axis2(self):
        m, s = self.h(self.X, axis=2)
        assert_equal(m.shape, (40, 10))

    def test_axisneg1(self):
        m, s = self.h(self.X, axis=-1)
        assert_equal(m.shape, (40, 10))


def test_mad_axis_none():
    # GH 7027
    a = np.array([[0, 1, 2], [2, 3, 2]])

    def m(x):
        return np.median(x)

    direct = mad(a=a, axis=None)
    custom = mad(a=a, axis=None, center=m)
    axis0 = mad(a=a.ravel(), axis=0)

    np.testing.assert_allclose(direct, custom)
    np.testing.assert_allclose(direct, axis0)


def test_tau_scale1():
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1000.0]

    # from R robustbase
    # > scaleTau2(x, mu.too = TRUE, consistency = FALSE)
    res2 = [4.09988889476747, 2.82997006475080]
    res1 = scale_tau(x, normalize=False, ddof=0)
    assert_allclose(res1, res2, rtol=1e-13)

    # > scaleTau2(x, mu.too = TRUE)
    res2 = [4.09988889476747, 2.94291554004125]
    res1 = scale_tau(x, ddof=0)
    assert_allclose(res1, res2, rtol=1e-13)


def test_tau_scale2():
    import pandas as pd
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    file_name = 'hbk.csv'
    file_path = os.path.join(cur_dir, 'results', file_name)
    dta_hbk = pd.read_csv(file_path)

    # from R robustbase
    # > scaleTau2(hbk[,1], mu.too = TRUE, consistency = FALSE)
    # [1] 1.55545438650723 1.93522607240954
    # > scaleTau2(hbk[,2], mu.too = TRUE, consistency = FALSE)
    # [1] 1.87924505206092 1.72121373687210
    # > scaleTau2(hbk[,3], mu.too = TRUE, consistency = FALSE)
    # [1] 1.74163126730520 1.81045973143159
    # > scaleTau2(hbk[,4], mu.too = TRUE, consistency = FALSE)
    # [1] -0.0443521228044396  0.8343974588144727

    res2 = np.array([
        [1.55545438650723, 1.93522607240954],
        [1.87924505206092, 1.72121373687210],
        [1.74163126730520, 1.81045973143159],
        [-0.0443521228044396, 0.8343974588144727]
        ])
    res1 = scale_tau(dta_hbk, normalize=False, ddof=0)
    assert_allclose(np.asarray(res1).T, res2, rtol=1e-13)

    # > scaleTau2(hbk[,1], mu.too = TRUE, consistency = TRUE)
    # [1] 1.55545438650723 2.01246188181448
    # > scaleTau2(hbk[,2], mu.too = TRUE, consistency = TRUE)
    # [1] 1.87924505206092 1.78990821036102
    # > scaleTau2(hbk[,3], mu.too = TRUE, consistency = TRUE)
    # [1] 1.74163126730520 1.88271605576794
    # > scaleTau2(hbk[,4], mu.too = TRUE, consistency = TRUE)
    # [1] -0.0443521228044396  0.8676986653327993

    res2 = np.array([
        [1.55545438650723, 2.01246188181448],
        [1.87924505206092, 1.78990821036102],
        [1.74163126730520, 1.88271605576794],
        [-0.0443521228044396, 0.8676986653327993]
        ])
    res1 = scale_tau(dta_hbk, ddof=0)
    assert_allclose(np.asarray(res1).T, res2, rtol=1e-13)


def test_scale_iter():
    # regression test, and approximately correct
    np.random.seed(54321)
    v = np.array([1, 0.5, 0.4])
    x = standard_normal((40, 3)) * np.sqrt(v)
    x[:2] = [2, 2, 2]

    x = x[:, 0]  # 1d only ?
    v = v[0]

    def meef_scale(x):
        return rnorms.TukeyBiweight().rho(x)

    scale_bias = 0.43684963023076195
    s = scale._scale_iter(x, meef_scale=meef_scale, scale_bias=scale_bias)
    assert_allclose(s, v, rtol=1e-1)
    assert_allclose(s, 1.0683298, rtol=1e-6)  # regression test number

    chi = rnorms.TukeyBiweight()
    scale_bias = 0.43684963023076195
    mscale_biw = scale.MScale(chi, scale_bias)
    s_biw = mscale_biw(x)
    assert_allclose(s_biw, s, rtol=1e-10)

    # regression test with 50% breakdown tuning
    chi = rnorms.TukeyBiweight(c=1.547)
    scale_bias = 0.1995
    mscale_biw = scale.MScale(chi, scale_bias)
    s_biw = mscale_biw(x)
    assert_allclose(s_biw, 1.0326176662, rtol=1e-9)  # regression test number


class TestMScale():

    def test_huber_equivalence(self):
        np.random.seed(54321)
        nobs = 50
        x = 1.5 * standard_normal(nobs)

        # test equivalence of HuberScale and TrimmedMean M-scale
        chi_tm = rnorms.TrimmedMean(c=2.5)
        scale_bias_tm = 0.4887799917273257
        mscale_tm = scale.MScale(chi_tm, scale_bias_tm)
        s_tm = mscale_tm(x)

        mscale_hub = scale.HuberScale()
        s_hub = mscale_hub(nobs, nobs, x)

        assert_allclose(s_tm, s_hub, rtol=1e-6)

    def test_biweight(self):
        y = dta_hbk["Y"].to_numpy()
        ry = y - np.median(y)

        chi = rnorms.TukeyBiweight(c=1.54764)
        scale_bias = 0.19959963130721095
        mscale_biw = scale.MScale(chi, scale_bias)
        scale0 = mscale_biw(ry)
        scale1 = 0.817260483784376   # from R RobStatTM scaleM
        assert_allclose(scale0, scale1, rtol=1e-6)


def test_scale_trimmed_approx():
    scale_trimmed = scale.scale_trimmed  # shorthand
    nobs = 500
    np.random.seed(965578)
    x = 2*np.random.randn(nobs)
    x[:10] = 60

    alpha = 0.2
    res = scale_trimmed(x, alpha)
    assert_allclose(res.scale, 2, rtol=1e-1)
    s = scale_trimmed(np.column_stack((x, 2*x)), alpha).scale
    assert_allclose(s, [2, 4], rtol=1e-1)
    s = scale_trimmed(np.column_stack((x, 2*x)).T, alpha, axis=1).scale
    assert_allclose(s, [2, 4], rtol=1e-1)
    s = scale_trimmed(np.column_stack((x, x)).T, alpha, axis=None).scale
    assert_allclose(s, [2], rtol=1e-1)
    s2 = scale_trimmed(np.column_stack((x, x)).ravel(), alpha).scale
    assert_allclose(s2, [2], rtol=1e-1)
    assert_allclose(s2, s, rtol=1e-1)
    s = scale_trimmed(x, alpha, distr=stats.t, distargs=(100,)).scale
    assert_allclose(s, [2], rtol=1e-1)
