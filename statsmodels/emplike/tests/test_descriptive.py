import itertools
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest

from statsmodels.datasets import star98
from statsmodels.emplike.descriptive import DescStat, EmpLikeTestResult

from .results.el_results import DescStatRes


class GenRes:
    """
    Reads in the data and creates class instance to be tested
    """

    @classmethod
    def setup_class(cls):
        data = star98.load()
        data.exog = np.asarray(data.exog)
        desc_stat_data = data.exog[:50, 5]
        mv_desc_stat_data = data.exog[:50, 5:7]  # mv = multivariate
        cls.res1 = DescStat(desc_stat_data)
        cls.res2 = DescStatRes()
        cls.mvres1 = DescStat(mv_desc_stat_data)


class TestDescriptiveStatistics(GenRes):
    @classmethod
    def setup_class(cls):
        super().setup_class()

    def test_test_mean(self):
        res = self.res1.test_mean(14, result_object=True)
        assert_almost_equal(res[:2], self.res2.test_mean_14, 4)

    def test_test_mean_weights(self):
        assert_almost_equal(
            self.res1.test_mean(14, return_weights=1)[2], self.res2.test_mean_weights, 4
        )

    def test_ci_mean(self):
        assert_almost_equal(self.res1.ci_mean(), self.res2.ci_mean, 4)

    @pytest.mark.thread_unsafe("calculation sets attributes and is not thread safe")
    def test_test_var(self):
        res = self.res1.test_var(3, result_object=True)
        assert_almost_equal(res[:2], self.res2.test_var_3, 4)

    @pytest.mark.thread_unsafe("calculation sets attributes and is not thread safe")
    def test_test_var_weights(self):
        assert_almost_equal(
            self.res1.test_var(3, return_weights=1)[2], self.res2.test_var_weights, 4
        )

    @pytest.mark.thread_unsafe("calculation sets attributes and is not thread safe")
    def test_ci_var(self):
        assert_almost_equal(self.res1.ci_var(), self.res2.ci_var, 4)

    def test_mv_test_mean(self):
        assert_almost_equal(
            self.mvres1.mv_test_mean(np.array([14, 56]), result_object=True)[:2],
            self.res2.mv_test_mean,
            4,
        )

    def test_mv_test_mean_weights(self):
        assert_almost_equal(
            self.mvres1.mv_test_mean(np.array([14, 56]), return_weights=1)[2],
            self.res2.mv_test_mean_wts,
            4,
        )

    @pytest.mark.thread_unsafe("calculation sets attributes and is not thread safe")
    def test_test_skew(self):
        res = self.res1.test_skew(0, result_object=True)
        assert_almost_equal(res[:2], self.res2.test_skew, 4)

    @pytest.mark.thread_unsafe("calculation sets attributes and is not thread safe")
    def test_ci_skew(self):
        # This will be tested in a round about way since MATLAB fails when
        # computing CI with multiple nuisance parameters.  The process is:
        #
        # (1) Get CI for skewness from ci.skew()
        # (2) In MATLAB test the hypotheis that skew=results of test_skew.
        # (3) If p-value approx .05, test confirmed
        skew_ci = self.res1.ci_skew()
        lower_lim = skew_ci[0]
        upper_lim = skew_ci[1]
        ul_pval = self.res1.test_skew(lower_lim, result_object=True).pvalue
        ll_pval = self.res1.test_skew(upper_lim, result_object=True).pvalue
        assert_almost_equal(ul_pval, 0.050000, 4)
        assert_almost_equal(ll_pval, 0.050000, 4)

    @pytest.mark.thread_unsafe("calculation sets attributes and is not thread safe")
    def test_ci_skew_weights(self):
        assert_almost_equal(
            self.res1.test_skew(0, return_weights=1)[2], self.res2.test_skew_wts, 4
        )

    @pytest.mark.thread_unsafe("calculation sets attributes and is not thread safe")
    def test_test_kurt(self):
        res = self.res1.test_kurt(0, result_object=True)
        assert_almost_equal(res[:2], self.res2.test_kurt_0, 4)

    @pytest.mark.thread_unsafe("calculation sets attributes and is not thread safe")
    def test_ci_kurt(self):
        # Same strategy for skewness CI
        kurt_ci = self.res1.ci_kurt(upper_bound=0.5, lower_bound=-1.5)
        lower_lim = kurt_ci[0]
        upper_lim = kurt_ci[1]
        ul_pval = self.res1.test_kurt(upper_lim, result_object=True).pvalue
        ll_pval = self.res1.test_kurt(lower_lim, result_object=True).pvalue
        assert_almost_equal(ul_pval, 0.050000, 4)
        assert_almost_equal(ll_pval, 0.050000, 4)

    @pytest.mark.thread_unsafe("calculation sets attributes and is not thread safe")
    def test_joint_skew_kurt(self):
        assert_almost_equal(
            self.res1.test_joint_skew_kurt(0, 0, result_object=True)[:2],
            self.res2.test_joint_skew_kurt,
            4,
        )

    @pytest.mark.thread_unsafe("calculation sets attributes and is not thread safe")
    def test_test_corr(self):
        res = self.mvres1.test_corr(0.5, result_object=True)
        assert_almost_equal(res[:2], self.res2.test_corr, 4)

    @pytest.mark.thread_unsafe("calculation sets attributes and is not thread safe")
    def test_ci_corr(self):
        corr_ci = self.mvres1.ci_corr()
        lower_lim = corr_ci[0]
        upper_lim = corr_ci[1]
        ul_pval = self.mvres1.test_corr(upper_lim, result_object=True).pvalue
        ll_pval = self.mvres1.test_corr(lower_lim, result_object=True).pvalue
        assert_almost_equal(ul_pval, 0.050000, 4)
        assert_almost_equal(ll_pval, 0.050000, 4)

    @pytest.mark.thread_unsafe("calculation sets attributes and is not thread safe")
    def test_test_corr_weights(self):
        assert_almost_equal(
            self.mvres1.test_corr(0.5, return_weights=1)[2],
            self.res2.test_corr_weights,
            4,
        )

    @pytest.mark.parametrize(
        "endog",
        [
            np.array([]),
            np.zeros((2, 2, 2)),
        ],
    )
    def test_descstat_invalid_input(self, endog):
        with pytest.raises(ValueError):
            DescStat(endog)


@pytest.mark.parametrize(
    ("attr", "args"),
    [
        ("test_mean", (14,)),
        ("test_var", (3,)),
        ("test_skew", (0,)),
        ("test_kurt", (0,)),
        ("test_joint_skew_kurt", (0, 0)),
    ],
)
@pytest.mark.thread_unsafe("calculation sets attributes and is not thread safe")
def test_univariate_namedtuple(attr, args):
    data = np.asarray(star98.load().exog)[:50, 5]
    meth = getattr(DescStat(data), attr)

    # return_weights=True already yields three values, so the NamedTuple is
    # adopted silently
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        full = meth(*args, return_weights=True)
    assert isinstance(full, EmpLikeTestResult)
    assert len(full) == 3

    # the two-value path still warns and still returns a plain tuple
    with pytest.warns(FutureWarning, match=attr):
        legacy = meth(*args)
    assert not isinstance(legacy, EmpLikeTestResult)
    assert len(legacy) == 2
    assert_almost_equal(legacy, full[:2], 10)

    # opting in or out is silent either way
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        opted_in = meth(*args, result_object=True)
        opted_out = meth(*args, result_object=False)
    assert isinstance(opted_in, EmpLikeTestResult)
    assert opted_in.weights is not None
    assert len(opted_out) == 2


@pytest.mark.parametrize(
    ("attr", "args"),
    [
        ("mv_test_mean", (np.array([14, 56]),)),
        ("test_corr", (0.5,)),
    ],
)
@pytest.mark.thread_unsafe("calculation sets attributes and is not thread safe")
def test_multivariate_namedtuple(attr, args):
    data = np.asarray(star98.load().exog)[:50, 5:7]
    meth = getattr(DescStat(data), attr)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        full = meth(*args, return_weights=True)
    assert isinstance(full, EmpLikeTestResult)
    assert len(full) == 3

    with pytest.warns(FutureWarning, match=attr):
        legacy = meth(*args)
    assert len(legacy) == 2
    assert_almost_equal(legacy, full[:2], 10)


@pytest.mark.thread_unsafe(reason="Uses matplotlib and monkeypatches Axes.contour")
@pytest.mark.matplotlib
def test_uv_plot_contour(close_figures, monkeypatch):
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    rng = np.random.default_rng(348203)
    data = rng.standard_normal(150) * 2 + 5
    uv = DescStat(data)
    mean0 = data.mean()
    var0 = data.var()

    captured = {}
    real_contour = Axes.contour

    def fake_contour(self, *args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return real_contour(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "contour", fake_contour)

    fig = uv.plot_contour(mean0 - 3, mean0 + 3, var0 * 0.5, var0 * 1.5, 0.5, 0.5)
    assert isinstance(fig, Figure)

    mu_vect, var_vect, z = captured["args"]
    levels = captured["kwargs"]["levels"]
    # levels are significance levels and must be passed to matplotlib in
    # increasing order -- matplotlib raises ValueError otherwise, which the
    # (.2, .1, .05, .01, .001) default used to trigger.
    assert list(levels) == sorted(levels)

    z = np.asarray(z)
    # z holds p-values (see _opt_var(..., pval=True)), so must lie in [0, 1]
    assert np.all((z >= 0) & (z <= 1))

    # independent recomputation via _opt_var, which is separately exercised
    # (against MATLAB-verified reference values) by test_test_var/test_ci_var
    z_expected = np.empty_like(z)
    for i, sig0 in enumerate(var_vect):
        uv.sig2_0 = sig0
        for j, mu0 in enumerate(mu_vect):
            z_expected[i, j] = uv._opt_var(mu0, pval=True)
    assert_allclose(z, z_expected)

    # the grid point closest to the sample (mean, var) should be the least
    # rejected (highest p-value) point on the grid, and clearly "inside"
    # even the tightest conventional confidence region
    i_star = int(np.argmin(np.abs(np.array(var_vect) - var0)))
    j_star = int(np.argmin(np.abs(np.array(mu_vect) - mean0)))
    assert (i_star, j_star) == np.unravel_index(np.argmax(z), z.shape)
    assert z[i_star, j_star] > 0.5


@pytest.mark.thread_unsafe(reason="Uses matplotlib and monkeypatches Axes.contour")
@pytest.mark.matplotlib
def test_mv_mean_contour(close_figures, monkeypatch):
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    rng = np.random.default_rng(9082)
    cov = np.array([[1.0, 0.3], [0.3, 1.0]])
    data = rng.standard_normal((250, 2)) @ cov + np.array([2.0, -1.0])
    mv = DescStat(data)
    mean_mv = data.mean(0)

    captured = {}
    real_contour = Axes.contour

    def fake_contour(self, *args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return real_contour(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "contour", fake_contour)

    fig = mv.mv_mean_contour(
        mean_mv[0] - 1, mean_mv[0] + 1, mean_mv[1] - 1, mean_mv[1] + 1, 0.25, 0.25
    )
    assert isinstance(fig, Figure)

    x, y, z = captured["args"]
    z = np.asarray(z)
    # z must hold p-values (bounded in [0, 1]), not the unbounded -2 log
    # likelihood ratio statistic -- which can run into the thousands and
    # would make every requested level (<= 0.2) meaningless/degenerate,
    # since it would then only ever be crossed in an infinitesimal region
    # right at the sample mean.
    assert np.all((z >= 0) & (z <= 1))

    # independent recomputation via mv_test_mean, which is separately
    # exercised (against MATLAB-verified reference values) by
    # test_mv_test_mean
    pairs = list(itertools.product(x, y))
    z_expected = np.array(
        [mv.mv_test_mean(np.asarray(i), result_object=True).pvalue for i in pairs]
    )
    X, Y = np.meshgrid(x, y)
    z_expected = z_expected.reshape(X.shape[1], Y.shape[0]).T
    assert_allclose(z, z_expected)

    # the sample mean lies deep inside the confidence region (p ~ 1); a
    # point far away in every direction is clearly excluded (p ~ 0)
    p0 = mv.mv_test_mean(mean_mv, result_object=True).pvalue
    assert p0 > 0.99
    p_far = mv.mv_test_mean(mean_mv + 10, result_object=True).pvalue
    assert p_far < 1e-6
