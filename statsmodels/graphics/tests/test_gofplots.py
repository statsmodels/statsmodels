import numpy as np
import numpy.testing as nptest
from numpy.testing import assert_equal
import pytest
from scipy import stats

import statsmodels.api as sm
from statsmodels.graphics import gofplots
from statsmodels.graphics.gofplots import (
    ProbPlot,
    qqline,
    qqplot,
    qqplot_2samples,
)
from statsmodels.graphics.utils import _import_mpl


class BaseProbplotMixin:
    def setup(self):
        try:
            import matplotlib.pyplot as plt

            self.fig, self.ax = plt.subplots()
        except ImportError:
            pass
        self.other_array = np.random.normal(size=self.prbplt.data.shape)
        self.other_prbplot = ProbPlot(self.other_array)
        self.plot_options = dict(
            marker="d",
            markerfacecolor="cornflowerblue",
            markeredgecolor="white",
            alpha=0.5,
        )

    @pytest.mark.matplotlib
    def test_qqplot(self, close_figures):
        self.prbplt.qqplot(ax=self.ax, line=self.line, **self.plot_options)

    @pytest.mark.matplotlib
    def test_ppplot(self, close_figures):
        self.prbplt.ppplot(ax=self.ax, line=self.line)

    @pytest.mark.matplotlib
    def test_probplot(self, close_figures):
        self.prbplt.probplot(ax=self.ax, line=self.line, **self.plot_options)

    @pytest.mark.matplotlib
    def test_probplot_exceed(self, close_figures):
        self.prbplt.probplot(
            ax=self.ax, exceed=True, line=self.line, **self.plot_options
        )

    @pytest.mark.matplotlib
    def test_qqplot_other_array(self, close_figures):
        self.prbplt.qqplot(
            ax=self.ax,
            line=self.line,
            other=self.other_array,
            **self.plot_options,
        )

    @pytest.mark.matplotlib
    def test_ppplot_other_array(self, close_figures):
        self.prbplt.ppplot(
            ax=self.ax,
            line=self.line,
            other=self.other_array,
            **self.plot_options,
        )

    @pytest.mark.xfail(strict=True)
    @pytest.mark.matplotlib
    def test_probplot_other_array(self, close_figures):
        self.prbplt.probplot(
            ax=self.ax,
            line=self.line,
            other=self.other_array,
            **self.plot_options,
        )

    @pytest.mark.matplotlib
    def test_qqplot_other_prbplt(self, close_figures):
        self.prbplt.qqplot(
            ax=self.ax,
            line=self.line,
            other=self.other_prbplot,
            **self.plot_options,
        )

    @pytest.mark.matplotlib
    def test_ppplot_other_prbplt(self, close_figures):
        self.prbplt.ppplot(
            ax=self.ax,
            line=self.line,
            other=self.other_prbplot,
            **self.plot_options,
        )

    @pytest.mark.xfail(strict=True)
    @pytest.mark.matplotlib
    def test_probplot_other_prbplt(self, close_figures):
        self.prbplt.probplot(
            ax=self.ax,
            line=self.line,
            other=self.other_prbplot,
            **self.plot_options,
        )

    @pytest.mark.matplotlib
    def test_qqplot_custom_labels(self, close_figures):
        self.prbplt.qqplot(
            ax=self.ax,
            line=self.line,
            xlabel="Custom X-Label",
            ylabel="Custom Y-Label",
            **self.plot_options,
        )

    @pytest.mark.matplotlib
    def test_ppplot_custom_labels(self, close_figures):
        self.prbplt.ppplot(
            ax=self.ax,
            line=self.line,
            xlabel="Custom X-Label",
            ylabel="Custom Y-Label",
            **self.plot_options,
        )

    @pytest.mark.matplotlib
    def test_probplot_custom_labels(self, close_figures):
        self.prbplt.probplot(
            ax=self.ax,
            line=self.line,
            xlabel="Custom X-Label",
            ylabel="Custom Y-Label",
            **self.plot_options,
        )

    @pytest.mark.matplotlib
    def test_qqplot_pltkwargs(self, close_figures):
        self.prbplt.qqplot(
            ax=self.ax,
            line=self.line,
            marker="d",
            markerfacecolor="cornflowerblue",
            markeredgecolor="white",
            alpha=0.5,
        )

    @pytest.mark.matplotlib
    def test_ppplot_pltkwargs(self, close_figures):
        self.prbplt.ppplot(
            ax=self.ax,
            line=self.line,
            marker="d",
            markerfacecolor="cornflowerblue",
            markeredgecolor="white",
            alpha=0.5,
        )

    @pytest.mark.matplotlib
    def test_probplot_pltkwargs(self, close_figures):
        self.prbplt.probplot(
            ax=self.ax,
            line=self.line,
            marker="d",
            markerfacecolor="cornflowerblue",
            markeredgecolor="white",
            alpha=0.5,
        )

    def test_fit_params(self):
        assert self.prbplt.fit_params[-2] == self.prbplt.loc
        assert self.prbplt.fit_params[-1] == self.prbplt.scale


class TestProbPlotLongelyNoFit(BaseProbplotMixin):
    def setup(self):
        np.random.seed(5)
        self.data = sm.datasets.longley.load()
        self.data.exog = sm.add_constant(self.data.exog, prepend=False)
        self.mod_fit = sm.OLS(self.data.endog, self.data.exog).fit()
        self.prbplt = ProbPlot(
            self.mod_fit.resid, dist=stats.t, distargs=(4,), fit=False
        )
        self.line = "r"
        super().setup()


class TestProbPlotLongelyWithFit(BaseProbplotMixin):
    def setup(self):
        np.random.seed(5)
        self.data = sm.datasets.longley.load(as_pandas=False)
        self.data.exog = sm.add_constant(self.data.exog, prepend=False)
        self.mod_fit = sm.OLS(self.data.endog, self.data.exog).fit()
        self.prbplt = ProbPlot(
            self.mod_fit.resid, dist=stats.t, distargs=(4,), fit=True
        )
        self.line = "r"
        super().setup()


class TestProbPlotRandomNormalMinimal(BaseProbplotMixin):
    def setup(self):
        np.random.seed(5)
        self.data = np.random.normal(loc=8.25, scale=3.25, size=37)
        self.prbplt = ProbPlot(self.data)
        self.line = None
        super(TestProbPlotRandomNormalMinimal, self).setup()


class TestProbPlotRandomNormalWithFit(BaseProbplotMixin):
    def setup(self):
        np.random.seed(5)
        self.data = np.random.normal(loc=8.25, scale=3.25, size=37)
        self.prbplt = ProbPlot(self.data, fit=True)
        self.line = "q"
        super(TestProbPlotRandomNormalWithFit, self).setup()


class TestProbPlotRandomNormalFullDist(BaseProbplotMixin):
    def setup(self):
        np.random.seed(5)
        self.data = np.random.normal(loc=8.25, scale=3.25, size=37)
        self.prbplt = ProbPlot(self.data, dist=stats.norm(loc=8.5, scale=3.0))
        self.line = "45"
        super().setup()

    def test_loc_set(self):
        assert self.prbplt.loc == 8.5

    def test_scale_set(self):
        assert self.prbplt.scale == 3.0

    def test_exceptions(self):
        with pytest.raises(ValueError):
            ProbPlot(self.data, dist=stats.norm(loc=8.5, scale=3.0), fit=True)
        with pytest.raises(ValueError):
            ProbPlot(
                self.data,
                dist=stats.norm(loc=8.5, scale=3.0),
                distargs=(8.5, 3.0),
            )
        with pytest.raises(ValueError):
            ProbPlot(self.data, dist=stats.norm(loc=8.5, scale=3.0), loc=8.5)
        with pytest.raises(ValueError):
            ProbPlot(self.data, dist=stats.norm(loc=8.5, scale=3.0), scale=3.0)


class TestCompareSamplesDifferentSize:
    def setup(self):
        np.random.seed(5)
        self.data1 = ProbPlot(np.random.normal(loc=8.25, scale=3.25, size=37))
        self.data2 = ProbPlot(np.random.normal(loc=8.25, scale=3.25, size=55))

    @pytest.mark.matplotlib
    def test_qqplot(self, close_figures):
        self.data1.qqplot(other=self.data2)
        with pytest.raises(ValueError):
            self.data2.qqplot(other=self.data1)

    @pytest.mark.matplotlib
    def test_ppplot(self, close_figures):
        self.data1.ppplot(other=self.data2)
        self.data2.ppplot(other=self.data1)


class TestProbPlotRandomNormalLocScaleDist(BaseProbplotMixin):
    def setup(self):
        np.random.seed(5)
        self.data = np.random.normal(loc=8.25, scale=3.25, size=37)
        self.prbplt = ProbPlot(self.data, loc=8, scale=3)
        self.line = "45"
        super(TestProbPlotRandomNormalLocScaleDist, self).setup()

    def test_loc_set(self):
        assert self.prbplt.loc == 8

    def test_scale_set(self):
        assert self.prbplt.scale == 3

    def test_loc_set_in_dist(self):
        assert self.prbplt.dist.mean() == 8.0

    def test_scale_set_in_dist(self):
        assert self.prbplt.dist.var() == 9.0


class TestTopLevel:
    def setup(self):
        self.data = sm.datasets.longley.load(as_pandas=False)
        self.data.exog = sm.add_constant(self.data.exog, prepend=False)
        self.mod_fit = sm.OLS(self.data.endog, self.data.exog).fit()
        self.res = self.mod_fit.resid
        self.prbplt = ProbPlot(self.mod_fit.resid, dist=stats.t, distargs=(4,))
        self.other_array = np.random.normal(size=self.prbplt.data.shape)
        self.other_prbplot = ProbPlot(self.other_array)

    @pytest.mark.matplotlib
    def test_qqplot(self, close_figures):
        qqplot(self.res, line="r")

    @pytest.mark.matplotlib
    def test_qqplot_pltkwargs(self, close_figures):
        qqplot(
            self.res,
            line="r",
            marker="d",
            markerfacecolor="cornflowerblue",
            markeredgecolor="white",
            alpha=0.5,
        )

    @pytest.mark.matplotlib
    def test_qqplot_2samples_prob_plot_objects(self, close_figures):
        # also tests all valuesg for line
        for line in ["r", "q", "45", "s"]:
            # test with `ProbPlot` instances
            qqplot_2samples(self.prbplt, self.other_prbplot, line=line)

    @pytest.mark.matplotlib
    def test_qqplot_2samples_arrays(self, close_figures):
        # also tests all values for line
        for line in ["r", "q", "45", "s"]:
            # test with arrays
            qqplot_2samples(self.res, self.other_array, line=line)


def test_invalid_dist_config(close_figures):
    # GH 4226
    np.random.seed(5)
    data = sm.datasets.longley.load(as_pandas=False)
    data.exog = sm.add_constant(data.exog, prepend=False)
    mod_fit = sm.OLS(data.endog, data.exog).fit()
    with pytest.raises(TypeError, match=r"dist\(0, 1, 4, loc=0, scale=1\)"):
        ProbPlot(mod_fit.resid, stats.t, distargs=(0, 1, 4))


@pytest.mark.matplotlib
def test_qqplot_unequal():
    rs = np.random.RandomState(0)
    data1 = rs.standard_normal(100)
    data2 = rs.standard_normal(200)
    fig1 = qqplot_2samples(data1, data2)
    fig2 = qqplot_2samples(data2, data1)
    x1, y1 = fig1.get_axes()[0].get_children()[0].get_data()
    x2, y2 = fig2.get_axes()[0].get_children()[0].get_data()
    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(y1, y2)
    numobj1 = len(fig1.get_axes()[0].get_children())
    numobj2 = len(fig2.get_axes()[0].get_children())
    assert numobj1 == numobj2

    @pytest.mark.matplotlib
    def test_qqplot(self, close_figures):
        qqplot(self.res, line="r")

    @pytest.mark.matplotlib
    def test_qqplot_2samples_prob_plot_obj(self, close_figures):
        # also tests all values for line
        for line in ["r", "q", "45", "s"]:
            # test with `ProbPlot` instances
            qqplot_2samples(self.prbplt, self.other_prbplot, line=line)

    @pytest.mark.matplotlib
    def test_qqplot_2samples_arrays(self, close_figures):
        # also tests all values for line
        for line in ["r", "q", "45", "s"]:
            # test with arrays
            qqplot_2samples(self.res, self.other_array, line=line)


class TestCheckDist:
    def test_good(self):
        gofplots._check_for(stats.norm, "ppf")
        gofplots._check_for(stats.norm, "cdf")

    def test_bad(self):
        with pytest.raises(AttributeError):
            gofplots._check_for("junk", "ppf")
        with pytest.raises(AttributeError):
            gofplots._check_for("junk", "cdf")


class TestDoPlot:
    def setup(self):
        try:
            import matplotlib.pyplot as plt

            self.fig, self.ax = plt.subplots()
        except ImportError:
            pass

        self.x = [0.2, 0.6, 2.0, 4.5, 10.0, 50.0, 83.0, 99.1, 99.7]
        self.y = [1.2, 1.4, 1.7, 2.1, 3.2, 3.7, 4.5, 5.1, 6.3]
        self.full_options = {
            "marker": "s",
            "markerfacecolor": "cornflowerblue",
            "markeredgecolor": "firebrick",
            "markeredgewidth": 1.25,
            "linestyle": "--",
        }
        self.step_options = {"linestyle": "-", "where": "mid"}

    @pytest.mark.matplotlib
    def test_baseline(self, close_figures):
        plt = _import_mpl()
        fig, ax = gofplots._do_plot(self.x, self.y)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert self.fig is not fig
        assert self.ax is not ax

    @pytest.mark.matplotlib
    def test_with_ax(self, close_figures):
        plt = _import_mpl()
        fig, ax = gofplots._do_plot(self.x, self.y, ax=self.ax)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert self.fig is fig
        assert self.ax is ax

    @pytest.mark.matplotlib
    def test_plot_full_options(self, close_figures):
        gofplots._do_plot(
            self.x, self.y, ax=self.ax, step=False, **self.full_options,
        )

    @pytest.mark.matplotlib
    def test_step_baseline(self, close_figures):
        gofplots._do_plot(
            self.x, self.y, ax=self.ax, step=True, **self.step_options,
        )

    @pytest.mark.matplotlib
    def test_step_full_options(self, close_figures):
        gofplots._do_plot(
            self.x, self.y, ax=self.ax, step=True, **self.full_options,
        )

    @pytest.mark.matplotlib
    def test_plot_qq_line(self, close_figures):
        gofplots._do_plot(self.x, self.y, ax=self.ax, line="r")

    @pytest.mark.matplotlib
    def test_step_qq_line(self, close_figures):
        gofplots._do_plot(self.x, self.y, ax=self.ax, step=True, line="r")


class TestQQLine:
    def setup(self):
        np.random.seed(0)
        self.x = np.sort(np.random.normal(loc=2.9, scale=1.2, size=37))
        self.y = np.sort(np.random.normal(loc=3.0, scale=1.1, size=37))
        try:
            import matplotlib.pyplot as plt

            self.fig, self.ax = plt.subplots()
            self.ax.plot(self.x, self.y, "ko")
        except ImportError:
            pass

        self.lineoptions = {
            "linewidth": 2,
            "dashes": (10, 1, 3, 4),
            "color": "green",
        }
        self.fmt = "bo-"

    @pytest.mark.matplotlib
    def test_badline(self):
        with pytest.raises(ValueError):
            qqline(self.ax, "junk")

    @pytest.mark.matplotlib
    def test_non45_no_x(self, close_figures):
        with pytest.raises(ValueError):
            qqline(self.ax, "s", y=self.y)

    @pytest.mark.matplotlib
    def test_non45_no_y(self, close_figures):
        with pytest.raises(ValueError):
            qqline(self.ax, "s", x=self.x)

    @pytest.mark.matplotlib
    def test_non45_no_x_no_y(self, close_figures):
        with pytest.raises(ValueError):
            qqline(self.ax, "s")

    @pytest.mark.matplotlib
    def test_45(self, close_figures):
        nchildren = len(self.ax.get_children())
        qqline(self.ax, "45")
        assert len(self.ax.get_children()) > nchildren

    @pytest.mark.matplotlib
    def test_45_fmt(self, close_figures):
        qqline(self.ax, "45", fmt=self.fmt)

    @pytest.mark.matplotlib
    def test_45_fmt_lineoptions(self, close_figures):
        qqline(self.ax, "45", fmt=self.fmt, **self.lineoptions)

    @pytest.mark.matplotlib
    def test_r(self, close_figures):
        nchildren = len(self.ax.get_children())
        qqline(self.ax, "r", x=self.x, y=self.y)
        assert len(self.ax.get_children()) > nchildren

    @pytest.mark.matplotlib
    def test_r_fmt(self, close_figures):
        qqline(self.ax, "r", x=self.x, y=self.y, fmt=self.fmt)

    @pytest.mark.matplotlib
    def test_r_fmt_lineoptions(self, close_figures):
        qqline(
            self.ax, "r", x=self.x, y=self.y, fmt=self.fmt, **self.lineoptions
        )

    @pytest.mark.matplotlib
    def test_s(self, close_figures):
        nchildren = len(self.ax.get_children())
        qqline(self.ax, "s", x=self.x, y=self.y)
        assert len(self.ax.get_children()) > nchildren

    @pytest.mark.matplotlib
    def test_s_fmt(self, close_figures):
        qqline(self.ax, "s", x=self.x, y=self.y, fmt=self.fmt)

    @pytest.mark.matplotlib
    def test_s_fmt_lineoptions(self, close_figures):
        qqline(
            self.ax, "s", x=self.x, y=self.y, fmt=self.fmt, **self.lineoptions
        )

    @pytest.mark.matplotlib
    def test_q(self, close_figures):
        nchildren = len(self.ax.get_children())
        qqline(self.ax, "q", dist=stats.norm, x=self.x, y=self.y)
        assert len(self.ax.get_children()) > nchildren

    @pytest.mark.matplotlib
    def test_q_fmt(self, close_figures):
        qqline(self.ax, "q", dist=stats.norm, x=self.x, y=self.y, fmt=self.fmt)

    @pytest.mark.matplotlib
    def test_q_fmt_lineoptions(self, close_figures):
        qqline(
            self.ax,
            "q",
            dist=stats.norm,
            x=self.x,
            y=self.y,
            fmt=self.fmt,
            **self.lineoptions,
        )


class TestPlottingPosition:
    def setup(self):
        self.N = 13
        self.data = np.arange(self.N)

    def do_test(self, alpha, beta):
        smpp = gofplots.plotting_pos(self.N, a=alpha, b=beta)
        sppp = stats.mstats.plotting_positions(
            self.data, alpha=alpha, beta=beta
        )

        nptest.assert_array_almost_equal(smpp, sppp, decimal=5)

    @pytest.mark.matplotlib
    def test_weibull(self, close_figures):
        self.do_test(0, 0)

    @pytest.mark.matplotlib
    def test_lininterp(self, close_figures):
        self.do_test(0, 1)

    @pytest.mark.matplotlib
    def test_piecewise(self, close_figures):
        self.do_test(0.5, 0.5)

    @pytest.mark.matplotlib
    def test_approx_med_unbiased(self, close_figures):
        self.do_test(1.0 / 3.0, 1.0 / 3.0)

    @pytest.mark.matplotlib
    def test_cunnane(self, close_figures):
        self.do_test(0.4, 0.4)


def test_param_unpacking():
    expected = np.array([2.0, 3, 0, 1])
    pp = ProbPlot(np.empty(100), dist=stats.beta(2, 3))
    assert_equal(pp.fit_params, expected)
    pp = ProbPlot(np.empty(100), stats.beta(2, b=3))
    assert_equal(pp.fit_params, expected)
    pp = ProbPlot(np.empty(100), stats.beta(a=2, b=3))
    assert_equal(pp.fit_params, expected)

    expected = np.array([2.0, 3, 4, 1])
    pp = ProbPlot(np.empty(100), stats.beta(2, 3, 4))
    assert_equal(pp.fit_params, expected)
    pp = ProbPlot(np.empty(100), stats.beta(a=2, b=3, loc=4))
    assert_equal(pp.fit_params, expected)

    expected = np.array([2.0, 3, 4, 5])
    pp = ProbPlot(np.empty(100), stats.beta(2, 3, 4, 5))
    assert_equal(pp.fit_params, expected)
    pp = ProbPlot(np.empty(100), stats.beta(2, 3, 4, scale=5))
    assert_equal(pp.fit_params, expected)
    pp = ProbPlot(np.empty(100), stats.beta(2, 3, loc=4, scale=5))
    assert_equal(pp.fit_params, expected)
    pp = ProbPlot(np.empty(100), stats.beta(2, b=3, loc=4, scale=5))
    assert_equal(pp.fit_params, expected)
    pp = ProbPlot(np.empty(100), stats.beta(a=2, b=3, loc=4, scale=5))
    assert_equal(pp.fit_params, expected)
