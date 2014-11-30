import numpy as np
import numpy.testing as nptest

import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot, qqline, ProbPlot
from statsmodels.graphics import gofplots
from scipy import stats
import nose.tools as nt

try:
    import matplotlib.pyplot as plt
    import matplotlib
    have_matplotlib = True
except ImportError:
    have_matplotlib = False


class BaseProbplotMixin(object):
    def base_setup(self):
        if have_matplotlib:
            self.fig, self.ax = plt.subplots()
        self.other_array = np.random.normal(size=self.prbplt.data.shape)
        self.other_prbplot = sm.ProbPlot(self.other_array)
        self.plot_options = dict(
            marker='d',
            markerfacecolor='cornflowerblue',
            markeredgecolor='white',
            alpha=0.5
        )

    def teardown(self):
        if have_matplotlib:
            plt.close('all')

    @nptest.dec.skipif(not have_matplotlib)
    def test_qqplot(self):
        self.fig = self.prbplt.qqplot(ax=self.ax, line=self.line,
                                        plot_options=self.plot_options)

    @nptest.dec.skipif(not have_matplotlib)
    def test_ppplot(self):
        self.fig = self.prbplt.ppplot(ax=self.ax, line=self.line,
                                        plot_options=self.plot_options)

    @nptest.dec.skipif(not have_matplotlib)
    def test_probplot(self):
        self.fig = self.prbplt.probplot(ax=self.ax, line=self.line,
                                        plot_options=self.plot_options)

    @nptest.dec.skipif(not have_matplotlib)
    def test_qqplot_other_array(self):
        self.fig = self.prbplt.qqplot(ax=self.ax, line=self.line,
                                        other=self.other_array,
                                        plot_options=self.plot_options)
    @nptest.dec.skipif(not have_matplotlib)
    def test_ppplot_other_array(self):
        self.fig = self.prbplt.ppplot(ax=self.ax, line=self.line,
                                        other=self.other_array,
                                        plot_options=self.plot_options)
    @nptest.dec.skipif(not have_matplotlib)
    def test_probplot_other_array(self):
        self.fig = self.prbplt.probplot(ax=self.ax, line=self.line,
                                        plot_options=self.plot_options)

    @nptest.dec.skipif(not have_matplotlib)
    def test_qqplot_other_prbplt(self):
        self.fig = self.prbplt.qqplot(ax=self.ax, line=self.line,
                                        other=self.other_prbplot,
                                        plot_options=self.plot_options)
    @nptest.dec.skipif(not have_matplotlib)
    def test_ppplot_other_prbplt(self):
        self.fig = self.prbplt.ppplot(ax=self.ax, line=self.line,
                                        other=self.other_prbplot,
                                        plot_options=self.plot_options)
    @nptest.dec.skipif(not have_matplotlib)
    def test_probplot_other_prbplt(self):
        self.fig = self.prbplt.probplot(ax=self.ax, line=self.line,
                                        plot_options=self.plot_options)

    @nptest.dec.skipif(not have_matplotlib)
    def test_qqplot_custom_labels(self):
        self.fig = self.prbplt.qqplot(ax=self.ax, line=self.line,
                                      xlabel='Custom X-Label',
                                      ylabel='Custom Y-Label',
                                      plot_options=self.plot_options)

    @nptest.dec.skipif(not have_matplotlib)
    def test_ppplot_custom_labels(self):
        self.fig = self.prbplt.ppplot(ax=self.ax, line=self.line,
                                      xlabel='Custom X-Label',
                                      ylabel='Custom Y-Label',
                                      plot_options=self.plot_options)

    @nptest.dec.skipif(not have_matplotlib)
    def test_probplot_custom_labels(self):
        self.fig = self.prbplt.probplot(ax=self.ax, line=self.line,
                                        xlabel='Custom X-Label',
                                        ylabel='Custom Y-Label',
                                        plot_options=self.plot_options)

    @nptest.dec.skipif(not have_matplotlib)
    def test_qqplot_pltkwargs(self):
        self.fig = self.prbplt.qqplot(ax=self.ax, line=self.line,
                                      plot_options=self.plot_options)

    @nptest.dec.skipif(not have_matplotlib)
    def test_ppplot_pltkwargs(self):
        self.fig = self.prbplt.ppplot(ax=self.ax, line=self.line,
                                      plot_options=self.plot_options)

    @nptest.dec.skipif(not have_matplotlib)
    def test_probplot_pltkwargs(self):
        self.fig = self.prbplt.probplot(ax=self.ax, line=self.line,
                                        plot_options=self.plot_options)


class TestProbPlotLongelyNoFit(BaseProbplotMixin):
    def setup(self):
        np.random.seed(5)
        self.data = sm.datasets.longley.load()
        self.data.exog = sm.add_constant(self.data.exog, prepend=False)
        self.mod_fit = sm.OLS(self.data.endog, self.data.exog).fit()
        self.prbplt = sm.ProbPlot(
            self.mod_fit.resid,
            dist=stats.t,
            distargs=(4,),
            fit=False
        )
        self.line = 'r'
        self.base_setup()


class TestProbPlotLongelyWithFit(BaseProbplotMixin):
    def setup(self):
        np.random.seed(5)
        self.data = sm.datasets.longley.load()
        self.data.exog = sm.add_constant(self.data.exog, prepend=False)
        self.mod_fit = sm.OLS(self.data.endog, self.data.exog).fit()
        self.prbplt = sm.ProbPlot(
            self.mod_fit.resid,
            dist=stats.t,
            distargs=(4,),
            fit=True
        )
        self.line = 'r'
        self.base_setup()


class TestProbPlotRandomNormalMinimal(BaseProbplotMixin):
    def setup(self):
        np.random.seed(5)
        self.data = np.random.normal(loc=8.25, scale=3.25, size=37)
        self.prbplt = sm.ProbPlot(self.data)
        self.line = None
        self.base_setup()


class TestProbPlotRandomNormalWithFit(BaseProbplotMixin):
    def setup(self):
        np.random.seed(5)
        self.data = np.random.normal(loc=8.25, scale=3.25, size=37)
        self.prbplt = sm.ProbPlot(self.data, fit=True)
        self.line = 'q'
        self.base_setup()


class TestProbPlotRandomNormalFullDist(BaseProbplotMixin):
    def setup(self):
        np.random.seed(5)
        self.data = np.random.normal(loc=8.25, scale=3.25, size=37)
        self.prbplt = sm.ProbPlot(self.data, dist=stats.norm(loc=8.5, scale=3.0))
        self.line = '45'
        self.base_setup()


class TestTopLevel(object):
    def setup(self):
        self.data = sm.datasets.longley.load()
        self.data.exog = sm.add_constant(self.data.exog, prepend=False)
        self.mod_fit = sm.OLS(self.data.endog, self.data.exog).fit()
        self.res = self.mod_fit.resid
        self.prbplt = sm.ProbPlot(self.mod_fit.resid, dist=stats.t, distargs=(4,))
        self.other_array = np.random.normal(size=self.prbplt.data.shape)
        self.other_prbplot = sm.ProbPlot(self.other_array)

    def teardown(self):
        if have_matplotlib:
            plt.close('all')

    @nptest.dec.skipif(not have_matplotlib)
    def test_qqplot(self):
        fig = sm.qqplot(self.res, line='r')

    @nptest.dec.skipif(not have_matplotlib)
    def test_qqplot_2samples_ProbPlotObjects(self):
        # also tests all values for line
        for line in ['r', 'q', '45', 's']:
            # test with `ProbPlot` instances
            fig = sm.qqplot_2samples(self.prbplt, self.other_prbplot,
                                     line=line)
    @nptest.dec.skipif(not have_matplotlib)
    def test_qqplot_2samples_arrays(self):
        # also tests all values for line
        for line in ['r', 'q', '45', 's']:
            # test with arrays
            fig = sm.qqplot_2samples(self.res, self.other_array, line=line)


class test_check_dist(object):
    def test_good(self):
        gofplots._check_dist(stats.norm)

    @nt.raises(ValueError)
    def test_bad(self):
        gofplots._check_dist('junk')


class test__do_plot(object):
    def setup(self):
        self.fig, self.ax = plt.subplots()
        self.x = [0.2, 0.6, 2.0, 4.5, 10.0, 50.0, 83.0, 99.1, 99.7]
        self.y = [1.2, 1.4, 1.7, 2.1, 3.2, 3.7, 4.5, 5.1, 6.3]
        self.full_options = {
            'marker': 's',
            'markerfacecolor': 'cornflowerblue',
            'markeredgecolor': 'firebrick',
            'markeredgewidth': 1.25,
            'linestyle': '--'
        }
        self.step_options = {
            'linestyle': '-',
            'where': 'mid'
        }

    def teardown(self):
        plt.close('all')

    def test_baseline(self):
        fig, ax = gofplots._do_plot(self.x, self.y)
        nt.assert_true(isinstance(fig, plt.Figure))
        nt.assert_true(isinstance(ax, plt.Axes))
        nt.assert_not_equal(self.fig, fig)
        nt.assert_not_equal(self.ax, ax)

    def test_with_ax(self):
        fig, ax = gofplots._do_plot(self.x, self.y, ax=self.ax)
        nt.assert_true(isinstance(fig, plt.Figure))
        nt.assert_true(isinstance(ax, plt.Axes))
        nt.assert_equal(self.fig, fig)
        nt.assert_equal(self.ax, ax)

    def test_plot_full_options(self):
        fig, ax = gofplots._do_plot(self.x, self.y, ax=self.ax, step=False,
                                    plot_options=self.full_options)

    def test_step_baseline(self):
        fig, ax = gofplots._do_plot(self.x, self.y, ax=self.ax, step=True,
                                    plot_options=self.step_options)

    def test_step_full_options(self):
        fig, ax = gofplots._do_plot(self.x, self.y, ax=self.ax, step=True,
                                    plot_options=self.full_options)

    def test_plot_qq_line(self):
        fig, ax= gofplots._do_plot(self.x, self.y, ax=self.ax, line='r')

    def test_step_qq_line(self):
        fig, ax= gofplots._do_plot(self.x, self.y, ax=self.ax,
                                   step=True, line='r')


class test_qqline(object):
    def setup(self):
        self.fig, self.ax = plt.subplots()

        np.random.seed(0)
        self.x = sorted(np.random.normal(loc=2.9, scale=1.2, size=37))
        self.y = sorted(np.random.normal(loc=3.0, scale=1.1, size=37))
        self.ax.plot(self.x, self.y, 'ko')
        self.lineoptions = {'linewidth': 2, 'dashes': (10, 1 ,3 ,4),
                            'color': 'green'}
        self.fmt = 'bo-'

    @nt.raises(ValueError)
    def test_badline(self):
        qqline(self.ax, 'junk')

    @nt.raises(ValueError)
    def test_non45_no_x(self):
        qqline(self.ax, 's', y=self.y)

    @nt.raises(ValueError)
    def test_non45_no_y(self):
        qqline(self.ax, 's', x=self.x)

    @nt.raises(ValueError)
    def test_non45_no_x_no_y(self):
        qqline(self.ax, 's')

    def teardown(self):
        plt.close('all')

    def test_45(self):
        l = qqline(self.ax, '45')
        nt.assert_true(isinstance(l, plt.Line2D))

    def test_45_fmt(self):
        l = qqline(self.ax, '45', fmt=self.fmt)

    def test_45_fmt_lineoptions(self):
        l = qqline(self.ax, '45', fmt=self.fmt, **self.lineoptions)

    def test_r(self):
        l = qqline(self.ax, 'r', x=self.x, y=self.y)
        nt.assert_true(isinstance(l, plt.Line2D))

    def test_r_fmt(self):
        l = qqline(self.ax, 'r', x=self.x, y=self.y, fmt=self.fmt)

    def test_r_fmt_lineoptions(self):
        l = qqline(self.ax, 'r', x=self.x, y=self.y, fmt=self.fmt, **self.lineoptions)

    def test_s(self):
        l = qqline(self.ax, 's', x=self.x, y=self.y)
        nt.assert_true(isinstance(l, plt.Line2D))

    def test_s_fmt(self):
        l = qqline(self.ax, 's', x=self.x, y=self.y, fmt=self.fmt)

    def test_s_fmt_lineoptions(self):
        l = qqline(self.ax, 's', x=self.x, y=self.y, fmt=self.fmt, **self.lineoptions)

    def test_q(self):
        l = qqline(self.ax, 'q', dist=stats.norm, x=self.x, y=self.y)
        nt.assert_true(isinstance(l, plt.Line2D))

    def test_q_fmt(self):
        l = qqline(self.ax, 'q', dist=stats.norm, x=self.x, y=self.y, fmt=self.fmt)

    def test_q_fmt_lineoptions(self):
        l = qqline(self.ax, 'q', dist=stats.norm, x=self.x, y=self.y, fmt=self.fmt, **self.lineoptions)


class test_plotting_pos(object):
    def setup(self):
        self.N = 13
        self.data = np.arange(self.N)

    @nt.nottest
    def do_test(self, alpha,  beta):
        smpp = gofplots.plotting_pos(self.N, a=alpha, b=beta)
        sppp = stats.mstats.plotting_positions(
            self.data, alpha=alpha, beta=beta
        )

        nptest.assert_array_almost_equal(smpp, sppp, decimal=5)

    def test_weibull(self):
        self.do_test(0, 0)

    def test_lininterp(self):
        self.do_test(0, 1)

    def test_piecewise(self):
        self.do_test(0.5, 0.5)

    def test_approx_med_unbiased(self):
        self.do_test(1./3., 1./3.)

    def test_cunnane(self):
        self.do_test(0.4, 0.4)




