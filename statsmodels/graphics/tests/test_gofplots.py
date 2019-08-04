import numpy as np
import pytest
from scipy import stats

import statsmodels.api as sm


class BaseProbplotMixin(object):
    # TODO: can this be setup_class?  same below
    def setup(self):
        try:
            import matplotlib.pyplot as plt
            self.fig, self.ax = plt.subplots()
        except ImportError:
            pass
        self.other_array = np.random.normal(size=self.prbplt.data.shape)
        self.other_prbplot = sm.ProbPlot(self.other_array)

    @pytest.mark.matplotlib
    def test_qqplot(self, close_figures):
        self.prbplt.qqplot(ax=self.ax, line=self.line)

    @pytest.mark.matplotlib
    def test_ppplot(self, close_figures):
        self.prbplt.ppplot(ax=self.ax, line=self.line)

    @pytest.mark.matplotlib
    def test_probplot(self, close_figures):
        self.prbplt.probplot(ax=self.ax, line=self.line)

    @pytest.mark.matplotlib
    def test_qqplot_other_array(self, close_figures):
        self.prbplt.qqplot(ax=self.ax, line=self.line,
                           other=self.other_array)

    @pytest.mark.matplotlib
    def test_ppplot_other_array(self, close_figures):
        self.prbplt.ppplot(ax=self.ax, line=self.line,
                           other=self.other_array)

    @pytest.mark.xfail(strict=True)
    @pytest.mark.matplotlib
    def test_probplot_other_array(self, close_figures):
        self.prbplt.probplot(ax=self.ax, line=self.line,
                             other=self.other_array)

    @pytest.mark.matplotlib
    def test_qqplot_other_prbplt(self, close_figures):
        self.prbplt.qqplot(ax=self.ax, line=self.line,
                           other=self.other_prbplot)

    @pytest.mark.matplotlib
    def test_ppplot_other_prbplt(self, close_figures):
        self.prbplt.ppplot(ax=self.ax, line=self.line,
                           other=self.other_prbplot)

    @pytest.mark.xfail(strict=True)
    @pytest.mark.matplotlib
    def test_probplot_other_prbplt(self, close_figures):
        self.prbplt.probplot(ax=self.ax, line=self.line,
                             other=self.other_prbplot)

    @pytest.mark.matplotlib
    def test_qqplot_custom_labels(self, close_figures):
        self.prbplt.qqplot(ax=self.ax, line=self.line,
                           xlabel='Custom X-Label',
                           ylabel='Custom Y-Label')

    @pytest.mark.matplotlib
    def test_ppplot_custom_labels(self, close_figures):
        self.prbplt.ppplot(ax=self.ax, line=self.line,
                           xlabel='Custom X-Label',
                           ylabel='Custom Y-Label')

    @pytest.mark.matplotlib
    def test_probplot_custom_labels(self, close_figures):
        self.prbplt.probplot(ax=self.ax, line=self.line,
                             xlabel='Custom X-Label',
                             ylabel='Custom Y-Label')

    @pytest.mark.matplotlib
    def test_qqplot_pltkwargs(self, close_figures):
        self.prbplt.qqplot(ax=self.ax, line=self.line,
                           marker='d',
                           markerfacecolor='cornflowerblue',
                           markeredgecolor='white',
                           alpha=0.5)

    @pytest.mark.matplotlib
    def test_ppplot_pltkwargs(self, close_figures):
        self.prbplt.ppplot(ax=self.ax, line=self.line,
                           marker='d',
                           markerfacecolor='cornflowerblue',
                           markeredgecolor='white',
                           alpha=0.5)

    @pytest.mark.matplotlib
    def test_probplot_pltkwargs(self, close_figures):
        self.prbplt.probplot(ax=self.ax, line=self.line,
                             marker='d',
                             markerfacecolor='cornflowerblue',
                             markeredgecolor='white',
                             alpha=0.5)

    def test_fit_params(self):
        assert self.prbplt.fit_params[-2] == self.prbplt.loc
        assert self.prbplt.fit_params[-1] == self.prbplt.scale


class TestProbPlotLongely(BaseProbplotMixin):
    def setup(self):
        np.random.seed(5)
        self.data = sm.datasets.longley.load(as_pandas=False)
        self.data.exog = sm.add_constant(self.data.exog, prepend=False)
        self.mod_fit = sm.OLS(self.data.endog, self.data.exog).fit()
        self.prbplt = sm.ProbPlot(self.mod_fit.resid, stats.t, distargs=(4,))
        self.line = 'r'
        super(TestProbPlotLongely, self).setup()


class TestProbPlotRandomNormalMinimal(BaseProbplotMixin):
    def setup(self):
        np.random.seed(5)
        self.data = np.random.normal(loc=8.25, scale=3.25, size=37)
        self.prbplt = sm.ProbPlot(self.data)
        self.line = None
        super(TestProbPlotRandomNormalMinimal, self).setup()


class TestProbPlotRandomNormalWithFit(BaseProbplotMixin):
    def setup(self):
        np.random.seed(5)
        self.data = np.random.normal(loc=8.25, scale=3.25, size=37)
        self.prbplt = sm.ProbPlot(self.data, fit=True)
        self.line = 'q'
        super(TestProbPlotRandomNormalWithFit, self).setup()


class TestProbPlotRandomNormalLocScale(BaseProbplotMixin):
    def setup(self):
        np.random.seed(5)
        self.data = np.random.normal(loc=8.25, scale=3.25, size=37)
        self.prbplt = sm.ProbPlot(self.data, loc=8.25, scale=3.25)
        self.line = '45'
        super(TestProbPlotRandomNormalLocScale, self).setup()

    def test_loc_set(self):
        assert self.prbplt.loc == 8.25

    def test_scale_set(self):
        assert self.prbplt.scale == 3.25


class TestCompareSamplesDifferentSize(object):
    def setup(self):
        np.random.seed(5)
        self.data1 = sm.ProbPlot(np.random.normal(loc=8.25, scale=3.25,
                                                  size=37))
        self.data2 = sm.ProbPlot(np.random.normal(loc=8.25, scale=3.25,
                                                  size=55))

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
        self.prbplt = sm.ProbPlot(self.data, loc=8, scale=3)
        self.line = '45'
        super(TestProbPlotRandomNormalLocScaleDist, self).setup()

    def test_loc_set(self):
        assert self.prbplt.loc == 8

    def test_scale_set(self):
        assert self.prbplt.scale == 3

    def test_loc_set_in_dist(self):
        assert self.prbplt.dist.mean() == 8.

    def test_scale_set_in_dist(self):
        assert self.prbplt.dist.var() == 9.


class TestTopLevel(object):
    def setup(self):
        self.data = sm.datasets.longley.load(as_pandas=False)
        self.data.exog = sm.add_constant(self.data.exog, prepend=False)
        self.mod_fit = sm.OLS(self.data.endog, self.data.exog).fit()
        self.res = self.mod_fit.resid
        self.prbplt = sm.ProbPlot(self.mod_fit.resid, stats.t, distargs=(4,))
        self.other_array = np.random.normal(size=self.prbplt.data.shape)
        self.other_prbplot = sm.ProbPlot(self.other_array)

    @pytest.mark.matplotlib
    def test_qqplot(self, close_figures):
        sm.qqplot(self.res, line='r')

    @pytest.mark.matplotlib
    def test_qqplot_pltkwargs(self, close_figures):
        sm.qqplot(self.res, line='r', marker='d',
                  markerfacecolor='cornflowerblue',
                  markeredgecolor='white',
                  alpha=0.5)

    @pytest.mark.matplotlib
    def test_qqplot_2samples_ProbPlotObjects(self, close_figures):
        # also tests all values for line
        for line in ['r', 'q', '45', 's']:
            # test with `ProbPlot` instances
            sm.qqplot_2samples(self.prbplt, self.other_prbplot,
                               line=line)

    @pytest.mark.matplotlib
    def test_qqplot_2samples_arrays(self, close_figures):
        # also tests all values for line
        for line in ['r', 'q', '45', 's']:
            # test with arrays
            sm.qqplot_2samples(self.res, self.other_array, line=line)


def test_invalid_dist_config(close_figures):
    # GH 4226
    np.random.seed(5)
    data = sm.datasets.longley.load(as_pandas=False)
    data.exog = sm.add_constant(data.exog, prepend=False)
    mod_fit = sm.OLS(data.endog, data.exog).fit()
    with pytest.raises(TypeError, match=r'dist\(0, 1, 4, loc=0, scale=1\)'):
        sm.ProbPlot(mod_fit.resid, stats.t, distargs=(0, 1, 4))
