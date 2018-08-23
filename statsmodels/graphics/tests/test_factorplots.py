import numpy as np
from numpy.testing import assert_raises, assert_equal
from pandas import Series
import pytest

from statsmodels.graphics.factorplots import interaction_plot

try:
    import matplotlib.pyplot as plt
    import matplotlib
    have_matplotlib = True
except ImportError:
    have_matplotlib = False


@pytest.mark.skipif(not have_matplotlib, reason='matplotlib not available')
class TestInteractionPlot(object):

    @classmethod
    def setup_class(cls):
        if not have_matplotlib:
            pytest.skip('matplotlib not available')
        np.random.seed(12345)
        cls.weight = np.random.randint(1,4,size=60)
        cls.duration = np.random.randint(1,3,size=60)
        cls.days = np.log(np.random.randint(1,30, size=60))


    def test_plot_both(self, close_figures):
        fig = interaction_plot(self.weight, self.duration, self.days,
                 colors=['red','blue'], markers=['D','^'], ms=10)

    def test_plot_rainbow(self, close_figures):
        fig = interaction_plot(self.weight, self.duration, self.days,
                 markers=['D','^'], ms=10)

    def test_plot_pandas(self, close_figures):
        weight = Series(self.weight, name='Weight')
        duration = Series(self.duration, name='Duration')
        days = Series(self.days, name='Days')
        fig = interaction_plot(weight, duration, days,
                 markers=['D','^'], ms=10)
        ax = fig.axes[0]
        trace = ax.get_legend().get_title().get_text()
        assert_equal(trace, 'Duration')
        assert_equal(ax.get_ylabel(), 'mean of Days')
        assert_equal(ax.get_xlabel(), 'Weight')

    def test_plot_string_data(self, close_figures):
        weight = Series(self.weight, name='Weight').astype('str')
        duration = Series(self.duration, name='Duration')
        days = Series(self.days, name='Days')
        fig = interaction_plot(weight, duration, days,
                               markers=['D', '^'], ms=10)
        ax = fig.axes[0]
        trace = ax.get_legend().get_title().get_text()
        assert_equal(trace, 'Duration')
        assert_equal(ax.get_ylabel(), 'mean of Days')
        assert_equal(ax.get_xlabel(), 'Weight')

    def test_formatting(self, close_figures):
        fig = interaction_plot(self.weight, self.duration, self.days, colors=['r','g'], linestyles=['--','-.'])
        assert_equal(isinstance(fig, plt.Figure), True)

    def test_formatting_errors(self, close_figures):
        assert_raises(ValueError, interaction_plot, self.weight, self.duration, self.days, markers=['D'])
        assert_raises(ValueError, interaction_plot, self.weight, self.duration, self.days, colors=['b','r','g'])
        assert_raises(ValueError, interaction_plot, self.weight, self.duration, self.days, linestyles=['--','-.',':'])

    def test_plottype(self, close_figures):
        fig = interaction_plot(self.weight, self.duration, self.days, plottype='line')
        assert_equal(isinstance(fig, plt.Figure), True)
        fig = interaction_plot(self.weight, self.duration, self.days, plottype='scatter')
        assert_equal(isinstance(fig, plt.Figure), True)
        assert_raises(ValueError, interaction_plot, self.weight, self.duration, self.days, plottype='unknown')
