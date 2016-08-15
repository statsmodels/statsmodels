from nose import SkipTest
from nose.tools import assert_raises, assert_equal
import numpy as np
from pandas import Series

from statsmodels.graphics.factorplots import interaction_plot

try:
    import matplotlib.pyplot as plt
    import matplotlib
    have_matplotlib = True
except ImportError:
    have_matplotlib = False


class TestInteractionPlot(object):

    @classmethod
    def setupClass(cls):
        if not have_matplotlib:
            raise SkipTest('matplotlib not available')
        np.random.seed(12345)
        cls.weight = np.random.randint(1,4,size=60)
        cls.duration = np.random.randint(1,3,size=60)
        cls.days = np.log(np.random.randint(1,30, size=60))


    def test_plot_both(self):
        fig = interaction_plot(self.weight, self.duration, self.days,
                 colors=['red','blue'], markers=['D','^'], ms=10)
        plt.close(fig)

    def test_plot_rainbow(self):
        fig = interaction_plot(self.weight, self.duration, self.days,
                 markers=['D','^'], ms=10)
        plt.close(fig)

    def test_plot_pandas(self):
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
        plt.close(fig)


    def test_plot_string_data(self):
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
        plt.close(fig)

    def test_formatting(self):
        fig = interaction_plot(self.weight, self.duration, self.days, colors=['r','g'], linestyles=['--','-.'])
        assert_equal(isinstance(fig, plt.Figure), True)
        plt.close(fig)

    def test_formatting_errors(self):
        #plt.close('all')
        assert_raises(ValueError, interaction_plot, self.weight, self.duration, self.days, markers=['D'])
        plt.close('all')
        assert_raises(ValueError, interaction_plot, self.weight, self.duration, self.days, colors=['b','r','g'])
        plt.close('all')
        assert_raises(ValueError, interaction_plot, self.weight, self.duration, self.days, linestyles=['--','-.',':'])
        plt.close('all')


    def test_plottype(self):
        fig = interaction_plot(self.weight, self.duration, self.days, plottype='line')
        assert_equal(isinstance(fig, plt.Figure), True)
        plt.close(fig)
        fig = interaction_plot(self.weight, self.duration, self.days, plottype='scatter')
        assert_equal(isinstance(fig, plt.Figure), True)
        plt.close(fig)
        assert_raises(ValueError, interaction_plot, self.weight, self.duration, self.days, plottype='unknown')
        plt.close('all')