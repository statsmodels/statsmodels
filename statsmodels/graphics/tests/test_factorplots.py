import numpy as np
from nose import SkipTest
from pandas import Series
from statsmodels.graphics.factorplots import interaction_plot

try:
    import matplotlib.pyplot as plt
    import matplotlib
    if matplotlib.__version__ < '1':
        raise
    have_matplotlib = True
except:
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
        assert trace == 'Duration'
        assert ax.get_ylabel() == 'mean of Days'
        assert ax.get_xlabel() == 'Weight'
        plt.close(fig)
