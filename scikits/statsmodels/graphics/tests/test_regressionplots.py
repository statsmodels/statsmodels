'''Tests for regressionplots, entire module is skipped

'''

import numpy as np
import nose

import scikits.statsmodels.api as sm
from scikits.statsmodels.graphics.regressionplots import (plot_fit, plot_ccpr,
                  plot_partregress, plot_regress_exog )

try:
    import matplotlib.pyplot as plt  #makes plt available for test functions
    have_matplotlib = True
except:
    have_matplotlib = False

def setup():
    if not have_matplotlib:
        raise nose.SkipTest('No tests here')

def teardown_module():
    plt.close('all')

class TestPlot(object):

    def __init__(self):
        self.setup() #temp: for testing without nose

    def setup(self):
        nsample = 100
        sig = 0.5
        x1 = np.linspace(0, 20, nsample)
        x2 = 5 + 3* np.random.randn(nsample)
        X = np.c_[x1, x2, np.sin(0.5*x1), (x2-5)**2, np.ones(nsample)]
        beta = [0.5, 0.5, 1, -0.04, 5.]
        y_true = np.dot(X, beta)
        y = y_true + sig * np.random.normal(size=nsample)
        exog0 = sm.add_constant(np.c_[x1, x2], prepend=False)
        res = sm.OLS(y, exog0).fit()

        self.res = res

    def test_plot_fit(self):
        res = self.res

        fig = plot_fit(res, 0, y_true=None)

        x0 = res.model.exog[:, 0]
        yf = res.fittedvalues
        y = res.model.endog

        px1, px2 = fig.axes[0].get_lines()[0].get_data()
        np.testing.assert_equal(x0, px1)
        np.testing.assert_equal(y, px2)

        px1, px2 = fig.axes[0].get_lines()[1].get_data()
        np.testing.assert_equal(x0, px1)
        np.testing.assert_equal(yf, px2)

        plt.close(fig)

    def test_plot_oth(self):
        #just test that they run
        res = self.res
        endog = res.model.endog
        exog = res.model.exog

        plot_fit(res, 0, y_true=None)
        plot_partregress(endog, exog, exog_idx=[0,1])
        plot_regress_exog(res, exog_idx=[0])
        plot_ccpr(res, exog_idx=[0])
        plot_ccpr(res, exog_idx=[0,1])

        plt.close('all')

