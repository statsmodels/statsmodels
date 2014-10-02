'''
Tests for regressionplots, entire module is skipped
'''

import numpy as np
import nose

import statsmodels.api as sm
from statsmodels.graphics.regressionplots import (plot_fit, plot_ccpr,
                  plot_partregress, plot_regress_exog, abline_plot,
                  plot_partregress_grid, plot_ccpr_grid, add_lowess,
                  covariate_effect_plot)
from pandas import Series, DataFrame
from numpy.testing import dec

# Set to False in master and releases
pdf_output = True

try:
    import matplotlib.pyplot as plt  #makes plt available for test functions
    have_matplotlib = True
except:
    have_matplotlib = False

if pdf_output:
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = PdfPages("test_regressionplots.pdf")
else:
    pdf = None


def setup():
    if not have_matplotlib:
        raise nose.SkipTest('No tests here')

def close_or_save(pdf, fig):
    if pdf_output:
        pdf.savefig(fig)
    else:
        plt.close(fig)

def teardown_module():
    plt.close('all')
    if pdf_output:
        pdf.close()

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

        fig.suptitle("plot_fit")

        close_or_save(pdf, fig)

    def test_plot_oth(self):
        #just test that they run
        res = self.res
        endog = res.model.endog
        exog = res.model.exog

        plot_fit(res, 0, y_true=None)
        plot_partregress_grid(res, exog_idx=[0,1])
        plot_regress_exog(res, exog_idx=0)
        plot_ccpr(res, exog_idx=0)
        plot_ccpr_grid(res, exog_idx=[0])
        fig = plot_ccpr_grid(res, exog_idx=[0,1])
        for ax in fig.axes:
            add_lowess(ax)

        fig.suptitle("plot_oth")

        close_or_save(pdf, fig)

class TestPlotPandas(TestPlot):
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
        exog0 = DataFrame(exog0, columns=["const", "var1", "var2"])
        y = Series(y, name="outcome")
        res = sm.OLS(y, exog0).fit()
        self.res = res

class TestABLine(object):

    @classmethod
    def setupClass(cls):
        np.random.seed(12345)
        X = sm.add_constant(np.random.normal(0, 20, size=30))
        y = np.dot(X, [25, 3.5]) + np.random.normal(0, 30, size=30)
        mod = sm.OLS(y,X).fit()
        cls.X = X
        cls.y = y
        cls.mod = mod

    def test_abline_model(self):
        fig = abline_plot(model_results=self.mod)
        ax = fig.axes[0]
        ax.scatter(self.X[:,1], self.y)
        fig.suptitle("abline_model")
        close_or_save(pdf, fig)

    def test_abline_model_ax(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(self.X[:,1], self.y)
        fig = abline_plot(model_results=self.mod, ax=ax)
        fig.suptitle("abline_model_ax")
        close_or_save(pdf, fig)

    def test_abline_ab(self):
        mod = self.mod
        intercept, slope = mod.params
        fig = abline_plot(intercept=intercept, slope=slope)
        fig.suptitle("abline_ab")
        close_or_save(pdf, fig)

    def test_abline_ab_ax(self):
        mod = self.mod
        intercept, slope = mod.params
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(self.X[:,1], self.y)
        fig = abline_plot(intercept=intercept, slope=slope, ax=ax)
        fig.suptitle("abline_ab_ax")
        close_or_save(pdf, fig)

class TestABLinePandas(TestABLine):
    @classmethod
    def setupClass(cls):
        np.random.seed(12345)
        X = sm.add_constant(np.random.normal(0, 20, size=30))
        y = np.dot(X, [25, 3.5]) + np.random.normal(0, 30, size=30)
        cls.X = X
        cls.y = y
        X = DataFrame(X, columns=["const", "someX"])
        y = Series(y, name="outcome")
        mod = sm.OLS(y,X).fit()
        cls.mod = mod


@dec.skipif(not have_matplotlib)
def test_covariate_effect_plot():

    n = 200
    p = 3
    exog = np.random.normal(size=(n, p))
    exog[:, 0] = 1
    exog[:, 2] = np.random.uniform(-2, 2, size=n)

    lin_pred = 4 - exog[:, 1] + exog[:, 2]
    expval = np.exp(lin_pred)
    endog = np.random.poisson(expval, size=n)
    model = sm.GLM(endog, exog, family=sm.families.Poisson())
    result = model.fit()

    for focus_col in 1, 2:
        effect_type = {1: "True effect is linear (slope = -1)",
                       2: "True effect is linear (slope = 1)"}[focus_col]
        for show_hist in False, True:
            show_hist_str = {True: "Show histogram",
                             False: "No histogram"}[show_hist]
            for summary_value in 0, 1, 2:
                if summary_value == 0:
                    summary_type = None
                    summary_type_str = "Default summaries"
                elif summary_value == 1:
                    summary_type = [0.75, 0.75, 0.75]
                    summary_type_str = "Summarize with 75th percentile"
                elif summary_value == 2:
                    summary_type = [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]
                    summary_type_str = "Summarize with 25th and 75th percentiles"

                fig = result.covariate_effect_plot(focus_col,
                                                   show_hist=show_hist,
                                                   summary_type=summary_type)
                for a in fig.get_axes():
                    a.set_position([0.1, 0.1, 0.8, 0.75])

                if summary_value == 2:
                    ax = fig.get_axes()[0]
                    ha, la = ax.get_legend_handles_labels()
                    la = ["25th pctl", "75th pctl"]
                    title = "Exog %d" % {2: 1, 1: 2}[focus_col]
                    leg = plt.figlegend(ha, la, "center right",
                                        title=title)
                    leg.draw_frame(False)

                    for a in fig.get_axes():
                        a.set_position([0.1, 0.1, 0.68, 0.75])

                ax = fig.get_axes()[0]
                fig.suptitle("covariate_effect_plot")
                ax.set_title(effect_type + "\n" + show_hist_str + "\n" +
                             summary_type_str)
                close_or_save(pdf, fig)
