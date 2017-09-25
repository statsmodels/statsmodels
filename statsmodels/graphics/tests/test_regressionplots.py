from statsmodels.compat.testing import skipif
import numpy as np

import statsmodels.api as sm
from numpy.testing import assert_equal, assert_raises
from statsmodels.graphics.regressionplots import (plot_fit, plot_ccpr,
                  plot_partregress, plot_regress_exog, abline_plot,
                  plot_partregress_grid, plot_ccpr_grid, add_lowess,
                  plot_added_variable, plot_partial_residuals,
                  plot_ceres_residuals, influence_plot, plot_leverage_resid2)
from pandas import Series, DataFrame
from numpy.testing.utils import assert_array_less

try:
    import matplotlib.pyplot as plt  #makes plt available for test functions
    have_matplotlib = True
except:
    have_matplotlib = False

pdf_output = False

if pdf_output:
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = PdfPages("test_regressionplots.pdf")
else:
    pdf = None

def close_or_save(pdf, fig):
    if pdf_output:
        pdf.savefig(fig)
    plt.close(fig)

@skipif(not have_matplotlib, reason='matplotlib not available')
def teardown_module():
    plt.close('all')
    if pdf_output:
        pdf.close()

class TestPlot(object):

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

    @skipif(not have_matplotlib, reason='matplotlib not available')
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

        close_or_save(pdf, fig)

    @skipif(not have_matplotlib, reason='matplotlib not available')
    def test_plot_oth(self):
        #just test that they run
        res = self.res
        plt.close('all')
        plot_fit(res, 0, y_true=None)
        plt.close('all')
        plot_partregress_grid(res, exog_idx=[0,1])
        plt.close('all')
        plot_regress_exog(res, exog_idx=0)
        plt.close('all')
        plot_ccpr(res, exog_idx=0)
        plt.close('all')
        plot_ccpr_grid(res, exog_idx=[0])
        plt.close('all')
        fig = plot_ccpr_grid(res, exog_idx=[0,1])
        for ax in fig.axes:
            add_lowess(ax)
   
        close_or_save(pdf, fig)
        plt.close('all')

    @skipif(not have_matplotlib, reason='matplotlib not available')
    def test_plot_influence(self):
        infl = self.res.get_influence()
        fig = influence_plot(self.res)
        assert_equal(isinstance(fig, plt.Figure), True)
        # test that we have the correct criterion for sizes #3103
        try:
            sizes = fig.axes[0].get_children()[0]._sizes
            ex = sm.add_constant(infl.cooks_distance[0])
            ssr = sm.OLS(sizes, ex).fit().ssr
            assert_array_less(ssr, 1e-12)
        except AttributeError:
            import warnings
            warnings.warn('test not compatible with matplotlib version')
        plt.close(fig)

        fig = influence_plot(self.res, criterion='DFFITS')
        assert_equal(isinstance(fig, plt.Figure), True)
        try:
            sizes = fig.axes[0].get_children()[0]._sizes
            ex = sm.add_constant(np.abs(infl.dffits[0]))
            ssr = sm.OLS(sizes, ex).fit().ssr
            assert_array_less(ssr, 1e-12)
        except AttributeError:
            pass
        plt.close(fig)

        assert_raises(ValueError, influence_plot, self.res, criterion='unknown')

    @skipif(not have_matplotlib, reason='matplotlib not available')
    def test_plot_leverage_resid2(self):
        fig = plot_leverage_resid2(self.res)
        assert_equal(isinstance(fig, plt.Figure), True)
        plt.close(fig)


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
        data = DataFrame(exog0, columns=["const", "var1", "var2"])
        data['y'] = y
        self.data = data

class TestPlotFormula(TestPlotPandas):
    @skipif(not have_matplotlib, reason='matplotlib not available')
    def test_one_column_exog(self):
        from statsmodels.formula.api import ols
        res = ols("y~var1-1", data=self.data).fit()
        fig = plot_regress_exog(res, "var1")
        plt.close(fig)
        res = ols("y~var1", data=self.data).fit()
        fig = plot_regress_exog(res, "var1")
        plt.close(fig)


class TestABLine(object):

    @classmethod
    def setup_class(cls):
        np.random.seed(12345)
        X = sm.add_constant(np.random.normal(0, 20, size=30))
        y = np.dot(X, [25, 3.5]) + np.random.normal(0, 30, size=30)
        mod = sm.OLS(y,X).fit()
        cls.X = X
        cls.y = y
        cls.mod = mod

    @skipif(not have_matplotlib, reason='matplotlib not available')
    def test_abline_model(self):
        fig = abline_plot(model_results=self.mod)
        ax = fig.axes[0]
        ax.scatter(self.X[:,1], self.y)
        close_or_save(pdf, fig)

    @skipif(not have_matplotlib, reason='matplotlib not available')
    def test_abline_model_ax(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(self.X[:,1], self.y)
        fig = abline_plot(model_results=self.mod, ax=ax)
        close_or_save(pdf, fig)

    @skipif(not have_matplotlib, reason='matplotlib not available')
    def test_abline_ab(self):
        mod = self.mod
        intercept, slope = mod.params
        fig = abline_plot(intercept=intercept, slope=slope)
        close_or_save(pdf, fig)

    @skipif(not have_matplotlib, reason='matplotlib not available')
    def test_abline_ab_ax(self):
        mod = self.mod
        intercept, slope = mod.params
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(self.X[:,1], self.y)
        fig = abline_plot(intercept=intercept, slope=slope, ax=ax)
        close_or_save(pdf, fig)

    @skipif(not have_matplotlib, reason='matplotlib not available')
    def test_abline_remove(self):
        mod = self.mod
        intercept, slope = mod.params
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(self.X[:,1], self.y)
        abline_plot(intercept=intercept, slope=slope, ax=ax)
        abline_plot(intercept=intercept, slope=2*slope, ax=ax)
        lines = ax.get_lines()
        lines.pop(0).remove()
        close_or_save(pdf, fig)


class TestABLinePandas(TestABLine):
    @classmethod
    def setup_class(cls):
        np.random.seed(12345)
        X = sm.add_constant(np.random.normal(0, 20, size=30))
        y = np.dot(X, [25, 3.5]) + np.random.normal(0, 30, size=30)
        cls.X = X
        cls.y = y
        X = DataFrame(X, columns=["const", "someX"])
        y = Series(y, name="outcome")
        mod = sm.OLS(y,X).fit()
        cls.mod = mod

class TestAddedVariablePlot(object):

    @skipif(not have_matplotlib, reason='matplotlib not available')
    def test_added_variable_poisson(self):

        np.random.seed(3446)

        n = 100
        p = 3
        exog = np.random.normal(size=(n, p))
        lin_pred = 4 + exog[:, 0] + 0.2*exog[:, 1]**2
        expval = np.exp(lin_pred)
        endog = np.random.poisson(expval)

        model = sm.GLM(endog, exog, family=sm.families.Poisson())
        results = model.fit()

        for focus_col in 0, 1, 2:
            for use_glm_weights in False, True:
                for resid_type in "resid_deviance", "resid_response":
                    weight_str = ["Unweighted", "Weighted"][use_glm_weights]

                    # Run directly and called as a results method.
                    for j in 0,1:

                        if j == 0:
                            fig = plot_added_variable(results, focus_col,
                                                      use_glm_weights=use_glm_weights,
                                                      resid_type=resid_type)
                            ti = "Added variable plot"
                        else:
                            fig = results.plot_added_variable(focus_col,
                                                 use_glm_weights=use_glm_weights,
                                                 resid_type=resid_type)
                            ti = "Added variable plot (called as method)"
                        ax = fig.get_axes()[0]

                        add_lowess(ax)
                        ax.set_position([0.1, 0.1, 0.8, 0.7])
                        effect_str = ["Linear effect, slope=1",
                                      "Quadratic effect", "No effect"][focus_col]
                        ti += "\nPoisson regression\n"
                        ti += effect_str + "\n"
                        ti += weight_str + "\n"
                        ti += "Using '%s' residuals" % resid_type
                        ax.set_title(ti)
                        close_or_save(pdf, fig)


class TestPartialResidualPlot(object):

    @skipif(not have_matplotlib, reason='matplotlib not available')
    def test_partial_residual_poisson(self):

        np.random.seed(3446)

        n = 100
        p = 3
        exog = np.random.normal(size=(n, p))
        exog[:, 0] = 1
        lin_pred = 4 + exog[:, 1] + 0.2*exog[:, 2]**2
        expval = np.exp(lin_pred)
        endog = np.random.poisson(expval)

        model = sm.GLM(endog, exog, family=sm.families.Poisson())
        results = model.fit()

        for focus_col in 1, 2:
            for j in 0,1:
                if j == 0:
                    fig = plot_partial_residuals(results, focus_col)
                else:
                    fig = results.plot_partial_residuals(focus_col)
                ax = fig.get_axes()[0]
                add_lowess(ax)
                ax.set_position([0.1, 0.1, 0.8, 0.77])
                effect_str = ["Intercept", "Linear effect, slope=1",
                              "Quadratic effect"][focus_col]
                ti = "Partial residual plot"
                if j == 1:
                    ti += " (called as method)"
                ax.set_title(ti + "\nPoisson regression\n" +
                             effect_str)
                close_or_save(pdf, fig)

class TestCERESPlot(object):

    @skipif(not have_matplotlib, reason='matplotlib not available')
    def test_ceres_poisson(self):

        np.random.seed(3446)

        n = 100
        p = 3
        exog = np.random.normal(size=(n, p))
        exog[:, 0] = 1
        lin_pred = 4 + exog[:, 1] + 0.2*exog[:, 2]**2
        expval = np.exp(lin_pred)
        endog = np.random.poisson(expval)

        model = sm.GLM(endog, exog, family=sm.families.Poisson())
        results = model.fit()

        for focus_col in 1, 2:
            for j in 0, 1:
                if j == 0:
                    fig = plot_ceres_residuals(results, focus_col)
                else:
                    fig = results.plot_ceres_residuals(focus_col)
                ax = fig.get_axes()[0]
                add_lowess(ax)
                ax.set_position([0.1, 0.1, 0.8, 0.77])
                effect_str = ["Intercept", "Linear effect, slope=1",
                              "Quadratic effect"][focus_col]
                ti = "CERES plot"
                if j == 1:
                    ti += " (called as method)"
                ax.set_title(ti + "\nPoisson regression\n" +
                             effect_str)
                close_or_save(pdf, fig)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, '-vvs', '-x', '--pdb'])
