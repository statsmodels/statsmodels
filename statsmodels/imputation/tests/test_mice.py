import numpy as np
import pandas as pd
from statsmodels.imputation import mice
import statsmodels.api as sm
from numpy.testing import assert_equal, assert_allclose, dec

try:
    import matplotlib.pyplot as plt  #makes plt available for test functions
    have_matplotlib = True
except:
    have_matplotlib = False

pdf_output = False


if pdf_output:
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = PdfPages("test_mice.pdf")
else:
    pdf = None


def close_or_save(pdf, fig):
    if not have_matplotlib:
        return
    if pdf_output:
        pdf.savefig(fig)
    plt.close(fig)


def teardown_module():
    if not have_matplotlib:
        return
    plt.close('all')
    if pdf_output:
        pdf.close()


def gendat():
    """
    Create a data set with missing values.
    """

    np.random.seed(34243)

    n = 200
    p = 5

    exog = np.random.normal(size=(n, p))
    exog[:, 0] = exog[:, 1] - exog[:, 2] + 2*exog[:, 4]
    exog[:, 0] += np.random.normal(size=n)
    exog[:, 2] = 1*(exog[:, 2] > 0)

    endog = exog.sum(1) + np.random.normal(size=n)

    df = pd.DataFrame(exog)
    df.columns = ["x%d" % k for k in range(1, p+1)]

    df["y"] = endog

    df.x1[0:60] = np.nan
    df.x2[0:40] = np.nan
    df.x3[10:30:2] = np.nan
    df.x4[20:50:3] = np.nan
    df.x5[40:45] = np.nan
    df.y[30:100:2] = np.nan

    return df


class TestMICEData(object):

    def test_default(self):
        """
        Test with all defaults.
        """

        df = gendat()
        orig = df.copy()
        mx = pd.notnull(df)
        imp_data = mice.MICEData(df)
        nrow, ncol = df.shape

        assert_allclose(imp_data.ix_miss['x1'], np.arange(60))
        assert_allclose(imp_data.ix_obs['x1'], np.arange(60, 200))
        assert_allclose(imp_data.ix_miss['x2'], np.arange(40))
        assert_allclose(imp_data.ix_miss['x3'], np.arange(10, 30, 2))
        assert_allclose(imp_data.ix_obs['x3'],
                        np.concatenate((np.arange(10),
                                        np.arange(11, 30, 2),
                                        np.arange(30, 200))))

        for k in range(3):
            imp_data.update_all()
            assert_equal(imp_data.data.shape[0], nrow)
            assert_equal(imp_data.data.shape[1], ncol)
            assert_allclose(orig[mx], imp_data.data[mx])

        fml = 'x1 ~ x2 + x3 + x4 + x5 + y'
        assert_equal(imp_data.conditional_formula['x1'], fml)

        assert_equal(imp_data._cycle_order, ['x5', 'x3', 'x4', 'y', 'x2', 'x1'])

        # Should make a copy
        assert(not (df is imp_data.data))

        (endog_obs, exog_obs, exog_miss,
         predict_obs_kwds, predict_miss_kwds) = imp_data.get_split_data('x3')
        assert_equal(len(endog_obs), 190)
        assert_equal(exog_obs.shape, [190, 6])
        assert_equal(exog_miss.shape, [10, 6])


    def test_next_sample(self):

        df = gendat()
        imp_data = mice.MICEData(df)

        all_x = []
        for j in range(2):
            x = imp_data.next_sample()
            assert(isinstance(x, pd.DataFrame))
            assert_equal(df.shape, x.shape)
            all_x.append(x)

        # The returned dataframes are all the same object
        assert(all_x[0] is all_x[1])


    def test_pertmeth(self):
        """
        Test with specified perturbation method.
        """

        df = gendat()
        orig = df.copy()
        mx = pd.notnull(df)
        nrow, ncol = df.shape

        for pert_meth in "gaussian", "boot":

            imp_data = mice.MICEData(df, perturbation_method=pert_meth)

            for k in range(2):
                imp_data.update_all()
                assert_equal(imp_data.data.shape[0], nrow)
                assert_equal(imp_data.data.shape[1], ncol)
                assert_allclose(orig[mx], imp_data.data[mx])

        assert_equal(imp_data._cycle_order, ['x5', 'x3', 'x4', 'y', 'x2', 'x1'])


    def test_phreg(self):

        np.random.seed(8742)
        n = 300
        x1 = np.random.normal(size=n)
        x2 = np.random.normal(size=n)
        event_time = np.random.exponential(size=n) * np.exp(x1)
        obs_time = np.random.exponential(size=n)
        time = np.where(event_time < obs_time, event_time, obs_time)
        status = np.where(time == event_time, 1, 0)
        df = pd.DataFrame({"time": time, "status": status, "x1": x1, "x2": x2})
        df.loc[10:40, 'time'] = np.nan
        df.loc[10:40, 'status'] = np.nan
        df.loc[30:50, 'x1'] = np.nan
        df.loc[40:60, 'x2'] = np.nan

        from statsmodels.duration.hazard_regression import PHReg

        idata = mice.MICEData(df)
        idata.set_imputer("time", "0 + x1 + x2", model_class=PHReg,
                          init_kwds={"status": mice.PatsyFormula("status")},
                          predict_kwds={"pred_type": "hr"})

        x = idata.next_sample()
        assert(isinstance(x, pd.DataFrame))


    def test_set_imputer(self):
        """
        Test with specified perturbation method.
        """

        from statsmodels.regression.linear_model import RegressionResultsWrapper
        from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper

        df = gendat()
        orig = df.copy()
        mx = pd.notnull(df)
        nrow, ncol = df.shape

        imp_data = mice.MICEData(df)
        imp_data.set_imputer('x1', 'x3 + x4 + x3*x4')
        imp_data.set_imputer('x2', 'x4 + I(x5**2)')
        imp_data.set_imputer('x3', model_class=sm.GLM,
                             init_kwds={"family": sm.families.Binomial()})

        imp_data.update_all()
        assert_equal(imp_data.data.shape[0], nrow)
        assert_equal(imp_data.data.shape[1], ncol)
        assert_allclose(orig[mx], imp_data.data[mx])
        for j in range(1, 6):
            if j == 3:
                assert_equal(isinstance(imp_data.models['x3'], sm.GLM), True)
                assert_equal(isinstance(imp_data.models['x3'].family, sm.families.Binomial), True)
                assert_equal(isinstance(imp_data.results['x3'], GLMResultsWrapper), True)
            else:
                assert_equal(isinstance(imp_data.models['x%d' % j], sm.OLS), True)
                assert_equal(isinstance(imp_data.results['x%d' % j], RegressionResultsWrapper), True)

        fml = 'x1 ~ x3 + x4 + x3*x4'
        assert_equal(imp_data.conditional_formula['x1'], fml)

        fml = 'x4 ~ x1 + x2 + x3 + x5 + y'
        assert_equal(imp_data.conditional_formula['x4'], fml)

        assert_equal(imp_data._cycle_order, ['x5', 'x3', 'x4', 'y', 'x2', 'x1'])


    @dec.skipif(not have_matplotlib)
    def test_plot_missing_pattern(self):

        df = gendat()
        imp_data = mice.MICEData(df)

        for row_order in "pattern", "raw":
            for hide_complete_rows in False, True:
                for color_row_patterns in False, True:
                    plt.clf()
                    fig = imp_data.plot_missing_pattern(row_order=row_order,
                                      hide_complete_rows=hide_complete_rows,
                                      color_row_patterns=color_row_patterns)
                    close_or_save(pdf, fig)


    @dec.skipif(not have_matplotlib)
    def test_plot_bivariate(self):

        df = gendat()
        imp_data = mice.MICEData(df)
        imp_data.update_all()

        plt.clf()
        for plot_points in False, True:
            fig = imp_data.plot_bivariate('x2', 'x4', plot_points=plot_points)
            fig.get_axes()[0].set_title('plot_bivariate')
            close_or_save(pdf, fig)


    @dec.skipif(not have_matplotlib)
    def test_fit_obs(self):

        df = gendat()
        imp_data = mice.MICEData(df)
        imp_data.update_all()

        plt.clf()
        for plot_points in False, True:
            fig = imp_data.plot_fit_obs('x4', plot_points=plot_points)
            fig.get_axes()[0].set_title('plot_fit_scatterplot')
            close_or_save(pdf, fig)


    @dec.skipif(not have_matplotlib)
    def test_plot_imputed_hist(self):

        df = gendat()
        imp_data = mice.MICEData(df)
        imp_data.update_all()

        plt.clf()
        for plot_points in False, True:
            fig = imp_data.plot_imputed_hist('x4')
            fig.get_axes()[0].set_title('plot_imputed_hist')
            close_or_save(pdf, fig)



class TestMICE(object):

    def test_MICE(self):

        df = gendat()
        imp_data = mice.MICEData(df)
        mi = mice.MICE("y ~ x1 + x2 + x1:x2", sm.OLS, imp_data)
        result = mi.fit(1, 3)

        assert(issubclass(result.__class__, mice.MICEResults))

        # Smoke test for results
        smr = result.summary()


    def test_MICE1(self):

        df = gendat()
        imp_data = mice.MICEData(df)
        mi = mice.MICE("y ~ x1 + x2 + x1:x2", sm.OLS, imp_data)

        from statsmodels.regression.linear_model import RegressionResultsWrapper

        for j in range(3):
            x = mi.next_sample()
            assert(issubclass(x.__class__, RegressionResultsWrapper))


    def test_MICE2(self):

        from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper

        df = gendat()
        imp_data = mice.MICEData(df)
        mi = mice.MICE("x3 ~ x1 + x2", sm.GLM, imp_data,
                       init_kwds={"family": sm.families.Binomial()})

        for j in range(3):
            x = mi.next_sample()
            assert(isinstance(x, GLMResultsWrapper))
            assert(isinstance(x.family, sm.families.Binomial))


    def test_combine(self):

        np.random.seed(3897)
        x1 = np.random.normal(size=300)
        x2 = np.random.normal(size=300)
        y = x1 + x2 + np.random.normal(size=300)
        x1[0:100] = np.nan
        x2[250:] = np.nan
        df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
        idata = mice.MICEData(df)
        mi = mice.MICE("y ~ x1 + x2", sm.OLS, idata, n_skip=20)
        result = mi.fit(10, 20)

        fmi = np.asarray([ 0.1920533 ,  0.1587287 ,  0.33174032])
        assert_allclose(result.frac_miss_info, fmi, atol=1e-5)

        params = np.asarray([-0.05397474,  0.97273307,  1.01652293])
        assert_allclose(result.params, params, atol=1e-5)

        tvalues = np.asarray([ -0.84781698,  15.10491582,  13.59998039])
        assert_allclose(result.tvalues, tvalues, atol=1e-5)


if  __name__=="__main__":

    import nose

    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb', '--pdb-failure'],
                   exit=False)
