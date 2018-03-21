# -*- coding: utf-8 -*-
"""Tests that use cross-checks for generic methods

Should be easy to check consistency across models
Does not cover tsa

Initial cases copied from test_shrink_pickle

Created on Wed Oct 30 14:01:27 2013

Author: Josef Perktold
"""
from statsmodels.compat.python import range
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.compat.scipy import NumpyVersion
from statsmodels.compat.testing import SkipTest

from numpy.testing import (assert_, assert_allclose, assert_equal,
                           assert_array_equal)

import platform


class CheckGenericMixin(object):

    @classmethod
    def setup_class(cls):
        nobs = 500
        np.random.seed(987689)
        x = np.random.randn(nobs, 3)
        x = sm.add_constant(x)
        cls.exog = x
        cls.xf = 0.25 * np.ones((2, 4))
        cls.predict_kwds = {}

    def test_ttest_tvalues(self):
        # test that t_test has same results a params, bse, tvalues, ...
        res = self.results
        mat = np.eye(len(res.params))
        tt = res.t_test(mat)

        assert_allclose(tt.effect, res.params, rtol=1e-12)
        # TODO: tt.sd and tt.tvalue are 2d also for single regressor, squeeze
        assert_allclose(np.squeeze(tt.sd), res.bse, rtol=1e-10)
        assert_allclose(np.squeeze(tt.tvalue), res.tvalues, rtol=1e-12)
        assert_allclose(tt.pvalue, res.pvalues, rtol=5e-10)
        assert_allclose(tt.conf_int(), res.conf_int(), rtol=1e-10)

        # test params table frame returned by t_test
        table_res = np.column_stack((res.params, res.bse, res.tvalues,
                                    res.pvalues, res.conf_int()))
        table1 = np.column_stack((tt.effect, tt.sd, tt.tvalue, tt.pvalue,
                                 tt.conf_int()))
        table2 = tt.summary_frame().values
        assert_allclose(table2, table_res, rtol=1e-12)

        # move this to test_attributes ?
        assert_(hasattr(res, 'use_t'))

        tt = res.t_test(mat[0])
        string_confint = lambda alpha: "[%4.3F      %4.3F]" % (
                                       alpha / 2, 1- alpha / 2)
        summ = tt.summary()   # smoke test for #1323
        assert_allclose(tt.pvalue, res.pvalues[0], rtol=5e-10)
        assert_(string_confint(0.05) in str(summ))
        # issue #3116 alpha not used in column headers
        summ = tt.summary(alpha=0.1)
        ss = "[0.05       0.95]"   # different formatting
        assert_(ss in str(summ))
        summf = tt.summary_frame(alpha=0.1)
        pvstring_use_t = 'P>|z|' if res.use_t is False else 'P>|t|'
        tstring_use_t = 'z' if res.use_t is False else 't'
        cols = ['coef', 'std err', tstring_use_t, pvstring_use_t,
                'Conf. Int. Low', 'Conf. Int. Upp.']
        assert_array_equal(summf.columns.values, cols)


    def test_ftest_pvalues(self):
        res = self.results
        use_t = res.use_t
        k_vars = len(res.params)
        # check default use_t
        pvals = [res.wald_test(np.eye(k_vars)[k], use_f=use_t).pvalue
                                                   for k in range(k_vars)]
        assert_allclose(pvals, res.pvalues, rtol=5e-10, atol=1e-25)

        # sutomatic use_f based on results class use_t
        pvals = [res.wald_test(np.eye(k_vars)[k]).pvalue
                                                   for k in range(k_vars)]
        assert_allclose(pvals, res.pvalues, rtol=5e-10, atol=1e-25)

        # label for pvalues in summary
        string_use_t = 'P>|z|' if use_t is False else 'P>|t|'
        summ = str(res.summary())
        assert_(string_use_t in summ)

        # try except for models that don't have summary2
        try:
            summ2 = str(res.summary2())
        except AttributeError:
            summ2 = None
        if summ2 is not None:
            assert_(string_use_t in summ2)


    # TODO The following is not (yet) guaranteed across models
    #@knownfailureif(True)
    def test_fitted(self):
        # ignore wrapper for isinstance check
        from statsmodels.genmod.generalized_linear_model import GLMResults
        from statsmodels.discrete.discrete_model import DiscreteResults
        # FIXME: work around GEE has no wrapper
        if hasattr(self.results, '_results'):
            results = self.results._results
        else:
            results = self.results
        if (isinstance(results, GLMResults) or
            isinstance(results, DiscreteResults)):
            raise SkipTest('Infeasible for {0}'.format(type(results)))

        res = self.results
        fitted = res.fittedvalues
        assert_allclose(res.model.endog - fitted, res.resid, rtol=1e-12)
        assert_allclose(fitted, res.predict(), rtol=1e-12)

    def test_predict_types(self):

        res = self.results
        # squeeze to make 1d for single regressor test case
        p_exog = np.squeeze(np.asarray(res.model.exog[:2]))

        # ignore wrapper for isinstance check
        from statsmodels.genmod.generalized_linear_model import GLMResults
        from statsmodels.discrete.discrete_model import DiscreteResults

        # FIXME: work around GEE has no wrapper
        if hasattr(self.results, '_results'):
            results = self.results._results
        else:
            results = self.results

        if (isinstance(results, GLMResults) or
            isinstance(results, DiscreteResults)):
            # SMOKE test only  TODO
            res.predict(p_exog)
            res.predict(p_exog.tolist())
            res.predict(p_exog[0].tolist())
        else:

            import pandas as pd
            from pandas.util.testing import assert_series_equal

            fitted = res.fittedvalues[:2]
            assert_allclose(fitted, res.predict(p_exog), rtol=1e-12)
            # this needs reshape to column-vector:
            assert_allclose(fitted, res.predict(np.squeeze(p_exog).tolist()),
                            rtol=1e-12)
            # only one prediction:
            assert_allclose(fitted[:1], res.predict(p_exog[0].tolist()),
                            rtol=1e-12)
            assert_allclose(fitted[:1], res.predict(p_exog[0]),
                            rtol=1e-12)

            exog_index = range(len(p_exog))
            predicted = res.predict(p_exog)

            if p_exog.ndim == 1:
                predicted_pandas = res.predict(pd.Series(p_exog, index=exog_index))

            else:
                predicted_pandas = res.predict(pd.DataFrame(p_exog, index=exog_index))

            if predicted.ndim == 1:

                assert_(isinstance(predicted_pandas, pd.Series))

                predicted_expected = pd.Series(predicted, index=exog_index)
                assert_series_equal(predicted_expected, predicted_pandas)

            else:
                assert_(isinstance(predicted_pandas, pd.DataFrame))

                predicted_expected = pd.DataFrame(predicted, index=exog_index)
                assert_(predicted_expected.equals(predicted_pandas))


#########  subclasses for individual models, unchanged from test_shrink_pickle
# TODO: check if setup_class is faster than setup

class TestGenericOLS(CheckGenericMixin):

    def setup(self):
        #fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        y = x.sum(1) + np.random.randn(x.shape[0])
        self.results = sm.OLS(y, self.exog).fit()


class TestGenericOLSOneExog(CheckGenericMixin):
    # check with single regressor (no constant)

    def setup(self):
        #fit for each test, because results will be changed by test
        x = self.exog[:, 1]
        np.random.seed(987689)
        y = x + np.random.randn(x.shape[0])
        self.results = sm.OLS(y, x).fit()


class TestGenericWLS(CheckGenericMixin):

    def setup(self):
        #fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        y = x.sum(1) + np.random.randn(x.shape[0])
        self.results = sm.WLS(y, self.exog, weights=np.ones(len(y))).fit()


class TestGenericPoisson(CheckGenericMixin):

    def setup(self):
        #fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        y_count = np.random.poisson(np.exp(x.sum(1) - x.mean()))
        model = sm.Poisson(y_count, x)  #, exposure=np.ones(nobs), offset=np.zeros(nobs)) #bug with default
        # use start_params to converge faster
        start_params = np.array([0.75334818, 0.99425553, 1.00494724, 1.00247112])
        self.results = model.fit(start_params=start_params, method='bfgs',
                                 disp=0)

        #TODO: temporary, fixed in master
        self.predict_kwds = dict(exposure=1, offset=0)

class TestGenericNegativeBinomial(CheckGenericMixin):

    def setup(self):
        #fit for each test, because results will be changed by test
        np.random.seed(987689)
        data = sm.datasets.randhie.load()
        exog = sm.add_constant(data.exog, prepend=False)
        mod = sm.NegativeBinomial(data.endog, data.exog)
        start_params = np.array([-0.0565406 , -0.21213599,  0.08783076,
                                 -0.02991835,  0.22901974,  0.0621026,
                                  0.06799283,  0.08406688,  0.18530969,
                                  1.36645452])
        self.results = mod.fit(start_params=start_params, disp=0)


class TestGenericLogit(CheckGenericMixin):

    def setup(self):
        #fit for each test, because results will be changed by test
        x = self.exog
        nobs = x.shape[0]
        np.random.seed(987689)
        y_bin = (np.random.rand(nobs) < 1.0 / (1 + np.exp(x.sum(1) - x.mean()))).astype(int)
        model = sm.Logit(y_bin, x)  #, exposure=np.ones(nobs), offset=np.zeros(nobs)) #bug with default
        # use start_params to converge faster
        start_params = np.array([-0.73403806, -1.00901514, -0.97754543, -0.95648212])
        self.results = model.fit(start_params=start_params, method='bfgs', disp=0)


class TestGenericRLM(CheckGenericMixin):

    def setup(self):
        #fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        y = x.sum(1) + np.random.randn(x.shape[0])
        self.results = sm.RLM(y, self.exog).fit()


class TestGenericGLM(CheckGenericMixin):

    def setup(self):
        #fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        y = x.sum(1) + np.random.randn(x.shape[0])
        self.results = sm.GLM(y, self.exog).fit()


class TestGenericGEEPoisson(CheckGenericMixin):

    def setup(self):
        #fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        y_count = np.random.poisson(np.exp(x.sum(1) - x.mean()))
        groups = np.random.randint(0, 4, size=x.shape[0])
        # use start_params to speed up test, difficult convergence not tested
        start_params = np.array([0., 1., 1., 1.])

        vi = sm.cov_struct.Independence()
        family = sm.families.Poisson()
        self.results = sm.GEE(y_count, self.exog, groups, family=family,
                                cov_struct=vi).fit(start_params=start_params)


class TestGenericGEEPoissonNaive(CheckGenericMixin):

    def setup(self):
        #fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        #y_count = np.random.poisson(np.exp(x.sum(1) - x.mean()))
        y_count = np.random.poisson(np.exp(x.sum(1) - x.sum(1).mean(0)))
        groups = np.random.randint(0, 4, size=x.shape[0])
        # use start_params to speed up test, difficult convergence not tested
        start_params = np.array([0., 1., 1., 1.])

        vi = sm.cov_struct.Independence()
        family = sm.families.Poisson()
        self.results = sm.GEE(y_count, self.exog, groups, family=family,
                                cov_struct=vi).fit(start_params=start_params,
                                                   cov_type='naive')


class TestGenericGEEPoissonBC(CheckGenericMixin):

    def setup(self):
        #fit for each test, because results will be changed by test
        x = self.exog
        np.random.seed(987689)
        #y_count = np.random.poisson(np.exp(x.sum(1) - x.mean()))
        y_count = np.random.poisson(np.exp(x.sum(1) - x.sum(1).mean(0)))
        groups = np.random.randint(0, 4, size=x.shape[0])
        # use start_params to speed up test, difficult convergence not tested
        start_params = np.array([0., 1., 1., 1.])
        # params_est = np.array([-0.0063238 ,  0.99463752,  1.02790201,  0.98080081])

        vi = sm.cov_struct.Independence()
        family = sm.families.Poisson()
        mod = sm.GEE(y_count, self.exog, groups, family=family, cov_struct=vi)
        self.results = mod.fit(start_params=start_params,
                               cov_type='bias_reduced')


# Other test classes

class CheckAnovaMixin(object):

    @classmethod
    def setup_class(cls):
        import statsmodels.stats.tests.test_anova as ttmod

        test = ttmod.TestAnova3()
        test.setup_class()

        cls.data = test.data.drop([0,1,2])
        cls.initialize()


    def test_combined(self):
        res = self.res
        wa = res.wald_test_terms(skip_single=False, combine_terms=['Duration', 'Weight'])
        eye = np.eye(len(res.params))
        c_const = eye[0]
        c_w = eye[[2,3]]
        c_d = eye[1]
        c_dw = eye[[4,5]]
        c_weight = eye[2:6]
        c_duration = eye[[1, 4, 5]]

        compare_waldres(res, wa, [c_const, c_d, c_w, c_dw, c_duration, c_weight])


    def test_categories(self):
        # test only multicolumn terms
        res = self.res
        wa = res.wald_test_terms(skip_single=True)
        eye = np.eye(len(res.params))
        c_w = eye[[2,3]]
        c_dw = eye[[4,5]]

        compare_waldres(res, wa, [c_w, c_dw])


def compare_waldres(res, wa, constrasts):
    for i, c in enumerate(constrasts):
        wt = res.wald_test(c)
        assert_allclose(wa.table.values[i, 0], wt.statistic)
        assert_allclose(wa.table.values[i, 1], wt.pvalue)
        df = c.shape[0] if c.ndim == 2 else 1
        assert_equal(wa.table.values[i, 2], df)
        # attributes
        assert_allclose(wa.statistic[i], wt.statistic)
        assert_allclose(wa.pvalues[i], wt.pvalue)
        assert_equal(wa.df_constraints[i], df)
        if res.use_t:
            assert_equal(wa.df_denom[i], res.df_resid)

    col_names = wa.col_names
    if res.use_t:
        assert_equal(wa.distribution, 'F')
        assert_equal(col_names[0], 'F')
        assert_equal(col_names[1], 'P>F')
    else:
        assert_equal(wa.distribution, 'chi2')
        assert_equal(col_names[0], 'chi2')
        assert_equal(col_names[1], 'P>chi2')

    # SMOKETEST
    wa.summary_frame()


class TestWaldAnovaOLS(CheckAnovaMixin):

    @classmethod
    def initialize(cls):
        from statsmodels.formula.api import ols, glm, poisson
        from statsmodels.discrete.discrete_model import Poisson

        mod = ols("np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)", cls.data)
        cls.res = mod.fit(use_t=False)


    def test_noformula(self):
        endog = self.res.model.endog
        exog = self.res.model.data.orig_exog
        exog = pd.DataFrame(exog)

        res = sm.OLS(endog, exog).fit()
        wa = res.wald_test_terms(skip_single=True,
                                 combine_terms=['Duration', 'Weight'])
        eye = np.eye(len(res.params))
        c_weight = eye[2:6]
        c_duration = eye[[1, 4, 5]]

        compare_waldres(res, wa, [c_duration, c_weight])


class TestWaldAnovaOLSF(CheckAnovaMixin):

    @classmethod
    def initialize(cls):
        from statsmodels.formula.api import ols, glm, poisson
        from statsmodels.discrete.discrete_model import Poisson

        mod = ols("np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)", cls.data)
        cls.res = mod.fit()  # default use_t=True

    def test_predict_missing(self):
        ex = self.data[:5].copy()
        ex.iloc[0, 1] = np.nan
        predicted1 = self.res.predict(ex)
        predicted2 = self.res.predict(ex[1:])
        from pandas.util.testing import assert_series_equal
        try:
            from pandas.util.testing import assert_index_equal
        except ImportError:
            # for old pandas
            from numpy.testing import assert_array_equal as assert_index_equal

        assert_index_equal(predicted1.index, ex.index)
        assert_series_equal(predicted1[1:], predicted2)
        assert_equal(predicted1.values[0], np.nan)


class TestWaldAnovaGLM(CheckAnovaMixin):

    @classmethod
    def initialize(cls):
        from statsmodels.formula.api import ols, glm, poisson
        from statsmodels.discrete.discrete_model import Poisson

        mod = glm("np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)", cls.data)
        cls.res = mod.fit(use_t=False)


class TestWaldAnovaPoisson(CheckAnovaMixin):

    @classmethod
    def initialize(cls):
        from statsmodels.discrete.discrete_model import Poisson

        mod = Poisson.from_formula("Days ~ C(Duration, Sum)*C(Weight, Sum)", cls.data)
        cls.res = mod.fit(cov_type='HC0')


class TestWaldAnovaNegBin(CheckAnovaMixin):

    @classmethod
    def initialize(cls):
        from statsmodels.discrete.discrete_model import NegativeBinomial

        formula = "Days ~ C(Duration, Sum)*C(Weight, Sum)"
        mod = NegativeBinomial.from_formula(formula, cls.data,
                                            loglike_method='nb2')
        cls.res = mod.fit()


class TestWaldAnovaNegBin1(CheckAnovaMixin):

    @classmethod
    def initialize(cls):
        from statsmodels.discrete.discrete_model import NegativeBinomial

        formula = "Days ~ C(Duration, Sum)*C(Weight, Sum)"
        mod = NegativeBinomial.from_formula(formula, cls.data,
                                            loglike_method='nb1')
        cls.res = mod.fit(cov_type='HC0')


class T_estWaldAnovaOLSNoFormula(object):

    @classmethod
    def initialize(cls):
        from statsmodels.formula.api import ols, glm, poisson
        from statsmodels.discrete.discrete_model import Poisson

        mod = ols("np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)", cls.data)
        cls.res = mod.fit()  # default use_t=True


class CheckPairwise(object):

    def test_default(self):
        res = self.res

        tt = res.t_test(self.constraints)

        pw = res.t_test_pairwise(self.term_name)
        pw_frame = pw.result_frame
        assert_allclose(pw_frame.iloc[:, :6].values,
                        tt.summary_frame().values)


class TestTTestPairwiseOLS(CheckPairwise):

    @classmethod
    def setup_class(cls):
        from statsmodels.formula.api import ols
        import statsmodels.stats.tests.test_anova as ttmod

        test = ttmod.TestAnova3()
        test.setup_class()
        cls.data = test.data.drop([0,1,2])

        mod = ols("np.log(Days+1) ~ C(Duration) + C(Weight)", cls.data)
        cls.res = mod.fit()
        cls.term_name = "C(Weight)"
        cls.constraints = ['C(Weight)[T.2]',
                           'C(Weight)[T.3]',
                           'C(Weight)[T.3] - C(Weight)[T.2]']


    def test_alpha(self):
        pw1 = self.res.t_test_pairwise(self.term_name, method='hommel',
                                       factor_labels='A B C'.split())
        pw2 = self.res.t_test_pairwise(self.term_name, method='hommel',
                                       alpha=0.01)
        assert_allclose(pw1.result_frame.iloc[:, :7].values,
                        pw2.result_frame.iloc[:, :7].values, rtol=1e-10)
        assert_equal(pw1.result_frame.iloc[:, -1].values,
                     [True]*3)
        assert_equal(pw2.result_frame.iloc[:, -1].values,
                     [False, True, False])

        assert_equal(pw1.result_frame.index.values,
                     np.array(['B-A', 'C-A', 'C-B'], dtype=object))


class TestTTestPairwiseOLS2(CheckPairwise):

    @classmethod
    def setup_class(cls):
        from statsmodels.formula.api import ols
        import statsmodels.stats.tests.test_anova as ttmod

        test = ttmod.TestAnova3()
        test.setup_class()
        cls.data = test.data.drop([0,1,2])

        mod = ols("np.log(Days+1) ~ C(Weight) + C(Duration)", cls.data)
        cls.res = mod.fit()
        cls.term_name = "C(Weight)"
        cls.constraints = ['C(Weight)[T.2]',
                           'C(Weight)[T.3]',
                           'C(Weight)[T.3] - C(Weight)[T.2]']


class TestTTestPairwiseOLS3(CheckPairwise):

    @classmethod
    def setup_class(cls):
        from statsmodels.formula.api import ols
        import statsmodels.stats.tests.test_anova as ttmod

        test = ttmod.TestAnova3()
        test.setup_class()
        cls.data = test.data.drop([0,1,2])

        mod = ols("np.log(Days+1) ~ C(Weight) + C(Duration) - 1", cls.data)
        cls.res = mod.fit()
        cls.term_name = "C(Weight)"
        cls.constraints = ['C(Weight)[2] - C(Weight)[1]',
                           'C(Weight)[3] - C(Weight)[1]',
                           'C(Weight)[3] - C(Weight)[2]']

class TestTTestPairwiseOLS4(CheckPairwise):

    @classmethod
    def setup_class(cls):
        from statsmodels.formula.api import ols
        import statsmodels.stats.tests.test_anova as ttmod

        test = ttmod.TestAnova3()
        test.setup_class()
        cls.data = test.data.drop([0,1,2])

        mod = ols("np.log(Days+1) ~ C(Weight, Treatment(2)) + C(Duration)", cls.data)
        cls.res = mod.fit()
        cls.term_name = "C(Weight, Treatment(2))"
        cls.constraints = ['-C(Weight, Treatment(2))[T.1]',
                           'C(Weight, Treatment(2))[T.3] - C(Weight, Treatment(2))[T.1]',
                           'C(Weight, Treatment(2))[T.3]',]


class TestTTestPairwisePoisson(CheckPairwise):

    @classmethod
    def setup_class(cls):
        from statsmodels.discrete.discrete_model import Poisson
        import statsmodels.stats.tests.test_anova as ttmod

        test = ttmod.TestAnova3()
        test.setup_class()
        cls.data = test.data.drop([0,1,2])

        mod = Poisson.from_formula("Days ~ C(Duration) + C(Weight)", cls.data)
        cls.res = mod.fit(cov_type='HC0')
        cls.term_name = "C(Weight)"
        cls.constraints = ['C(Weight)[T.2]',
                           'C(Weight)[T.3]',
                           'C(Weight)[T.3] - C(Weight)[T.2]']



if __name__ == '__main__':
    pass
