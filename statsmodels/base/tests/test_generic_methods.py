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
import statsmodels.api as sm
from statsmodels.compat.scipy import NumpyVersion

from numpy.testing import assert_, assert_allclose

from nose import SkipTest
import platform


iswin = platform.system() == 'Windows'
npversionless15 = NumpyVersion(np.__version__) < '1.5.0'
winoldnp = iswin & npversionless15



class CheckGenericMixin(object):

    def __init__(self):
        self.predict_kwds = {}

    @classmethod
    def setup_class(self):

        nobs = 500
        np.random.seed(987689)
        x = np.random.randn(nobs, 3)
        x = sm.add_constant(x)
        self.exog = x
        self.xf = 0.25 * np.ones((2, 4))

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
        tt.summary()   # smoke test for #1323
        assert_allclose(tt.pvalue, res.pvalues[0], rtol=5e-10)


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
            raise SkipTest

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

            # predict doesn't preserve DataFrame, e.g. dot converts to ndarray
#             import pandas
#             predicted = res.predict(pandas.DataFrame(p_exog))
#             assert_(isinstance(predicted, pandas.DataFrame))
#             assert_allclose(predicted, fitted, rtol=1e-12)


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


if __name__ == '__main__':
    pass
