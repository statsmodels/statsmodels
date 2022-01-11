
import numpy as np
from numpy.testing import assert_allclose, assert_equal

from statsmodels import datasets
from statsmodels.tools.tools import add_constant

from statsmodels.distributions.discrete import (
    truncatedpoisson,
    truncatednegbin,
    )
from statsmodels.discrete.truncated_model import (
    TruncatedPoisson,
    TruncatedNegativeBinomialP,
    Hurdle
    )

from statsmodels.sandbox.regression.tests.test_gmm_poisson import DATA
from .results.results_discrete import RandHIE
from .results import results_truncated as results_t
from .results import results_truncated_st as results_ts


class CheckResults(object):
    def test_params(self):
        assert_allclose(self.res1.params, self.res2.params,
                        atol=1e-5, rtol=1e-5)

    def test_llf(self):
        assert_allclose(self.res1.llf, self.res2.llf, atol=1e-5, rtol=1e-7)

    def test_conf_int(self):
        assert_allclose(self.res1.conf_int(), self.res2.conf_int,
                        atol=1e-3, rtol=1e-5)

    def test_bse(self):
        assert_allclose(self.res1.bse, self.res2.bse, atol=1e-3)

    def test_aic(self):
        assert_allclose(self.res1.aic, self.res2.aic, atol=1e-2, rtol=1e-12)

    def test_bic(self):
        assert_allclose(self.res1.bic, self.res2.bic, atol=1e-2, rtol=1e-12)

    def test_fit_regularized(self):
        model = self.res1.model

        alpha = np.ones(len(self.res1.params))
        res_reg = model.fit_regularized(alpha=alpha*0.01, disp=0)

        assert_allclose(res_reg.params, self.res1.params,
                        rtol=1e-3, atol=5e-3)
        assert_allclose(res_reg.bse, self.res1.bse,
                        rtol=1e-3, atol=5e-3)


class TestTruncatedPoissonModel(CheckResults):
    @classmethod
    def setup_class(cls):
        data = datasets.randhie.load()
        exog = add_constant(np.asarray(data.exog)[:, :4], prepend=False)
        mod = TruncatedPoisson(data.endog, exog, truncation=5)
        cls.res1 = mod.fit(method="newton", maxiter=500)
        res2 = RandHIE()
        res2.truncated_poisson()
        cls.res2 = res2


class TestZeroTruncatedPoissonModel(CheckResults):
    @classmethod
    def setup_class(cls):
        data = datasets.randhie.load()
        exog = add_constant(np.asarray(data.exog)[:, :4], prepend=False)
        mod = TruncatedPoisson(data.endog, exog, truncation=0)
        cls.res1 = mod.fit(maxiter=500)
        res2 = RandHIE()
        res2.zero_truncated_poisson()
        cls.res2 = res2


class TestZeroTruncatedNBPModel(CheckResults):
    @classmethod
    def setup_class(cls):
        data = datasets.randhie.load()
        exog = add_constant(np.asarray(data.exog)[:, :3], prepend=False)
        mod = TruncatedNegativeBinomialP(data.endog, exog, truncation=0)
        cls.res1 = mod.fit(maxiter=500)
        res2 = RandHIE()
        res2.zero_truncted_nbp()
        cls.res2 = res2

    def test_conf_int(self):
        pass


class TestTruncatedPoisson_predict(object):
    @classmethod
    def setup_class(cls):
        cls.expected_params = [1, 0.5]
        np.random.seed(123)
        nobs = 200
        exog = np.ones((nobs, 2))
        exog[:nobs//2, 1] = 2
        mu_true = exog.dot(cls.expected_params)
        cls.endog = truncatedpoisson.rvs(mu_true, 0, size=mu_true.shape)
        model = TruncatedPoisson(cls.endog, exog, truncation=0)
        cls.res = model.fit(method='bfgs', maxiter=5000)

    def test_mean(self):
        assert_allclose(self.res.predict().mean(), self.endog.mean(),
                        atol=2e-1, rtol=2e-1)

    def test_var(self):
        v = self.res.predict(which="var").mean()
        assert_allclose(v, self.endog.var(), atol=2e-1, rtol=2e-1)
        return
        assert_allclose((self.res.predict().mean() *
                        self.res._dispersion_factor.mean()),
                        self.endog.var(), atol=5e-2, rtol=5e-2)

    def test_predict_prob(self):
        res = self.res

        pr = res.predict(which='prob')
        pr2 = truncatedpoisson.pmf(
            np.arange(8), res.predict(which="mean-main")[:, None], 0)
        assert_allclose(pr, pr2, rtol=1e-10, atol=1e-10)


class TestTruncatedNBP_predict(object):
    @classmethod
    def setup_class(cls):
        cls.expected_params = [1, 0.5, 0.5]
        np.random.seed(1234)
        nobs = 200
        exog = np.ones((nobs, 2))
        exog[:nobs//2, 1] = 2
        mu_true = np.exp(exog.dot(cls.expected_params[:-1]))
        cls.endog = truncatednegbin.rvs(
            mu_true, cls.expected_params[-1], 2, 0, size=mu_true.shape)
        model = TruncatedNegativeBinomialP(cls.endog, exog, truncation=0, p=2)
        cls.res = model.fit(method='nm', maxiter=5000, maxfun=5000)

    def test_mean(self):
        assert_allclose(self.res.predict().mean(), self.endog.mean(),
                        atol=2e-1, rtol=2e-1)

    def test_var(self):
        v = self.res.predict(which="var").mean()
        assert_allclose(v, self.endog.var(), atol=1e-1, rtol=1e-2)
        return
        assert_allclose((self.res.predict().mean() *
                        self.res._dispersion_factor.mean()),
                        self.endog.var(), atol=5e-2, rtol=5e-2)

    def test_predict_prob(self):
        res = self.res

        pr = res.predict(which='prob')
        pr2 = truncatednegbin.pmf(
            np.arange(29),
            res.predict(which="mean-main")[:, None], res.params[-1], 2, 0)
        assert_allclose(pr, pr2, rtol=1e-10, atol=1e-10)


class CheckTruncatedST():

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2

        assert_allclose(res1.llf, res2.ll, rtol=1e-8)
        assert_allclose(res1.llnull, res2.ll_0, rtol=5e-6)
        pt2 = res2.params_table
        # Stata has different parameterization of alpha for negbin
        k = res1.model.exog.shape[1]
        assert_allclose(res1.params[:k], res2.params[:k], atol=1e-5)
        assert_allclose(res1.bse[:k], pt2[:k, 1], atol=1e-5)
        assert_allclose(res1.tvalues[:k], pt2[:k, 2], rtol=5e-4, atol=5e-4)
        assert_allclose(res1.pvalues[:k], pt2[:k, 3], rtol=5e-4, atol=1e-7)

        # df_resid not available in Stata
        # assert_equal(res1.df_resid, res2.df_residual)
        assert_equal(res1.df_model, res2.df_m)
        assert_allclose(res1.aic, res2.icr[-2], rtol=1e-8)
        assert_allclose(res1.bic, res2.icr[-1], rtol=1e-8)

    def test_predict(self):
        res1 = self.res1
        res2 = self.res2

        # mean of untruncated distribution
        rdf = res2.margins_means.table
        pred = res1.get_prediction(which="mean-main", average=True)
        assert_allclose(pred.predicted, rdf[0], rtol=5e-5)
        assert_allclose(pred.se, rdf[1], rtol=5e-4, atol=1e-10)
        ci = pred.conf_int()[0]
        assert_allclose(ci[0], rdf[4], rtol=1e-5, atol=1e-10)
        assert_allclose(ci[1], rdf[5], rtol=1e-5, atol=1e-10)

        # mean of untruncated distribution, evaluated and exog.mean()
        ex = res1.model.exog.mean(0)
        rdf = res2.margins_atmeans.table
        pred = res1.get_prediction(ex, which="mean-main")
        assert_allclose(pred.predicted, rdf[0], rtol=5e-5)
        assert_allclose(pred.se, rdf[1], rtol=5e-4, atol=1e-10)
        ci = pred.conf_int()[0]
        assert_allclose(ci[0], rdf[4], rtol=5e-5, atol=1e-10)
        assert_allclose(ci[1], rdf[5], rtol=5e-5, atol=1e-10)

        # mean of truncated distribution, E(y | y > trunc)
        rdf = res2.margins_cm.table
        try:
            pred = res1.get_prediction(average=True)
        except NotImplementedError:
            # not yet implemented for truncation > 0
            pred = None
        if pred is not None:
            assert_allclose(pred.predicted, rdf[0], rtol=5e-5)
            assert_allclose(pred.se, rdf[1], rtol=1e-5, atol=1e-10)
            ci = pred.conf_int()[0]
            assert_allclose(ci[0], rdf[4], rtol=1e-5, atol=1e-10)
            assert_allclose(ci[1], rdf[5], rtol=1e-5, atol=1e-10)

        # predicted probabilites, only subset is common to reference
        ex = res1.model.exog.mean(0)
        rdf = res2.margins_cpr.table
        start_idx = res1.model.truncation + 1
        k = rdf.shape[0] + res1.model.truncation
        pred = res1.get_prediction(which="prob", average=True)
        assert_allclose(pred.predicted[start_idx:k], rdf[:-1, 0], rtol=5e-5)
        assert_allclose(pred.se[start_idx:k], rdf[:-1, 1],
                        rtol=5e-4, atol=1e-10)
        ci = pred.conf_int()[start_idx:k]
        assert_allclose(ci[:, 0], rdf[:-1, 4], rtol=5e-5, atol=1e-10)
        assert_allclose(ci[:, 1], rdf[:-1, 5], rtol=5e-5, atol=1e-10)

        # untruncated predicted probabilites, subset is common to reference
        ex = res1.model.exog.mean(0)
        rdf = res2.margins_pr.table
        k = rdf.shape[0] - 1
        pred = res1.get_prediction(which="prob-main", average=True)
        assert_allclose(pred.predicted[:k], rdf[:-1, 0], rtol=5e-5)
        assert_allclose(pred.se[:k], rdf[:-1, 1],
                        rtol=8e-4, atol=1e-10)
        ci = pred.conf_int()[:k]
        assert_allclose(ci[:, 0], rdf[:-1, 4], rtol=5e-4, atol=1e-10)
        assert_allclose(ci[:, 1], rdf[:-1, 5], rtol=5e-4, atol=1e-10)


class TestTruncatedPoissonSt(CheckTruncatedST):
    # test against R pscl
    @classmethod
    def setup_class(cls):
        endog = DATA["docvis"]
        exog_names = ['aget', 'totchr', 'const']
        exog = DATA[exog_names]
        cls.res1 = TruncatedPoisson(endog, exog).fit(method="bfgs",
                                                     maxiter=300)
        cls.res2 = results_ts.results_trunc_poisson


class TestTruncatedNegBinSt(CheckTruncatedST):
    # test against R pscl
    @classmethod
    def setup_class(cls):
        endog = DATA["docvis"]
        exog_names = ['aget', 'totchr', 'const']
        exog = DATA[exog_names]
        cls.res1 = TruncatedNegativeBinomialP(endog, exog).fit(method="bfgs",
                                                               maxiter=300)
        cls.res2 = results_ts.results_trunc_negbin


class TestTruncatedPoisson1St(CheckTruncatedST):
    # test against R pscl
    @classmethod
    def setup_class(cls):
        endog = DATA["docvis"]
        exog_names = ['aget', 'totchr', 'const']
        exog = DATA[exog_names]
        cls.res1 = TruncatedPoisson(
            endog, exog, truncation=1
            ).fit(method="bfgs", maxiter=300)
        cls.res2 = results_ts.results_trunc_poisson1


class TestTruncatedNegBin1St(CheckTruncatedST):
    # test against R pscl
    @classmethod
    def setup_class(cls):
        endog = DATA["docvis"]
        exog_names = ['aget', 'totchr', 'const']
        exog = DATA[exog_names]
        cls.res1 = TruncatedNegativeBinomialP(
            endog, exog, truncation=1
            ).fit(method="newton", maxiter=300)  # "bfgs" is not close enough
        cls.res2 = results_ts.results_trunc_negbin1


class TestHurdlePoissonR():
    # test against R pscl
    @classmethod
    def setup_class(cls):
        endog = DATA["docvis"]
        exog_names = ['const', 'aget', 'totchr']
        exog = DATA[exog_names]
        cls.res1 = Hurdle(endog, exog).fit(method="bfgs", maxiter=300)
        cls.res2 = results_t.hurdle_poisson

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2

        assert_allclose(res1.llf, res2.loglik, rtol=1e-8)
        pt2 = res2.params_table
        assert_allclose(res1.params, pt2[:, 0], atol=1e-5)
        assert_allclose(res1.bse, pt2[:, 1], atol=1e-5)
        assert_allclose(res1.tvalues, pt2[:, 2], rtol=5e-4, atol=5e-4)
        assert_allclose(res1.pvalues, pt2[:, 3], rtol=5e-4, atol=1e-7)

        assert_equal(res1.df_resid, res2.df_residual)
        assert_equal(res1.df_model, res2.df_null - res2.df_residual)
        assert_allclose(res1.aic, res2.aic, rtol=1e-8)

        # we have zero model first
        idx = np.concatenate((np.arange(3, 6), np.arange(3)))
        vcov = res2.vcov[idx[:, None], idx]
        assert_allclose(np.asarray(res1.cov_params()), vcov,
                        rtol=1e-4, atol=1e-8)
