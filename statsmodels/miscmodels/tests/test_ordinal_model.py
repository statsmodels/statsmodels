"""
Test  for ordinal models
"""

import numpy as np

from numpy.testing import assert_allclose
from .results.results_ordinal_model import data_store as ds
from statsmodels.miscmodels.ordinal_model import OrderedModel


class CheckOrdinalModelMixin(object):

    def test_basic(self):
        n_cat = ds.n_ordinal_cat
        res1 = self.res1
        res2 = self.res2
        # coefficients values, standard errors, t & p values
        assert_allclose(res1.params[:-n_cat + 1], res2.coefficients_val, atol=2e-4)
        assert_allclose(res1.bse[:-n_cat + 1], res2.coefficients_stdE, rtol=0.003, atol=1e-5)
        assert_allclose(res1.tvalues[:-n_cat + 1], res2.coefficients_tval, rtol=0.003, atol=7e-4)
        assert_allclose(res1.pvalues[:-n_cat + 1], res2.coefficients_pval, rtol=0.009, atol=1e-5)
        # thresholds are given with exponentiated increments from the first threshold
        assert_allclose(res1.model.transform_threshold_params(res1.params)[1:-1], res2.thresholds, atol=4e-4)

        # probabilities
        assert_allclose(res1.predict()[:7, :],
                        res2.prob_pred, atol=5e-5)

    def test_Pandas(self):
        # makes sure that the Pandas ecosystem is supported
        res1 = self.res1
        resp = self.resp
        # converges slightly differently why?
        assert_allclose(res1.params, resp.params, atol=1e-10)
        assert_allclose(res1.bse, resp.bse, atol=1e-10)

        assert_allclose(res1.model.endog, resp.model.endog, rtol=1e-10)
        assert_allclose(res1.model.exog, resp.model.exog, rtol=1e-10)

    def test_formula(self):
        res1 = self.res1
        resf = self.resf
        # converges slightly differently why? yet e-5 is ok
        assert_allclose(res1.params, resf.params, atol=5e-5)
        assert_allclose(res1.bse, resf.bse, atol=5e-5)

        assert_allclose(res1.model.endog, resf.model.endog, rtol=1e-10)
        assert_allclose(res1.model.exog, resf.model.exog, rtol=1e-10)

    def test_unordered(self):
        # makes sure that ordered = True is optional for the endog Serie
        # et categories have to be set in the right order
        res1 = self.res1
        resf = self.resu
        # converges slightly differently why?
        assert_allclose(res1.params, resf.params, atol=1e-10)
        assert_allclose(res1.bse, resf.bse, atol=1e-10)

        assert_allclose(res1.model.endog, resf.model.endog, rtol=1e-10)
        assert_allclose(res1.model.exog, resf.model.exog, rtol=1e-10)


class TestLogitModel(CheckOrdinalModelMixin):

    @classmethod
    def setup_class(cls):
        data = ds.df
        data_unordered = ds.df_unordered

        # standard fit
        mod = OrderedModel(data['apply'].values.codes,
                           np.asarray(data[['pared', 'public', 'gpa']], float),
                           distr='logit')
        res = mod.fit(method='bfgs', disp=False)
        # standard fit with pandas input
        modp = OrderedModel(data['apply'],
                            data[['pared', 'public', 'gpa']],
                            distr='logit')
        resp = modp.fit(method='bfgs', disp=False)
        # fit with formula
        modf = OrderedModel.from_formula("apply ~ pared + public + gpa - 1",
                                         data={"apply": data['apply'].values.codes,
                                               "pared": data['pared'],
                                               "public": data['public'],
                                               "gpa": data['gpa']},
                                         distr='logit')
        resf = modf.fit(method='bfgs', disp=False)
        # fit on data with ordered=False
        modu = OrderedModel(data_unordered['apply'].values.codes,
                            np.asarray(data_unordered[['pared', 'public', 'gpa']], float),
                            distr='logit')
        resu = modu.fit(method='bfgs', disp=False)

        from .results.results_ordinal_model import res_ord_logit as res2
        cls.res2 = res2
        cls.res1 = res
        cls.resp = resp
        cls.resf = resf
        cls.resu = resu


class TestProbitModel(CheckOrdinalModelMixin):

    @classmethod
    def setup_class(cls):
        data = ds.df
        data_unordered = ds.df_unordered

        mod = OrderedModel(data['apply'].values.codes,
                           np.asarray(data[['pared', 'public', 'gpa']], float),
                           distr='probit')
        res = mod.fit(method='bfgs', disp=False)

        modp = OrderedModel(data['apply'],
                            data[['pared', 'public', 'gpa']],
                            distr='probit')
        resp = modp.fit(method='bfgs', disp=False)

        modf = OrderedModel.from_formula("apply ~ pared + public + gpa - 1",
                                         data={"apply": data['apply'].values.codes,
                                               "pared": data['pared'],
                                               "public": data['public'],
                                               "gpa": data['gpa']},
                                         distr='probit')
        resf = modf.fit(method='bfgs', disp=False)

        modu = OrderedModel(data_unordered['apply'].values.codes,
                            np.asarray(data_unordered[['pared', 'public', 'gpa']], float),
                            distr='probit')
        resu = modu.fit(method='bfgs', disp=False)

        from .results.results_ordinal_model import res_ord_probit as res2
        cls.res2 = res2
        cls.res1 = res
        cls.resp = resp
        cls.resf = resf
        cls.resu = resu
