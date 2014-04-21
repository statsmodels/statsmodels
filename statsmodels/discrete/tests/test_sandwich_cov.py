# -*- coding: utf-8 -*-
"""

Created on Mon Dec 09 21:29:20 2013

Author: Josef Perktold
"""

import os
import numpy as np
import pandas as pd
import statsmodels.discrete.discrete_model as smd
import statsmodels.stats.sandwich_covariance as sc
from statsmodels.tools.tools import add_constant

from numpy.testing import assert_allclose


# get data and results as module global for now, TODO: move to class
from .results import results_count_robust_cluster as results_st

cur_dir = os.path.dirname(os.path.abspath(__file__))

filepath = os.path.join(cur_dir, "results", "ships.csv")
data_raw = pd.read_csv(filepath, index_col=False)
data = data_raw.dropna()

#mod = smd.Poisson.from_formula('accident ~ yr_con + op_75_79', data=dat)
# Don't use formula for tests against Stata because intercept needs to be last
endog = data['accident']
exog_data = data['yr_con op_75_79'.split()]
exog = add_constant(exog_data, prepend=False)
group = np.asarray(data['ship'], int)
exposure = np.asarray(data['service'])


# TODO get the test methods from regression/tests
class CheckCountRobustMixin(object):


    def test_basic(self):
        res1 = self.res1
        res2 = self.res2

        if len(res1.params) == (len(res2.params) - 1):
            # Stata includes lnalpha in table for NegativeBinomial
            mask = np.ones(len(res2.params), np.bool_)
            mask[-2] = False
            res2_params = res2.params[mask]
            res2_bse = res2.bse[mask]
        else:
            res2_params = res2.params
            res2_bse = res2.bse

        assert_allclose(res1._results.params, res2_params, 1e-4)

        assert_allclose(self.bse_rob / self.corr_fact, res2_bse, 1e-5)

    @classmethod
    def get_robust_clu(cls):
        res1 = cls.res1
        cov_clu = sc.cov_cluster(res1, group)
        cls.bse_rob = sc.se_cov(cov_clu)

        nobs, k_vars = res1.model.exog.shape
        k_params = len(res1.params)
        #n_groups = len(np.unique(group))
        corr_fact = (nobs-1.) / float(nobs - k_params)
        # for bse we need sqrt of correction factor
        cls.corr_fact = np.sqrt(corr_fact)


    def test_oth(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1._results.llf, res2.ll, 1e-4)
        assert_allclose(res1._results.llnull, res2.ll_0, 1e-4)


class TestPoissonClu(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_poisson_clu
        mod = smd.Poisson(endog, exog)
        cls.res1 = mod.fit(disp=False)
        cls.get_robust_clu()


class TestPoissonCluExposure(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_poisson_exposure_clu #nonrobust
        mod = smd.Poisson(endog, exog, exposure=exposure)
        cls.res1 = mod.fit(disp=False)
        cls.get_robust_clu()


class TestNegbinClu(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_negbin_clu
        mod = smd.NegativeBinomial(endog, exog)
        cls.res1 = mod.fit(disp=False)
        cls.get_robust_clu()


class TestNegbinCluExposure(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_negbin_exposure_clu #nonrobust
        mod = smd.NegativeBinomial(endog, exog, exposure=exposure)
        cls.res1 = mod.fit(disp=False)
        cls.get_robust_clu()


#        mod_nbe = smd.NegativeBinomial(endog, exog, exposure=data['service'])
#        res_nbe = mod_nbe.fit()
#        mod_nb = smd.NegativeBinomial(endog, exog)
#        res_nb = mod_nb.fit()
#
#        cov_clu_nb = sc.cov_cluster(res_nb, group)
#        k_params = k_vars + 1
#        print sc.se_cov(cov_clu_nb / ((nobs-1.) / float(nobs - k_params)))
#
#        wt = res_nb.wald_test(np.eye(len(res_nb.params))[1:3], cov_p=cov_clu_nb/((nobs-1.) / float(nobs - k_params)))
#        print wt
#
#        print dir(results_st)

if __name__ == '__main__':
    tt = TestPoissonClu()
    tt.setup_class()
    tt.test_basic()
    tt = TestNegbinClu()
    tt.setup_class()
    tt.test_basic()
