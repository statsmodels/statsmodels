# -*- coding: utf-8 -*-
"""Testing OLS robust covariance matrices against STATA

Created on Mon Oct 28 15:25:14 2013

Author: Josef Perktold
"""

import numpy as np
from scipy import stats

from numpy.testing import assert_allclose, assert_equal

from statsmodels.regression.linear_model import OLS
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.tools import add_constant
from statsmodels.datasets import macrodata

import results.results_macro_ols_robust as res

#test_hac_simple():

class CheckOLSRobust(object):

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.params, res2.params, rtol=1e-10)
        assert_allclose(self.bse_robust, res2.bse, rtol=1e-10)
        assert_allclose(self.cov_robust, res2.cov, rtol=1e-10)

    def test_tests(self):
        # Note: differences between small (t-distribution, ddof) and large (normal)
        # F statistic has no ddof correction in large, but uses F distribution (?)
        res1 = self.res1
        res2 = self.res2
        mat = np.eye(len(res1.params))
        tt = res1.t_test(mat, cov_p=self.cov_robust)
        # has 'effect', 'pvalue', 'sd', 'tvalue'
        # TODO confint missing
        assert_allclose(tt.effect, res2.params, rtol=1e-12)
        assert_allclose(tt.sd, res2.bse, rtol=1e-10)
        assert_allclose(tt.tvalue, res2.tvalues, rtol=1e-12)
        if self.small:
            assert_allclose(tt.pvalue, res2.pvalues, rtol=5e-10)
        else:
            pval = stats.norm.sf(np.abs(tt.tvalue)) * 2
            assert_allclose(pval, res2.pvalues, rtol=5e-10)

        ft = res1.f_test(mat[:-1], cov_p=self.cov_robust)
        if self.small:
            #'df_denom', 'df_num', 'fvalue', 'pvalue'
            assert_allclose(ft.fvalue, res2.F, rtol=1e-10)
            # f-pvalue is not directly available in Stata results, but is in ivreg2
            if hasattr(res2, 'Fp'):
                assert_allclose(ft.pvalue, res2.Fp, rtol=1e-10)
        else:
            dof_corr = res1.df_resid * 1. / res1.nobs
            assert_allclose(ft.fvalue * dof_corr, res2.F, rtol=1e-10)

        if hasattr(res2, 'df_r'):
            assert_equal(ft.df_num, res2.df_m)
            assert_equal(ft.df_denom, res2.df_r)
        else:
            # ivreg2
            assert_equal(ft.df_num, res2.Fdf1)
            assert_equal(ft.df_denom, res2.Fdf2)





class TestOLSRobust1(CheckOLSRobust):
    # compare with regress robust

    def setup(self):
        res_ols = self.res1
        self.bse_robust = res_ols.HC1_se
        self.cov_robust = res_ols.cov_HC1
        self.small = True
        self.res2 = res.results_hc0

    @classmethod
    def setup_class(cls):
        d2 = macrodata.load().data
        g_gdp = 400*np.diff(np.log(d2['realgdp']))
        g_inv = 400*np.diff(np.log(d2['realinv']))
        exogg = add_constant(np.c_[g_gdp, d2['realint'][:-1]], prepend=False)

        cls.res1 = res_ols = OLS(g_inv, exogg).fit()


class TestOLSRobust2(TestOLSRobust1):
    # compare with ivreg robust small

    def setup(self):
        res_ols = self.res1
        self.bse_robust = res_ols.HC1_se
        self.cov_robust = res_ols.cov_HC1
        self.small = True

        self.res2 = res.results_ivhc0_small



class TestOLSRobust3(TestOLSRobust1):
    # compare with ivreg robust   (not small)

    def setup(self):
        res_ols = self.res1
        self.bse_robust = res_ols.HC0_se
        self.cov_robust = res_ols.cov_HC0
        self.small = False

        self.res2 = res.results_ivhc0_large


class TestOLSRobustHacSmall(TestOLSRobust1):
    # compare with ivreg robust small

    def setup(self):
        res_ols = self.res1
        cov1 = sw.cov_hac_simple(res_ols, nlags=4, use_correction=True)
        se1 =  sw.se_cov(cov1)
        self.bse_robust = se1
        self.cov_robust = cov1
        self.small = True

        self.res2 = res.results_ivhac4_small



class TestOLSRobustHacLarge(TestOLSRobust1):
    # compare with ivreg robust (not small)

    def setup(self):
        res_ols = self.res1
        cov1 = sw.cov_hac_simple(res_ols, nlags=4, use_correction=False)
        se1 =  sw.se_cov(cov1)
        self.bse_robust = se1
        self.cov_robust = cov1
        self.small = False

        self.res2 = res.results_ivhac4_large


class CheckOLSRobustNewMixin(object):


    def test_compare(self):
        assert_allclose(self.cov_robust, self.cov_robust2, rtol=1e-10)
        assert_allclose(self.bse_robust, self.bse_robust2, rtol=1e-10)


    def test_fvalue(self):
        assert_allclose(self.res1.fvalue, self.res2.F, rtol=1e-10)
        assert_allclose(self.res1.f_pvalue, self.res2.Fp, rtol=1e-10)


    def test_confint(self):
        ci1 = self.res1.conf_int()
        ci2 = self.res2.params_table[:,4:6]
        assert_allclose(ci1, ci2, rtol=1e-10)

    def test_smoke(self):
        self.res1.summary()



class TestOLSRobust2New(TestOLSRobust1, CheckOLSRobustNewMixin):
    # compare with ivreg robust small

    def setup(self):
        res_ols = self.res1.get_robustcov_results('HC1')
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        self.bse_robust2 = res_ols.HC1_se
        self.cov_robust2 = res_ols.cov_HC1
        self.small = True
        self.res2 = res.results_ivhc0_small




class TestOLSRobustHACSmallNew(TestOLSRobust1, CheckOLSRobustNewMixin):
    # compare with ivreg robust small

    def setup(self):
        res_ols = self.res1.get_robustcov_results('HAC', maxlags=4,
                                                  use_correction=True)
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        cov1 = sw.cov_hac_simple(res_ols, nlags=4, use_correction=True)
        se1 =  sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1
        self.small = True
        self.res2 = res.results_ivhac4_small


