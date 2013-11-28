# -*- coding: utf-8 -*-
"""Testing OLS robust covariance matrices against STATA

Created on Mon Oct 28 15:25:14 2013

Author: Josef Perktold
"""

import numpy as np
from scipy import stats

from numpy.testing import assert_allclose, assert_equal, assert_warns

from statsmodels.regression.linear_model import OLS, WLS
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.tools import add_constant
from statsmodels.datasets import macrodata

from .results import results_macro_ols_robust as res
from .results import results_grunfeld_ols_robust_cluster as res2

#test_hac_simple():

class CheckOLSRobust(object):

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2
        rtol = getattr(self, 'rtol', 1e-10)
        assert_allclose(res1.params, res2.params, rtol=rtol)
        assert_allclose(self.bse_robust, res2.bse, rtol=rtol)
        assert_allclose(self.cov_robust, res2.cov, rtol=rtol)

    def test_tests(self):
        # Note: differences between small (t-distribution, ddof) and large (normal)
        # F statistic has no ddof correction in large, but uses F distribution (?)
        res1 = self.res1
        res2 = self.res2
        rtol = getattr(self, 'rtol', 1e-10)
        rtolh = getattr(self, 'rtolh', 1e-12)
        mat = np.eye(len(res1.params))
        tt = res1.t_test(mat, cov_p=self.cov_robust)
        # has 'effect', 'pvalue', 'sd', 'tvalue'
        # TODO confint missing
        assert_allclose(tt.effect, res2.params, rtol=rtol)
        assert_allclose(tt.sd, res2.bse, rtol=rtol)
        assert_allclose(tt.tvalue, res2.tvalues, rtol=rtol)
        if self.small:
            assert_allclose(tt.pvalue, res2.pvalues, rtol=5 * rtol)
        else:
            pval = stats.norm.sf(np.abs(tt.tvalue)) * 2
            assert_allclose(pval, res2.pvalues, rtol=5 * rtol)

        ft = res1.f_test(mat[:-1], cov_p=self.cov_robust)
        if self.small:
            #'df_denom', 'df_num', 'fvalue', 'pvalue'
            assert_allclose(ft.fvalue, res2.F, rtol=rtol)
            # f-pvalue is not directly available in Stata results, but is in ivreg2
            if hasattr(res2, 'Fp'):
                assert_allclose(ft.pvalue, res2.Fp, rtol=rtol)
        else:
            if not getattr(self, 'skip_f', False):
                dof_corr = res1.df_resid * 1. / res1.nobs
                assert_allclose(ft.fvalue * dof_corr, res2.F, rtol=rtol)

        if hasattr(res2, 'df_r'):
            assert_equal(ft.df_num, res2.df_m)
            assert_equal(ft.df_denom, res2.df_r)
        else:
            # ivreg2
            assert_equal(ft.df_num, res2.Fdf1)
            assert_equal(ft.df_denom, res2.Fdf2)

        # SMOKE
        tt.summary()
        ft.summary()
        tt.summary_frame()



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
    # This uses the robust covariance as default covariance

    def test_compare(self):
        rtol = getattr(self, 'rtol', 1e-10)
        assert_allclose(self.cov_robust, self.cov_robust2, rtol=rtol)
        assert_allclose(self.bse_robust, self.bse_robust2, rtol=rtol)


    def test_fvalue(self):
        if not getattr(self, 'skip_f', False):
            rtol = getattr(self, 'rtol', 1e-10)
            assert_allclose(self.res1.fvalue, self.res2.F, rtol=rtol)
            if hasattr(self.res2, 'Fp'):
                #only available with ivreg2
                assert_allclose(self.res1.f_pvalue, self.res2.Fp, rtol=rtol)


    def test_confint(self):
        rtol = getattr(self, 'rtol', 1e-10)
        ci1 = self.res1.conf_int()
        ci2 = self.res2.params_table[:,4:6]
        assert_allclose(ci1, ci2, rtol=rtol)

        # check critical value
        crit1 = np.diff(ci1, 1).ravel() / 2 / self.res1.bse
        crit2 = np.diff(ci1, 1).ravel() / 2 / self.res1.bse
        assert_allclose(crit1, crit2, rtol=12)


    def test_ttest(self):
        res1 = self.res1
        res2 = self.res2
        rtol = getattr(self, 'rtol', 1e-10)
        rtolh = getattr(self, 'rtol', 1e-12)

        mat = np.eye(len(res1.params))
        tt = res1.t_test(mat, cov_p=self.cov_robust)
        # has 'effect', 'pvalue', 'sd', 'tvalue'
        # TODO confint missing
        assert_allclose(tt.effect, res2.params, rtol=rtolh)
        assert_allclose(tt.sd, res2.bse, rtol=rtol)
        assert_allclose(tt.tvalue, res2.tvalues, rtol=rtolh)
        assert_allclose(tt.pvalue, res2.pvalues, rtol=5 * rtol)
        ci1 = tt.conf_int()
        ci2 = self.res2.params_table[:,4:6]
        assert_allclose(ci1, ci2, rtol=rtol)


    def test_smoke(self):
        self.res1.summary()



class TestOLSRobust2SmallNew(TestOLSRobust1, CheckOLSRobustNewMixin):
    # compare with ivreg robust small

    def setup(self):
        res_ols = self.res1.get_robustcov_results('HC1', use_t=True)
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        self.bse_robust2 = res_ols.HC1_se
        self.cov_robust2 = res_ols.cov_HC1
        self.small = True
        self.res2 = res.results_ivhc0_small

    def test_compare(self):
        #check that we get a warning using the nested compare methods
        res1 = self.res1
        endog = res1.model.endog
        exog = res1.model.exog[:, [0, 2]]  # drop one variable
        res_ols2 = OLS(endog, exog).fit()
        # results from Stata
        r_pval =  .0307306938402991
        r_chi2 =  4.667944083588736
        r_df =  1
        assert_warns(UserWarning, res1.compare_lr_test, res_ols2)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chi2, pval, df = res1.compare_lr_test(res_ols2)
        assert_allclose(chi2, r_chi2, rtol=1e-11)
        assert_allclose(pval, r_pval, rtol=1e-11)
        assert_equal(df, r_df)

        assert_warns(UserWarning, res1.compare_f_test, res_ols2)
        #fva, pval, df = res1.compare_f_test(res_ols2)



class TestOLSRobustHACSmallNew(TestOLSRobust1, CheckOLSRobustNewMixin):
    # compare with ivreg robust small

    def setup(self):
        res_ols = self.res1.get_robustcov_results('HAC', maxlags=4,
                                            use_correction=True, use_t=True)
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


class TestOLSRobust2LargeNew(TestOLSRobust1, CheckOLSRobustNewMixin):
    # compare with ivreg robust small

    def setup(self):
        res_ols = self.res1.get_robustcov_results('HC0')
        res_ols.use_t = False
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        self.bse_robust2 = res_ols.HC0_se
        self.cov_robust2 = res_ols.cov_HC0
        self.small = False
        self.res2 = res.results_ivhc0_large


    # TODO: skipping next two for now, not refactored yet for `large`
    def test_fvalue(self):
        pass


    def test_confint(self):
        pass

#######################################################
#    cluster robust standard errors
#######################################################


class CheckOLSRobustCluster(CheckOLSRobust):
    # compare with regress robust


    @classmethod
    def setup_class(cls):
        #import pandas as pa
        from statsmodels.datasets import grunfeld

        dtapa = grunfeld.data.load_pandas()
        #Stata example/data seems to miss last firm
        dtapa_endog = dtapa.endog[:200]
        dtapa_exog = dtapa.exog[:200]
        exog = add_constant(dtapa_exog[['value', 'capital']], prepend=False)
        #asserts don't work for pandas
        cls.res1 = OLS(dtapa_endog, exog).fit()

        firm_names, firm_id = np.unique(np.asarray(dtapa_exog[['firm']], 'S20'),
                                    return_inverse=True)
        cls.groups = firm_id
        #time indicator in range(max Ti)
        time = np.asarray(dtapa_exog[['year']])
        time -= time.min()
        cls.time = np.squeeze(time).astype(int)
        # nw_panel function requires interval bounds
        cls.tidx = [(i*20, 20*(i+1)) for i in range(10)]



class TestOLSRobustCluster2(CheckOLSRobustCluster, CheckOLSRobustNewMixin):
    # compare with `reg cluster`

    def setup(self):
        res_ols = self.res1.get_robustcov_results('cluster',
                                                  groups=self.groups,
                                                  use_correction=True,
                                                  use_t=True)
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        cov1 = sw.cov_cluster(self.res1, self.groups, use_correction=True)
        se1 =  sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1
        self.small = True
        self.res2 = res2.results_cluster

        self.rtol = 1e-6
        self.rtolh = 1e-10


class TestOLSRobustCluster2Large(CheckOLSRobustCluster, CheckOLSRobustNewMixin):
    # compare with `reg cluster`

    def setup(self):
        res_ols = self.res1.get_robustcov_results('cluster',
                                                  groups=self.groups,
                                                  use_correction=False,
                                                  use_t=False,
                                                  df_correction=True)
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        cov1 = sw.cov_cluster(self.res1, self.groups, use_correction=False)
        se1 =  sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1
        self.small = False
        self.res2 = res2.results_cluster_large

        self.skip_f = True
        self.rtol = 1e-6
        self.rtolh = 1e-10

    # skipping see https://github.com/statsmodels/statsmodels/pull/1189#issuecomment-29141741
    def test_fvalue(self):
        pass



class TestOLSRobustClusterGS(CheckOLSRobustCluster, CheckOLSRobustNewMixin):
    # compare with `reg cluster`

    def setup(self):
        res_ols = self.res1.get_robustcov_results('nw-groupsum',
                                                  time=self.time,
                                                  maxlags=4,
                                                  use_correction=False,
                                                  use_t=True)
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        cov1 = sw.cov_nw_groupsum(self.res1, 4, self.time, use_correction=False)
        se1 =  sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1
        self.small = True
        self.res2 = res2.results_nw_groupsum4

        self.skip_f = True
        self.rtol = 1e-6
        self.rtolh = 1e-10


class TestOLSRobustClusterNWP(CheckOLSRobustCluster, CheckOLSRobustNewMixin):
    # compare with `reg cluster`

    def setup(self):
        res_ols = self.res1.get_robustcov_results('nw-panel',
                                                  time=self.time,
                                                  maxlags=4,
                                                  use_correction='hac',
                                                  use_t=True,
                                                  df_correction=False)
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        cov1 = sw.cov_nw_panel(self.res1, 4, self.tidx)
        se1 =  sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1
        self.small = True
        self.res2 = res2.results_nw_panel4

        self.skip_f = True
        self.rtol = 1e-6
        self.rtolh = 1e-10


# TODO: low precision/agreement
class TestOLSRobustCluster2G(CheckOLSRobustCluster, CheckOLSRobustNewMixin):
    # compare with `reg cluster`

    def setup(self):
        res_ols = self.res1.get_robustcov_results('cluster',
                                                  groups=(self.groups, self.time),
                                                  use_correction=True,
                                                  use_t=True)
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        cov1 = sw.cov_cluster_2groups(self.res1, self.groups, group2=self.time,
                                       use_correction=True)[0]
        se1 =  sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1
        self.small = True
        self.res2 = res2.results_cluster_2groups_small

        self.rtol = 0.35 # only f_pvalue and confint for constant differ >rtol=0.05
        self.rtolh = 1e-10



class TestOLSRobustCluster2GLarge(CheckOLSRobustCluster, CheckOLSRobustNewMixin):
    # compare with `reg cluster`

    def setup(self):
        res_ols = self.res1.get_robustcov_results('cluster',
                                                  groups=(self.groups, self.time),
                                                  use_correction=False, #True,
                                                  use_t=False)
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        cov1 = sw.cov_cluster_2groups(self.res1, self.groups, group2=self.time,
                                       use_correction=False)[0]
        se1 =  sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1
        self.small = False
        self.res2 = res2.results_cluster_2groups_large

        self.skip_f = True
        self.rtol = 1e-7
        self.rtolh = 1e-10


######################################
#                 WLS
######################################

class CheckWLSRobustCluster(CheckOLSRobust):
    # compare with regress robust


    @classmethod
    def setup_class(cls):
        #import pandas as pa
        from statsmodels.datasets import grunfeld

        dtapa = grunfeld.data.load_pandas()
        #Stata example/data seems to miss last firm
        dtapa_endog = dtapa.endog[:200]
        dtapa_exog = dtapa.exog[:200]
        exog = add_constant(dtapa_exog[['value', 'capital']], prepend=False)
        #asserts don't work for pandas
        cls.res1 = WLS(dtapa_endog, exog, weights=1/dtapa_exog['value']).fit()

        firm_names, firm_id = np.unique(np.asarray(dtapa_exog[['firm']], 'S20'),
                                    return_inverse=True)
        cls.groups = firm_id
        #time indicator in range(max Ti)
        time = np.asarray(dtapa_exog[['year']])
        time -= time.min()
        cls.time = np.squeeze(time).astype(int)
        # nw_panel function requires interval bounds
        cls.tidx = [(i*20, 20*(i+1)) for i in range(10)]


# not available yet for WLS
class T_estWLSRobustCluster2(CheckWLSRobustCluster, CheckOLSRobustNewMixin):
    # compare with `reg cluster`

    def setup(self):
        res_ols = self.res1.get_robustcov_results('cluster',
                                                  groups=self.groups,
                                                  use_correction=True,
                                                  use_t=True)
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        cov1 = sw.cov_cluster(self.res1, self.groups, use_correction=True)
        se1 =  sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1
        self.small = True
        self.res2 = res2.results_cluster_wls_small

        self.rtol = 1e-6
        self.rtolh = 1e-10


# not available yet for WLS
class T_estWLSRobustCluster2Large(CheckWLSRobustCluster, CheckOLSRobustNewMixin):
    # compare with `reg cluster`

    def setup(self):
        res_ols = self.res1.get_robustcov_results('cluster',
                                                  groups=self.groups,
                                                  use_correction=False,
                                                  use_t=False,
                                                  df_correction=True)
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        cov1 = sw.cov_cluster(self.res1, self.groups, use_correction=False)
        se1 =  sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1
        self.small = False
        self.res2 = res2.results_cluster_wls_large

        self.skip_f = True
        self.rtol = 1e-6
        self.rtolh = 1e-10

class TestWLSRobustSmall(CheckWLSRobustCluster, CheckOLSRobustNewMixin):
    # compare with `reg cluster`

    def setup(self):
        res_ols = self.res1.get_robustcov_results('HC1',
                                                  use_t=True)
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        #TODO: check standalone function
        #cov1 = sw.cov_cluster(self.res1, self.groups, use_correction=False)
        cov1 = res_ols.cov_HC1
        se1 =  sw.se_cov(cov1)
        self.bse_robust2 = se1
        self.cov_robust2 = cov1
        self.small = True
        self.res2 = res2.results_hc1_wls_small

        self.skip_f = True
        self.rtol = 1e-6
        self.rtolh = 1e-10


class TestWLSOLSRobustSmall(object):

    @classmethod
    def setup_class(cls):
        #import pandas as pa
        from statsmodels.datasets import grunfeld

        dtapa = grunfeld.data.load_pandas()
        #Stata example/data seems to miss last firm
        dtapa_endog = dtapa.endog[:200]
        dtapa_exog = dtapa.exog[:200]
        exog = add_constant(dtapa_exog[['value', 'capital']], prepend=False)
        #asserts don't work for pandas
        cls.res_wls = WLS(dtapa_endog, exog, weights=1/dtapa_exog['value']).fit()
        w_sqrt = 1 / np.sqrt(np.asarray(dtapa_exog['value']))
        cls.res_ols = WLS(dtapa_endog * w_sqrt,
                          np.asarray(exog) * w_sqrt[:, None]).fit()

        firm_names, firm_id = np.unique(np.asarray(dtapa_exog[['firm']], 'S20'),
                                    return_inverse=True)
        cls.groups = firm_id
        #time indicator in range(max Ti)
        time = np.asarray(dtapa_exog[['year']])
        time -= time.min()
        cls.time = np.squeeze(time).astype(int)
        # nw_panel function requires interval bounds
        cls.tidx = [(i*20, 20*(i+1)) for i in range(10)]


    def test_all(self):
        all_cov = [('HC0', dict(use_t=True)),
                   ('HC1', dict(use_t=True)),
                   ('HC2', dict(use_t=True)),
                   ('HC3', dict(use_t=True))]

        # fvalue are not the same
        #res_ols = self.res_ols
        #res_wls = self.res_wls
        #assert_allclose(res_ols.fvalue, res_wls.fvalue, rtol=1e-13)

        for cov_type, kwds in all_cov:
            res1 = self.res_ols.get_robustcov_results(cov_type, **kwds)
            res2 = self.res_wls.get_robustcov_results(cov_type, **kwds)
            assert_allclose(res1.params, res2.params, rtol=1e-13)
            assert_allclose(res1.cov_params(), res2.cov_params(), rtol=1e-13)
            assert_allclose(res1.bse, res2.bse, rtol=1e-13)
            assert_allclose(res1.pvalues, res2.pvalues, rtol=1e-13)
            #Note: Fvalue doesn't match up, difference in calculation ?
            #      The only difference should be in the constant detection
            #assert_allclose(res1.fvalue, res2.fvalue, rtol=1e-13)
            #assert_allclose(res1.f_value, res2.f_pvalue, rtol=1e-13)
            mat = np.eye(len(res1.params))
            ft1 = res1.f_test(mat)
            ft2 = res2.f_test(mat)
            assert_allclose(ft1.fvalue, ft2.fvalue, rtol=1e-13)
            assert_allclose(ft1.pvalue, ft2.pvalue, rtol=1e-13)
