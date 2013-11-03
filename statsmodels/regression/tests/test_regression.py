"""
Test functions for models.regression
"""
# TODO: Test for LM

import warnings
import pandas
import numpy as np
from numpy.testing import (assert_almost_equal, assert_approx_equal,
                            assert_raises, assert_equal, assert_allclose)
from scipy.linalg import toeplitz
from statsmodels.tools.tools import add_constant, categorical
from statsmodels.regression.linear_model import OLS, WLS, GLS, yule_walker
from statsmodels.datasets import longley
from scipy.stats import t as student_t

DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2
DECIMAL_1 = 1
DECIMAL_7 = 7
DECIMAL_0 = 0


class CheckRegressionResults(object):
    '''
    res2 contains results from Rmodelwrap or were obtained from a statistical
    packages such as R, Stata, or SAS and were written to model_results
    '''

    decimal_params = DECIMAL_4
    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params,
                self.decimal_params)

    decimal_standarderrors = DECIMAL_4
    def test_standarderrors(self):
        assert_almost_equal(self.res1.bse,self.res2.bse,
                self.decimal_standarderrors)

    decimal_confidenceintervals = DECIMAL_4
    def test_confidenceintervals(self):
    #NOTE: stata rounds residuals (at least) to sig digits so approx_equal
        conf1 = self.res1.conf_int()
        conf2 = self.res2.conf_int()
        for i in range(len(conf1)):
            assert_approx_equal(conf1[i][0], conf2[i][0],
                    self.decimal_confidenceintervals)
            assert_approx_equal(conf1[i][1], conf2[i][1],
                    self.decimal_confidenceintervals)

    decimal_conf_int_subset = DECIMAL_4
    def test_conf_int_subset(self):
        if len(self.res1.params) > 1:
            ci1 = self.res1.conf_int(cols=(1,2))
            ci2 = self.res1.conf_int()[1:3]
            assert_almost_equal(ci1, ci2, self.decimal_conf_int_subset)
        else:
            pass

    decimal_scale = DECIMAL_4
    def test_scale(self):
        assert_almost_equal(self.res1.scale, self.res2.scale,
                self.decimal_scale)

    decimal_rsquared = DECIMAL_4
    def test_rsquared(self):
        assert_almost_equal(self.res1.rsquared, self.res2.rsquared,
                self.decimal_rsquared)

    decimal_rsquared_adj = DECIMAL_4
    def test_rsquared_adj(self):
        assert_almost_equal(self.res1.rsquared_adj, self.res2.rsquared_adj,
                    self.decimal_rsquared_adj)

    def test_degrees(self):
        assert_equal(self.res1.model.df_model, self.res2.df_model)
        assert_equal(self.res1.model.df_resid, self.res2.df_resid)

    decimal_ess = DECIMAL_4
    def test_ess(self):
        #Explained Sum of Squares
        assert_almost_equal(self.res1.ess, self.res2.ess,
                    self.decimal_ess)

    decimal_ssr = DECIMAL_4
    def test_sumof_squaredresids(self):
        assert_almost_equal(self.res1.ssr, self.res2.ssr, self.decimal_ssr)

    decimal_mse_resid = DECIMAL_4
    def test_mse_resid(self):
        #Mean squared error of residuals
        assert_almost_equal(self.res1.mse_model, self.res2.mse_model,
                    self.decimal_mse_resid)

    decimal_mse_model = DECIMAL_4
    def test_mse_model(self):
        assert_almost_equal(self.res1.mse_resid, self.res2.mse_resid,
                    self.decimal_mse_model)

    decimal_mse_total = DECIMAL_4
    def test_mse_total(self):
        assert_almost_equal(self.res1.mse_total, self.res2.mse_total,
                    self.decimal_mse_total, err_msg="Test class %s" % self)

    decimal_fvalue = DECIMAL_4
    def test_fvalue(self):
        #didn't change this, not sure it should complain -inf not equal -inf
        #if not (np.isinf(self.res1.fvalue) and np.isinf(self.res2.fvalue)):
        assert_almost_equal(self.res1.fvalue, self.res2.fvalue,
                self.decimal_fvalue)

    decimal_loglike = DECIMAL_4
    def test_loglike(self):
        assert_almost_equal(self.res1.llf, self.res2.llf, self.decimal_loglike)

    decimal_aic = DECIMAL_4
    def test_aic(self):
        assert_almost_equal(self.res1.aic, self.res2.aic, self.decimal_aic)

    decimal_bic = DECIMAL_4
    def test_bic(self):
        assert_almost_equal(self.res1.bic, self.res2.bic, self.decimal_bic)

    decimal_pvalues = DECIMAL_4
    def test_pvalues(self):
        assert_almost_equal(self.res1.pvalues, self.res2.pvalues,
            self.decimal_pvalues)

    decimal_wresid = DECIMAL_4
    def test_wresid(self):
        assert_almost_equal(self.res1.wresid, self.res2.wresid,
            self.decimal_wresid)

    decimal_resids = DECIMAL_4
    def test_resids(self):
        assert_almost_equal(self.res1.resid, self.res2.resid,
            self.decimal_resids)

#TODO: test fittedvalues and what else?

class TestOLS(CheckRegressionResults):
    @classmethod
    def setupClass(cls):
        from results.results_regression import Longley
        data = longley.load()
        data.exog = add_constant(data.exog, prepend=False)
        res1 = OLS(data.endog, data.exog).fit()
        res2 = Longley()
        res2.wresid = res1.wresid # workaround hack
        cls.res1 = res1
        cls.res2 = res2

        res_qr = OLS(data.endog, data.exog).fit(method="qr")
        cls.res_qr = res_qr

    #  Robust error tests.  Compare values computed with SAS
    def test_HC0_errors(self):
        #They are split up because the copied results do not have any DECIMAL_4
        #places for the last place.
        assert_almost_equal(self.res1.HC0_se[:-1],
                self.res2.HC0_se[:-1], DECIMAL_4)
        assert_approx_equal(np.round(self.res1.HC0_se[-1]), self.res2.HC0_se[-1])

    def test_HC1_errors(self):
        assert_almost_equal(self.res1.HC1_se[:-1],
                self.res2.HC1_se[:-1], DECIMAL_4)
        assert_approx_equal(self.res1.HC1_se[-1], self.res2.HC1_se[-1])

    def test_HC2_errors(self):
        assert_almost_equal(self.res1.HC2_se[:-1],
                self.res2.HC2_se[:-1], DECIMAL_4)
        assert_approx_equal(self.res1.HC2_se[-1], self.res2.HC2_se[-1])

    def test_HC3_errors(self):
        assert_almost_equal(self.res1.HC3_se[:-1],
                self.res2.HC3_se[:-1], DECIMAL_4)
        assert_approx_equal(self.res1.HC3_se[-1], self.res2.HC3_se[-1])

    def test_qr_params(self):
        assert_almost_equal(self.res1.params,
                self.res_qr.params, 6)

    def test_qr_normalized_cov_params(self):
        #todo: need assert_close
        assert_almost_equal(np.ones_like(self.res1.normalized_cov_params),
                self.res1.normalized_cov_params /
                self.res_qr.normalized_cov_params, 5)

    def test_missing(self):
        data = longley.load()
        data.exog = add_constant(data.exog, prepend=False)
        data.endog[[3, 7, 14]] = np.nan
        mod = OLS(data.endog, data.exog, missing='drop')
        assert_equal(mod.endog.shape[0], 13)
        assert_equal(mod.exog.shape[0], 13)

    def test_rsquared_adj_overfit(self):
        # Test that if df_resid = 0, rsquared_adj = 0.
        # This is a regression test for user issue:
        # https://github.com/statsmodels/statsmodels/issues/868
        with warnings.catch_warnings(record=True):
            x = np.random.randn(5)
            y = np.random.randn(5, 6)
            results = OLS(x, y).fit()
            rsquared_adj = results.rsquared_adj
            assert_equal(rsquared_adj, np.nan)

class TestRTO(CheckRegressionResults):
    @classmethod
    def setupClass(cls):
        from results.results_regression import LongleyRTO
        data = longley.load()
        res1 = OLS(data.endog, data.exog).fit()
        res2 = LongleyRTO()
        res2.wresid = res1.wresid # workaround hack
        cls.res1 = res1
        cls.res2 = res2

        res_qr = OLS(data.endog, data.exog).fit(method="qr")
        cls.res_qr = res_qr

class TestFtest(object):
    """
    Tests f_test vs. RegressionResults
    """
    @classmethod
    def setupClass(cls):
        data = longley.load()
        data.exog = add_constant(data.exog, prepend=False)
        cls.res1 = OLS(data.endog, data.exog).fit()
        R = np.identity(7)[:-1,:]
        cls.Ftest = cls.res1.f_test(R)

    def test_F(self):
        assert_almost_equal(self.Ftest.fvalue, self.res1.fvalue, DECIMAL_4)

    def test_p(self):
        assert_almost_equal(self.Ftest.pvalue, self.res1.f_pvalue, DECIMAL_4)

    def test_Df_denom(self):
        assert_equal(self.Ftest.df_denom, self.res1.model.df_resid)

    def test_Df_num(self):
        assert_equal(self.Ftest.df_num, 6)

class TestFTest2(object):
    '''
    A joint test that the coefficient on
    GNP = the coefficient on UNEMP  and that the coefficient on
    POP = the coefficient on YEAR for the Longley dataset.

    Ftest1 is from statsmodels.  Results are from Rpy using R's car library.
    '''
    @classmethod
    def setupClass(cls):
        data = longley.load()
        data.exog = add_constant(data.exog, prepend=False)
        res1 = OLS(data.endog, data.exog).fit()
        R2 = [[0,1,-1,0,0,0,0],[0, 0, 0, 0, 1, -1, 0]]
        cls.Ftest1 = res1.f_test(R2)
        hyp = 'x2 = x3, x5 = x6'
        cls.NewFtest1 = res1.f_test(hyp)

    def test_new_ftest(self):
        assert_equal(self.NewFtest1.fvalue, self.Ftest1.fvalue)

    def test_fvalue(self):
        assert_almost_equal(self.Ftest1.fvalue, 9.7404618732968196, DECIMAL_4)

    def test_pvalue(self):
        assert_almost_equal(self.Ftest1.pvalue, 0.0056052885317493459,
                DECIMAL_4)

    def test_df_denom(self):
        assert_equal(self.Ftest1.df_denom, 9)

    def test_df_num(self):
        assert_equal(self.Ftest1.df_num, 2)

class TestFtestQ(object):
    """
    A joint hypothesis test that Rb = q.  Coefficient tests are essentially
    made up.  Test values taken from Stata.
    """
    @classmethod
    def setupClass(cls):
        data = longley.load()
        data.exog = add_constant(data.exog, prepend=False)
        res1 = OLS(data.endog, data.exog).fit()
        R = np.array([[0,1,1,0,0,0,0],
              [0,1,0,1,0,0,0],
              [0,1,0,0,0,0,0],
              [0,0,0,0,1,0,0],
              [0,0,0,0,0,1,0]])
        q = np.array([0,0,0,1,0])
        cls.Ftest1 = res1.f_test((R,q))

    def test_fvalue(self):
        assert_almost_equal(self.Ftest1.fvalue, 70.115557, 5)

    def test_pvalue(self):
        assert_almost_equal(self.Ftest1.pvalue, 6.229e-07, 10)

    def test_df_denom(self):
        assert_equal(self.Ftest1.df_denom, 9)

    def test_df_num(self):
        assert_equal(self.Ftest1.df_num, 5)

class TestTtest(object):
    '''
    Test individual t-tests.  Ie., are the coefficients significantly
    different than zero.

        '''
    @classmethod
    def setupClass(cls):
        data = longley.load()
        data.exog = add_constant(data.exog, prepend=False)
        cls.res1 = OLS(data.endog, data.exog).fit()
        R = np.identity(7)
        cls.Ttest = cls.res1.t_test(R)
        hyp = 'x1 = 0, x2 = 0, x3 = 0, x4 = 0, x5 = 0, x6 = 0, const = 0'
        cls.NewTTest = cls.res1.t_test(hyp)

    def test_new_tvalue(self):
        assert_equal(self.NewTTest.tvalue, self.Ttest.tvalue)

    def test_tvalue(self):
        assert_almost_equal(self.Ttest.tvalue, self.res1.tvalues, DECIMAL_4)

    def test_sd(self):
        assert_almost_equal(self.Ttest.sd, self.res1.bse, DECIMAL_4)

    def test_pvalue(self):
        assert_almost_equal(self.Ttest.pvalue, student_t.sf(
                        np.abs(self.res1.tvalues), self.res1.model.df_resid)*2,
                        DECIMAL_4)

    def test_df_denom(self):
        assert_equal(self.Ttest.df_denom, self.res1.model.df_resid)

    def test_effect(self):
        assert_almost_equal(self.Ttest.effect, self.res1.params)

class TestTtest2(object):
    '''
    Tests the hypothesis that the coefficients on POP and YEAR
    are equal.

    Results from RPy using 'car' package.
    '''
    @classmethod
    def setupClass(cls):
        R = np.zeros(7)
        R[4:6] = [1,-1]
        data = longley.load()
        data.exog = add_constant(data.exog, prepend=False)
        res1 = OLS(data.endog, data.exog).fit()
        cls.Ttest1 = res1.t_test(R)

    def test_tvalue(self):
        assert_almost_equal(self.Ttest1.tvalue, -4.0167754636397284,
                DECIMAL_4)

    def test_sd(self):
        assert_almost_equal(self.Ttest1.sd, 455.39079425195314, DECIMAL_4)

    def test_pvalue(self):
        assert_almost_equal(self.Ttest1.pvalue, 2*0.0015163772380932246,
            DECIMAL_4)

    def test_df_denom(self):
        assert_equal(self.Ttest1.df_denom, 9)

    def test_effect(self):
        assert_almost_equal(self.Ttest1.effect, -1829.2025687186533, DECIMAL_4)

class TestGLS(object):
    '''
    These test results were obtained by replication with R.
    '''
    @classmethod
    def setupClass(cls):
        from results.results_regression import LongleyGls

        data = longley.load()
        exog = add_constant(np.column_stack((data.exog[:,1],
                                             data.exog[:,4])), prepend=False)
        tmp_results = OLS(data.endog, exog).fit()
        rho = np.corrcoef(tmp_results.resid[1:],
                tmp_results.resid[:-1])[0][1] # by assumption
        order = toeplitz(np.arange(16))
        sigma = rho**order
        GLS_results = GLS(data.endog, exog, sigma=sigma).fit()
        cls.res1 = GLS_results
        cls.res2 = LongleyGls()
        # attach for test_missing
        cls.sigma = sigma
        cls.exog = exog
        cls.endog = data.endog

    def test_aic(self):
        assert_approx_equal(self.res1.aic+2, self.res2.aic, 3)

    def test_bic(self):
        assert_approx_equal(self.res1.bic, self.res2.bic, 2)

    def test_loglike(self):
        assert_almost_equal(self.res1.llf, self.res2.llf, DECIMAL_0)

    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_1)

    def test_resid(self):
        assert_almost_equal(self.res1.resid, self.res2.resid, DECIMAL_4)

    def test_scale(self):
        assert_almost_equal(self.res1.scale, self.res2.scale, DECIMAL_4)

    def test_tvalues(self):
        assert_almost_equal(self.res1.tvalues, self.res2.tvalues, DECIMAL_4)

    def test_standarderrors(self):
        assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL_4)

    def test_fittedvalues(self):
        assert_almost_equal(self.res1.fittedvalues, self.res2.fittedvalues,
                DECIMAL_4)

    def test_pvalues(self):
        assert_almost_equal(self.res1.pvalues, self.res2.pvalues, DECIMAL_4)

    def test_missing(self):
        endog = self.endog.copy() # copy or changes endog for other methods
        endog[[4,7,14]] = np.nan
        mod = GLS(endog, self.exog, sigma=self.sigma, missing='drop')
        assert_equal(mod.endog.shape[0], 13)
        assert_equal(mod.exog.shape[0], 13)
        assert_equal(mod.sigma.shape, (13,13))

class TestGLS_nosigma(CheckRegressionResults):
    '''
    Test that GLS with no argument is equivalent to OLS.
    '''
    @classmethod
    def setupClass(cls):
        data = longley.load()
        data.exog = add_constant(data.exog, prepend=False)
        ols_res = OLS(data.endog, data.exog).fit()
        gls_res = GLS(data.endog, data.exog).fit()
        cls.res1 = gls_res
        cls.res2 = ols_res
#        self.res2.conf_int = self.res2.conf_int()

#    def check_confidenceintervals(self, conf1, conf2):
#        assert_almost_equal(conf1, conf2, DECIMAL_4)

class TestLM(object):
    @classmethod
    def setupClass(cls):
        # TODO: Test HAC method
        X = np.random.randn(100,3)
        b = np.ones((3,1))
        e = np.random.randn(100,1)
        y = np.dot(X,b) + e
        # Cases?
        # Homoskedastic
        # HC0
        cls.res1_full = OLS(y,X).fit()
        cls.res1_restricted = OLS(y,X[:,0]).fit()

        cls.res2_full = cls.res1_full.get_robustcov_results('HC0')
        cls.res2_restricted = cls.res1_restricted.get_robustcov_results('HC0')

        cls.X = X
        cls.Y = y

        def test_LM_homoskedastic(self):
            resid = self.res1_restricted.wresid
            n = resid.shape[0]
            X = self.X
            S = np.dot(resid,resid) / n * np.dot(X.T,X) / n
            Sinv = np.linalg.inv(S)
            s = np.mean(X * resid[:,None], 0)
            LMstat = n * np.dot(np.dot(s,Sinv),s.T)
            LMstat_OLS = self.res1_full.compare_lm_test(self.res1_restricted)
            LMstat2 = LMstat_OLS[0]
            assert_almost_equal(LMstat, LMstat2, DECIMAL_7)

        def test_LM_heteroskedastic_nodemean(self):
            resid = self.res1_restricted.wresid
            n = resid.shape[0]
            X = self.X
            scores = X * resid[:,None]
            S = np.dot(scores.T,scores) / n
            Sinv = np.linalg.inv(S)
            s = np.mean(scores, 0)
            LMstat = n * np.dot(np.dot(s,Sinv),s.T)
            LMstat_OLS = self.res2_full.compare_lm_test(self.res2_restricted, demean=False)
            LMstat2 = LMstat_OLS[0]
            assert_almost_equal(LMstat, LMstat2, DECIMAL_7)

        def test_LM_heteroskedastic_demean(self):
            resid = self.res1_restricted.wresid
            n = resid.shape[0]
            X = self.X
            scores = X * resid[:,None]
            scores_demean = scores - scores.mean(0)
            S = np.dot(scores_demean.T,scores_demean) / n
            Sinv = np.linalg.inv(S)
            s = np.mean(scores, 0)
            LMstat = n * np.dot(np.dot(s,Sinv),s.T)
            LMstat_OLS = self.res2_full.compare_lm_test(self.res2_restricted)
            LMstat2 = LMstat_OLS[0]
            assert_almost_equal(LMstat, LMstat2, DECIMAL_7)

        def test_LM_heteroskedastic_LRversion(self):
            resid = self.res1_restricted.wresid
            resid_full = self.res1_full.wresid
            n = resid.shape[0]
            X = self.X
            scores = X * resid[:,None]
            s = np.mean(scores, 0)
            scores = X * resid_full[:,None]
            S = np.dot(scores.T,scores) / n
            Sinv = np.linalg.inv(S)
            LMstat = n * np.dot(np.dot(s,Sinv),s.T)
            LMstat_OLS = self.res2_full.compare_lm_test(self.res2_restricted, use_lr = True)
            LMstat2 = LMstat_OLS[0]
            assert_almost_equal(LMstat, LMstat2, DECIMAL_7)

            pass


        n = y.shape[0]
        w = np.ones(n)
        cls.results = []
        cls.results.append(OLS(y, X).fit())
        cls.results.append(WLS(y, X, w).fit())
        cls.results.append(WLS(y, X, 100.0*w).fit())
        cls.results.append(GLS(y, X, 0.5*w).fit())
        cls.results.append(GLS(y, X, np.diag(2.0*w)).fit())



class TestOLS_GLS_WLS_equivalence(object):
    @classmethod
    def setupClass(cls):
        data = longley.load()
        data.exog = add_constant(data.exog, prepend=False)
        y = data.endog
        X = data.exog
        n = y.shape[0]
        w = np.ones(n)
        cls.results = []
        cls.results.append(OLS(y, X).fit())
        cls.results.append(WLS(y, X, w).fit())
        cls.results.append(WLS(y, X, 100.0*w).fit())
        cls.results.append(GLS(y, X, 0.5*w).fit())
        cls.results.append(GLS(y, X, np.diag(2.0*w)).fit())

    def test_ll(self):
        llf = np.array([r.llf for r in self.results])
        llf_1 = np.ones_like(llf) * self.results[0].llf
        assert_almost_equal(llf, llf_1, DECIMAL_7)

    def test_params(self):
        params = np.array([r.params for r in self.results])
        params_1 = np.array([self.results[0].params] * len(self.results))
        assert_allclose(params, params_1)

    def test_ss(self):
        bse = np.array([r.bse for r in self.results])
        bse_1 = np.array([self.results[0].bse] * len(self.results))
        assert_allclose(bse, bse_1)

    def test_rsquared(self):
        rsquared = np.array([r.rsquared for r in self.results])
        rsquared_1 = np.array([self.results[0].rsquared] * len(self.results))
        assert_almost_equal(rsquared, rsquared_1, DECIMAL_7)


class TestWLSExogWeights(CheckRegressionResults):
    #Test WLS with Greene's credit card data
    #reg avgexp age income incomesq ownrent [aw=1/incomesq]
    def __init__(self):
        from results.results_regression import CCardWLS
        from statsmodels.datasets.ccard import load
        dta = load()

        dta.exog = add_constant(dta.exog, prepend=False)
        nobs = 72.

        weights = 1/dta.exog[:,2]
        # for comparison with stata analytic weights
        scaled_weights = ((weights * nobs)/weights.sum())

        self.res1 = WLS(dta.endog, dta.exog, weights=scaled_weights).fit()
        self.res2 = CCardWLS()
        self.res2.wresid = scaled_weights ** .5 * self.res2.resid

def test_wls_example():
    #example from the docstring, there was a note about a bug, should
    #be fixed now
    Y = [1,3,4,5,2,3,4]
    X = range(1,8)
    X = add_constant(X, prepend=False)
    wls_model = WLS(Y,X, weights=range(1,8)).fit()
    #taken from R lm.summary
    assert_almost_equal(wls_model.fvalue, 0.127337843215, 6)
    assert_almost_equal(wls_model.scale, 2.44608530786**2, 6)

def test_wls_tss():
    y = np.array([22, 22, 22, 23, 23, 23])
    X = [[1, 0], [1, 0], [1, 1], [0, 1], [0, 1], [0, 1]]

    ols_mod = OLS(y, add_constant(X, prepend=False)).fit()

    yw = np.array([22, 22, 23.])
    Xw = [[1,0],[1,1],[0,1]]
    w = np.array([2, 1, 3.])

    wls_mod = WLS(yw, add_constant(Xw, prepend=False), weights=w).fit()
    assert_equal(ols_mod.centered_tss, wls_mod.centered_tss)

class TestWLSScalarVsArray(CheckRegressionResults):
    @classmethod
    def setupClass(cls):
        from statsmodels.datasets.longley import load
        dta = load()
        dta.exog = add_constant(dta.exog, prepend=True)
        wls_scalar = WLS(dta.endog, dta.exog, weights=1./3).fit()
        weights = [1/3.] * len(dta.endog)
        wls_array = WLS(dta.endog, dta.exog, weights=weights).fit()
        cls.res1 = wls_scalar
        cls.res2 = wls_array

#class TestWLS_GLS(CheckRegressionResults):
#    @classmethod
#    def setupClass(cls):
#        from statsmodels.datasets.ccard import load
#        data = load()
#        cls.res1 = WLS(data.endog, data.exog, weights = 1/data.exog[:,2]).fit()
#        cls.res2 = GLS(data.endog, data.exog, sigma = data.exog[:,2]).fit()
#
#    def check_confidenceintervals(self, conf1, conf2):
#        assert_almost_equal(conf1, conf2(), DECIMAL_4)

def test_wls_missing():
    from statsmodels.datasets.ccard import load
    data = load()
    endog = data.endog
    endog[[10, 25]] = np.nan
    mod = WLS(data.endog, data.exog, weights = 1/data.exog[:,2], missing='drop')
    assert_equal(mod.endog.shape[0], 70)
    assert_equal(mod.exog.shape[0], 70)
    assert_equal(mod.weights.shape[0], 70)



class TestWLS_OLS(CheckRegressionResults):
    @classmethod
    def setupClass(cls):
        data = longley.load()
        data.exog = add_constant(data.exog, prepend=False)
        cls.res1 = OLS(data.endog, data.exog).fit()
        cls.res2 = WLS(data.endog, data.exog).fit()

    def check_confidenceintervals(self, conf1, conf2):
        assert_almost_equal(conf1, conf2(), DECIMAL_4)

class TestGLS_OLS(CheckRegressionResults):
    @classmethod
    def setupClass(cls):
        data = longley.load()
        data.exog = add_constant(data.exog, prepend=False)
        cls.res1 = GLS(data.endog, data.exog).fit()
        cls.res2 = OLS(data.endog, data.exog).fit()

    def check_confidenceintervals(self, conf1, conf2):
        assert_almost_equal(conf1, conf2(), DECIMAL_4)

#TODO: test AR
# why the two-stage in AR?
#class test_ar(object):
#    from statsmodels.datasets.sunspots import load
#    data = load()
#    model = AR(data.endog, rho=4).fit()
#    R_res = RModel(data.endog, aic="FALSE", order_max=4)

#    def test_params(self):
#        assert_almost_equal(self.model.rho,
#        pass

#    def test_order(self):
# In R this can be defined or chosen by minimizing the AIC if aic=True
#        pass


class TestYuleWalker(object):
    @classmethod
    def setupClass(cls):
        from statsmodels.datasets.sunspots import load
        data = load()
        cls.rho, cls.sigma = yule_walker(data.endog, order=4,
                method="mle")
        cls.R_params = [1.2831003105694765, -0.45240924374091945,
                -0.20770298557575195, 0.047943648089542337]

    def test_params(self):
        assert_almost_equal(self.rho, self.R_params, DECIMAL_4)

class TestDataDimensions(CheckRegressionResults):
    @classmethod
    def setupClass(cls):
        np.random.seed(54321)
        cls.endog_n_ = np.random.uniform(0,20,size=30)
        cls.endog_n_one = cls.endog_n_[:,None]
        cls.exog_n_ = np.random.uniform(0,20,size=30)
        cls.exog_n_one = cls.exog_n_[:,None]
        cls.degen_exog = cls.exog_n_one[:-1]
        cls.mod1 = OLS(cls.endog_n_one, cls.exog_n_one)
        cls.mod1.df_model += 1
        cls.res1 = cls.mod1.fit()
        # Note that these are created for every subclass..
        # A little extra overhead probably
        cls.mod2 = OLS(cls.endog_n_one, cls.exog_n_one)
        cls.mod2.df_model += 1
        cls.res2 = cls.mod2.fit()

    def check_confidenceintervals(self, conf1, conf2):
        assert_almost_equal(conf1, conf2(), DECIMAL_4)

class TestNxNx(TestDataDimensions):
    @classmethod
    def setupClass(cls):
        super(TestNxNx, cls).setupClass()
        cls.mod2 = OLS(cls.endog_n_, cls.exog_n_)
        cls.mod2.df_model += 1
        cls.res2 = cls.mod2.fit()

class TestNxOneNx(TestDataDimensions):
    @classmethod
    def setupClass(cls):
        super(TestNxOneNx, cls).setupClass()
        cls.mod2 = OLS(cls.endog_n_one, cls.exog_n_)
        cls.mod2.df_model += 1
        cls.res2 = cls.mod2.fit()

class TestNxNxOne(TestDataDimensions):
    @classmethod
    def setupClass(cls):
        super(TestNxNxOne, cls).setupClass()
        cls.mod2 = OLS(cls.endog_n_, cls.exog_n_one)
        cls.mod2.df_model += 1
        cls.res2 = cls.mod2.fit()

def test_bad_size():
    np.random.seed(54321)
    data = np.random.uniform(0,20,31)
    assert_raises(ValueError, OLS, data, data[1:])

def test_const_indicator():
    np.random.seed(12345)
    X = np.random.randint(0, 3, size=30)
    X = categorical(X, drop=True)
    y = np.dot(X, [1., 2., 3.]) + np.random.normal(size=30)
    modc = OLS(y, add_constant(X[:,1:], prepend=True)).fit()
    mod = OLS(y, X, hasconst=True).fit()
    assert_almost_equal(modc.rsquared, mod.rsquared, 12)

def test_706():
    # make sure one regressor pandas Series gets passed to DataFrame
    # for conf_int.
    y = pandas.Series(np.random.randn(10))
    x = pandas.Series(np.ones(10))
    res = OLS(y,x).fit()
    conf_int = res.conf_int()
    np.testing.assert_equal(conf_int.shape, (1, 2))
    np.testing.assert_(isinstance(conf_int, pandas.DataFrame))

def test_summary():
    # test 734
    import re
    dta = longley.load_pandas()
    X = dta.exog
    X["constant"] = 1
    y = dta.endog
    with warnings.catch_warnings(record=True):
        res = OLS(y, X).fit()
        table = res.summary().as_latex()
    # replace the date and time
    table = re.sub("(?<=\n\\\\textbf\{Date:\}             &).+?&",
                   " Sun, 07 Apr 2013 &", table)
    table = re.sub("(?<=\n\\\\textbf\{Time:\}             &).+?&",
                   "     13:46:07     &", table)

    expected = """\\begin{center}
\\begin{tabular}{lclc}
\\toprule
\\textbf{Dep. Variable:}    &      TOTEMP      & \\textbf{  R-squared:         } &     0.995   \\\\
\\textbf{Model:}            &       OLS        & \\textbf{  Adj. R-squared:    } &     0.992   \\\\
\\textbf{Method:}           &  Least Squares   & \\textbf{  F-statistic:       } &     330.3   \\\\
\\textbf{Date:}             & Sun, 07 Apr 2013 & \\textbf{  Prob (F-statistic):} &  4.98e-10   \\\\
\\textbf{Time:}             &     13:46:07     & \\textbf{  Log-Likelihood:    } &   -109.62   \\\\
\\textbf{No. Observations:} &          16      & \\textbf{  AIC:               } &     233.2   \\\\
\\textbf{Df Residuals:}     &           9      & \\textbf{  BIC:               } &     238.6   \\\\
\\bottomrule
\\end{tabular}
\\begin{tabular}{lccccc}
                  & \\textbf{coef} & \\textbf{std err} & \\textbf{t} & \\textbf{P$>$$|$t$|$} & \\textbf{[95.0\\% Conf. Int.]}  \\\\
\\midrule
\\textbf{GNPDEFL}  &      15.0619  &       84.915     &     0.177  &         0.863        &      -177.029   207.153       \\\\
\\textbf{GNP}      &      -0.0358  &        0.033     &    -1.070  &         0.313        &        -0.112     0.040       \\\\
\\textbf{UNEMP}    &      -2.0202  &        0.488     &    -4.136  &         0.003        &        -3.125    -0.915       \\\\
\\textbf{ARMED}    &      -1.0332  &        0.214     &    -4.822  &         0.001        &        -1.518    -0.549       \\\\
\\textbf{POP}      &      -0.0511  &        0.226     &    -0.226  &         0.826        &        -0.563     0.460       \\\\
\\textbf{YEAR}     &    1829.1515  &      455.478     &     4.016  &         0.003        &       798.788  2859.515       \\\\
\\textbf{constant} &   -3.482e+06  &      8.9e+05     &    -3.911  &         0.004        &      -5.5e+06 -1.47e+06       \\\\
\\bottomrule
\\end{tabular}
\\begin{tabular}{lclc}
\\textbf{Omnibus:}       &  0.749 & \\textbf{  Durbin-Watson:     } &    2.559  \\\\
\\textbf{Prob(Omnibus):} &  0.688 & \\textbf{  Jarque-Bera (JB):  } &    0.684  \\\\
\\textbf{Skew:}          &  0.420 & \\textbf{  Prob(JB):          } &    0.710  \\\\
\\textbf{Kurtosis:}      &  2.434 & \\textbf{  Cond. No.          } & 4.86e+09  \\\\
\\bottomrule
\\end{tabular}
%\\caption{OLS Regression Results}
\\end{center}"""
    assert_equal(table, expected)




if __name__=="__main__":

    import nose
    # run_module_suite()
    #nose.runmodule(argv=[__file__,'-vvs','-x','--pdb', '--pdb-failure'], exit=False)
    nose.runmodule()
    nose.runmodule(argv=[__file__,'-vvs','-x'], exit=False) #, '--pdb'




