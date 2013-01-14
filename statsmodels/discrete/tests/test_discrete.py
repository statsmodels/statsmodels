"""
Tests for discrete models

Notes
-----
DECIMAL_3 is used because it seems that there is a loss of precision
in the Stata *.dta -> *.csv output, NOT the estimator for the Poisson
tests.
"""
import os
import numpy as np
from numpy.testing import *
from statsmodels.discrete.discrete_model import (Logit, Probit, MNLogit,
                                                 Poisson)
from statsmodels.discrete.discrete_margins import _iscount, _isdummy
import statsmodels.api as sm
from sys import platform
from nose import SkipTest
from results.results_discrete import Spector, DiscreteL1
from statsmodels.tools.sm_exceptions import PerfectSeparationError
try:
    import cvxopt
    has_cvxopt = True
except ImportError:
    has_cvxopt = False

DECIMAL_14 = 14
DECIMAL_10 = 10
DECIMAL_9 = 9
DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2
DECIMAL_1 = 1
DECIMAL_0 = 0
iswindows = 'win' in platform.lower()

class CheckModelResults(object):
    """
    res2 should be the test results from RModelWrap
    or the results as defined in model_results_data
    """
    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_4)

    def test_conf_int(self):
        assert_almost_equal(self.res1.conf_int(), self.res2.conf_int, DECIMAL_4)

    def test_zstat(self):
        assert_almost_equal(self.res1.tvalues, self.res2.z, DECIMAL_4)

    def pvalues(self):
        assert_almost_equal(self.res1.pvalues, self.res2.pvalues, DECIMAL_4)

#    def test_cov_params(self):
#        assert_almost_equal(self.res1.cov_params(), self.res2.cov_params,
#                DECIMAL_4)

    def test_llf(self):
        assert_almost_equal(self.res1.llf, self.res2.llf, DECIMAL_4)

    def test_llnull(self):
        assert_almost_equal(self.res1.llnull, self.res2.llnull, DECIMAL_4)

    def test_llr(self):
        assert_almost_equal(self.res1.llr, self.res2.llr, DECIMAL_3)

    def test_llr_pvalue(self):
        assert_almost_equal(self.res1.llr_pvalue, self.res2.llr_pvalue,
                DECIMAL_4)

    def test_normalized_cov_params(self):
        pass

    def test_bse(self):
        assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL_4)

    def test_dof(self):
        assert_equal(self.res1.df_model, self.res2.df_model)
        assert_equal(self.res1.df_resid, self.res2.df_resid)

    def test_aic(self):
        assert_almost_equal(self.res1.aic, self.res2.aic, DECIMAL_3)

    def test_bic(self):
        assert_almost_equal(self.res1.bic, self.res2.bic, DECIMAL_3)

    def test_predict(self):
        assert_almost_equal(self.res1.model.predict(self.res1.params),
                            self.res2.phat, DECIMAL_4)

    def test_predict_xb(self):
        assert_almost_equal(self.res1.model.predict(self.res1.params,
                            linear=True),
                            self.res2.yhat, DECIMAL_4)

    def test_loglikeobs(self):
        #basic cross check
        llobssum = self.res1.model.loglikeobs(self.res1.params).sum()
        assert_almost_equal(llobssum, self.res1.llf, DECIMAL_14)

    def test_jac(self):
        #basic cross check
        jacsum = self.res1.model.jac(self.res1.params).sum(0)
        score = self.res1.model.score(self.res1.params)
        assert_almost_equal(jacsum, score, DECIMAL_9) #Poisson has low precision ?


class CheckBinaryResults(CheckModelResults):
    def test_pred_table(self):
        assert_array_equal(self.res1.pred_table(), self.res2.pred_table)

class CheckMargEff(object):
    """
    Test marginal effects (margeff) and its options
    """

    def test_nodummy_dydxoverall(self):
        me = self.res1.get_margeff()
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_dydx, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_dydx_se, DECIMAL_4)

    def test_nodummy_dydxmean(self):
        me = self.res1.get_margeff(at='mean')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_dydxmean, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_dydxmean_se, DECIMAL_4)

    def test_nodummy_dydxmedian(self):
        me = self.res1.get_margeff(at='median')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_dydxmedian, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_dydxmedian_se, DECIMAL_4)

    def test_nodummy_dydxzero(self):
        me = self.res1.get_margeff(at='zero')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_dydxzero, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_dydxzero, DECIMAL_4)

    def test_nodummy_dyexoverall(self):
        me = self.res1.get_margeff(method='dyex')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_dyex, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_dyex_se, DECIMAL_4)

    def test_nodummy_dyexmean(self):
        me = self.res1.get_margeff(at='mean', method='dyex')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_dyexmean, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_dyexmean_se, DECIMAL_4)

    def test_nodummy_dyexmedian(self):
        me = self.res1.get_margeff(at='median', method='dyex')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_dyexmedian, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_dyexmedian_se, DECIMAL_4)

    def test_nodummy_dyexzero(self):
        me = self.res1.get_margeff(at='zero', method='dyex')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_dyexzero, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_dyexzero_se, DECIMAL_4)

    def test_nodummy_eydxoverall(self):
        me = self.res1.get_margeff(method='eydx')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_eydx, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_eydx_se, DECIMAL_4)

    def test_nodummy_eydxmean(self):
        me = self.res1.get_margeff(at='mean', method='eydx')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_eydxmean, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_eydxmean_se, DECIMAL_4)

    def test_nodummy_eydxmedian(self):
        me = self.res1.get_margeff(at='median', method='eydx')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_eydxmedian, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_eydxmedian_se, DECIMAL_4)

    def test_nodummy_eydxzero(self):
        me = self.res1.get_margeff(at='zero', method='eydx')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_eydxzero, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_eydxzero_se, DECIMAL_4)

    def test_nodummy_eyexoverall(self):
        me = self.res1.get_margeff(method='eyex')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_eyex, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_eyex_se, DECIMAL_4)

    def test_nodummy_eyexmean(self):
        me = self.res1.get_margeff(at='mean', method='eyex')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_eyexmean, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_eyexmean_se, DECIMAL_4)

    def test_nodummy_eyexmedian(self):
        me = self.res1.get_margeff(at='median', method='eyex')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_eyexmedian, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_eyexmedian_se, DECIMAL_4)

    def test_nodummy_eyexzero(self):
        me = self.res1.get_margeff(at='zero', method='eyex')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_eyexzero, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_eyexzero_se, DECIMAL_4)

    def test_dummy_dydxoverall(self):
        me = self.res1.get_margeff(dummy=True)
        assert_almost_equal(me.margeff,
                self.res2.margeff_dummy_dydx, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_dummy_dydx_se, DECIMAL_4)

    def test_dummy_dydxmean(self):
        me = self.res1.get_margeff(at='mean', dummy=True)
        assert_almost_equal(me.margeff,
                self.res2.margeff_dummy_dydxmean, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_dummy_dydxmean_se, DECIMAL_4)

    def test_dummy_eydxoverall(self):
        me = self.res1.get_margeff(method='eydx', dummy=True)
        assert_almost_equal(me.margeff,
                self.res2.margeff_dummy_eydx, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_dummy_eydx_se, DECIMAL_4)

    def test_dummy_eydxmean(self):
        me = self.res1.get_margeff(at='mean', method='eydx', dummy=True)
        assert_almost_equal(me.margeff,
                self.res2.margeff_dummy_eydxmean, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_dummy_eydxmean_se, DECIMAL_4)

    def test_count_dydxoverall(self):
        me = self.res1.get_margeff(count=True)
        assert_almost_equal(me.margeff,
                self.res2.margeff_count_dydx, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_count_dydx_se, DECIMAL_4)

    def test_count_dydxmean(self):
        me = self.res1.get_margeff(count=True, at='mean')
        assert_almost_equal(me.margeff,
                self.res2.margeff_count_dydxmean, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_count_dydxmean_se, DECIMAL_4)

    def test_count_dummy_dydxoverall(self):
        me = self.res1.get_margeff(count=True, dummy=True)
        assert_almost_equal(me.margeff,
                self.res2.margeff_count_dummy_dydxoverall, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_count_dummy_dydxoverall_se, DECIMAL_4)

    def test_count_dummy_dydxmean(self):
        me = self.res1.get_margeff(count=True, dummy=True, at='mean')
        assert_almost_equal(me.margeff,
                self.res2.margeff_count_dummy_dydxmean, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_count_dummy_dydxmean_se, DECIMAL_4)

class TestProbitNewton(CheckBinaryResults):

    @classmethod
    def setupClass(cls):
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog, prepend=False)
        cls.res1 = Probit(data.endog, data.exog).fit(method="newton", disp=0)
        res2 = Spector()
        res2.probit()
        cls.res2 = res2

    #def test_predict(self):
    #    assert_almost_equal(self.res1.model.predict(self.res1.params),
    #            self.res2.predict, DECIMAL_4)

    def test_resid(self):
        assert_almost_equal(self.res1.resid, self.res2.resid, DECIMAL_4)


class TestProbitBFGS(CheckBinaryResults):

    @classmethod
    def setupClass(cls):
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog, prepend=False)
        cls.res1 = Probit(data.endog, data.exog).fit(method="bfgs",
            disp=0)
        res2 = Spector()
        res2.probit()
        cls.res2 = res2


class TestProbitNM(CheckBinaryResults):
    @classmethod
    def setupClass(cls):
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog, prepend=False)
        res2 = Spector()
        res2.probit()
        cls.res2 = res2
        cls.res1 = Probit(data.endog, data.exog).fit(method="nm",
            disp=0, maxiter=500)

class TestProbitPowell(CheckBinaryResults):
    @classmethod
    def setupClass(cls):
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog, prepend=False)
        res2 = Spector()
        res2.probit()
        cls.res2 = res2
        cls.res1 = Probit(data.endog, data.exog).fit(method="powell",
            disp=0, ftol=1e-8)

class TestProbitCG(CheckBinaryResults):
    @classmethod
    def setupClass(cls):
        if iswindows:   # does this work with classmethod?
            raise SkipTest("fmin_cg sometimes fails to converge on windows")
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog, prepend=False)
        res2 = Spector()
        res2.probit()
        cls.res2 = res2
        cls.res1 = Probit(data.endog, data.exog).fit(method="cg",
            disp=0, maxiter=500, gtol=1e-06)

class TestProbitNCG(CheckBinaryResults):
    @classmethod
    def setupClass(cls):
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog, prepend=False)
        res2 = Spector()
        res2.probit()
        cls.res2 = res2
        cls.res1 = Probit(data.endog, data.exog).fit(method="ncg",
            disp=0, avextol=1e-8)


class CheckLikelihoodModelL1(object):
    """
    For testing results generated with L1 regularization
    """
    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_4)

    def test_conf_int(self):
        assert_almost_equal(
                self.res1.conf_int(), self.res2.conf_int, DECIMAL_4)

    def test_bse(self):
        assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL_4)

    def test_nnz_params(self):
        assert_almost_equal(
                self.res1.nnz_params, self.res2.nnz_params, DECIMAL_4)

    def test_aic(self):
        assert_almost_equal(
                self.res1.aic, self.res2.aic, DECIMAL_3)

    def test_bic(self):
        assert_almost_equal(
                self.res1.bic, self.res2.bic, DECIMAL_3)


class TestProbitL1(CheckLikelihoodModelL1):
    @classmethod
    def setupClass(cls):
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog, prepend=True)
        alpha = np.array([0.1, 0.2, 0.3, 10]) #/ data.exog.shape[0]
        cls.res1 = Probit(data.endog, data.exog).fit_regularized(
            method="l1", alpha=alpha, disp=0, trim_mode='auto',
            auto_trim_tol=0.02, acc=1e-10, maxiter=1000)
        res2 = DiscreteL1()
        res2.probit()
        cls.res2 = res2

    def test_cov_params(self):
        assert_almost_equal(
                self.res1.cov_params(), self.res2.cov_params, DECIMAL_4)


class TestMNLogitL1(CheckLikelihoodModelL1):
    @classmethod
    def setupClass(cls):
        anes_data = sm.datasets.anes96.load()
        anes_exog = anes_data.exog
        anes_exog = sm.add_constant(anes_exog, prepend=False)
        mlogit_mod = sm.MNLogit(anes_data.endog, anes_exog)
        alpha = 10. * np.ones((mlogit_mod.J - 1, mlogit_mod.K)) #/ anes_exog.shape[0]
        alpha[-1,:] = 0
        cls.res1 = mlogit_mod.fit_regularized(
                method='l1', alpha=alpha, trim_mode='auto', auto_trim_tol=0.02,
                acc=1e-10, disp=0)
        res2 = DiscreteL1()
        res2.mnlogit()
        cls.res2 = res2


class TestLogitL1(CheckLikelihoodModelL1):
    @classmethod
    def setupClass(cls):
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog, prepend=True)
        cls.alpha = 3 * np.array([0., 1., 1., 1.]) #/ data.exog.shape[0]
        cls.res1 = Logit(data.endog, data.exog).fit_regularized(
            method="l1", alpha=cls.alpha, disp=0, trim_mode='size',
            size_trim_tol=1e-5, acc=1e-10, maxiter=1000)
        res2 = DiscreteL1()
        res2.logit()
        cls.res2 = res2

    def test_cov_params(self):
        assert_almost_equal(
                self.res1.cov_params(), self.res2.cov_params, DECIMAL_4)


class TestCVXOPT(object):
    @classmethod
    def setupClass(self):
        self.data = sm.datasets.spector.load()
        self.data.exog = sm.add_constant(self.data.exog, prepend=True)

    def test_cvxopt_versus_slsqp(self):
        #Compares resutls from cvxopt to the standard slsqp
        if has_cvxopt:
            self.alpha = 3. * np.array([0, 1, 1, 1.]) #/ self.data.endog.shape[0]
            res_slsqp = Logit(self.data.endog, self.data.exog).fit_regularized(
                method="l1", alpha=self.alpha, disp=0, acc=1e-10, maxiter=1000,
                trim_mode='auto')
            res_cvxopt = Logit(self.data.endog, self.data.exog).fit_regularized(
                method="l1_cvxopt_cp", alpha=self.alpha, disp=0, abstol=1e-10,
                trim_mode='auto', auto_trim_tol=0.01, maxiter=1000)
            assert_almost_equal(res_slsqp.params, res_cvxopt.params, DECIMAL_4)
        else:
            raise SkipTest("Skipped test_cvxopt since cvxopt is not available")


class TestSweepAlphaL1(object):
    @classmethod
    def setupClass(cls):
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog, prepend=True)
        cls.model = Logit(data.endog, data.exog)
        cls.alphas = np.array(
                   [[0.1, 0.1, 0.1, 0.1],
                    [0.4, 0.4, 0.5, 0.5],
                    [0.5, 0.5, 1, 1]]) #/ data.exog.shape[0]
        cls.res1 = DiscreteL1()
        cls.res1.sweep()

    def test_sweep_alpha(self):
        for i in range(3):
            alpha = self.alphas[i, :]
            res2 = self.model.fit_regularized(
                    method="l1", alpha=alpha, disp=0, acc=1e-10,
                    trim_mode='off', maxiter=1000)
            assert_almost_equal(res2.params, self.res1.params[i], DECIMAL_4)


class CheckL1Compatability(object):
    """
    Tests compatability between l1 and unregularized by setting alpha such
    that certain parameters should be effectively unregularized, and others
    should be ignored by the model.
    """
    def test_params(self):
        m = self.m
        assert_almost_equal(
            self.res_unreg.params, self.res_reg.params[:m], DECIMAL_4)
        # The last entry should be close to zero
        assert_almost_equal(0, self.res_reg.params[m:], DECIMAL_4)

    def test_cov_params(self):
        m = self.m
        # The restricted cov_params should be equal
        assert_almost_equal(
            self.res_unreg.cov_params(), self.res_reg.cov_params()[:m, :m],
            DECIMAL_1)

    def test_df(self):
        assert_equal(self.res_unreg.df_model, self.res_reg.df_model)
        assert_equal(self.res_unreg.df_resid, self.res_reg.df_resid)

    def test_t_test(self):
        m = self.m
        kvars = self.kvars
        t_unreg = self.res_unreg.t_test(np.eye(m))
        t_reg = self.res_reg.t_test(np.eye(kvars))
        assert_almost_equal(t_unreg.effect, t_reg.effect[:m], DECIMAL_3)
        assert_almost_equal(t_unreg.sd, t_reg.sd[:m], DECIMAL_3)
        assert_almost_equal(np.nan, t_reg.sd[m])
        assert_almost_equal(t_unreg.tvalue, t_reg.tvalue[:m], DECIMAL_3)
        assert_almost_equal(np.nan, t_reg.tvalue[m])

    def test_f_test(self):
        m = self.m
        kvars = self.kvars
        f_unreg = self.res_unreg.f_test(np.eye(m))
        f_reg = self.res_reg.f_test(np.eye(kvars)[:m])
        assert_almost_equal(f_unreg.fvalue, f_reg.fvalue, DECIMAL_2)
        assert_almost_equal(f_unreg.pvalue, f_reg.pvalue, DECIMAL_3)

    def test_bad_r_matrix(self):
        kvars = self.kvars
        assert_raises(ValueError, self.res_reg.f_test, np.eye(kvars) )


class TestPoissonL1Compatability(CheckL1Compatability):
    @classmethod
    def setupClass(cls):
        cls.kvars = 10 # Number of variables
        cls.m = 7 # Number of unregularized parameters
        rand_data = sm.datasets.randhie.load()
        rand_exog = rand_data.exog.view(float).reshape(len(rand_data.exog), -1)
        rand_exog = sm.add_constant(rand_exog, prepend=True)
        # Drop some columns and do an unregularized fit
        exog_no_PSI = rand_exog[:, :cls.m]
        cls.res_unreg = sm.Poisson(
            rand_data.endog, exog_no_PSI).fit(method="newton", disp=False)
        # Do a regularized fit with alpha, effectively dropping the last column
        alpha = 10 * len(rand_data.endog) * np.ones(cls.kvars)
        alpha[:cls.m] = 0
        cls.res_reg = sm.Poisson(rand_data.endog, rand_exog).fit_regularized(
            method='l1', alpha=alpha, disp=False, acc=1e-10, maxiter=2000,
            trim_mode='auto')


class TestLogitL1Compatability(CheckL1Compatability):
    @classmethod
    def setupClass(cls):
        cls.kvars = 4 # Number of variables
        cls.m = 3 # Number of unregularized parameters
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog, prepend=True)
        # Do a regularized fit with alpha, effectively dropping the last column
        alpha = np.array([0, 0, 0, 10])
        cls.res_reg = Logit(data.endog, data.exog).fit_regularized(
            method="l1", alpha=alpha, disp=0, acc=1e-15, maxiter=2000,
            trim_mode='auto')
        # Actually drop the last columnand do an unregularized fit
        exog_no_PSI = data.exog[:, :cls.m]
        cls.res_unreg = Logit(data.endog, exog_no_PSI).fit(disp=0, tol=1e-15)

class TestMNLogitL1Compatability(CheckL1Compatability):
    @classmethod
    def setupClass(cls):
        cls.kvars = 4 # Number of variables
        cls.m = 3 # Number of unregularized parameters
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog, prepend=True)
        alpha = np.array([0, 0, 0, 10])
        cls.res_reg = MNLogit(data.endog, data.exog).fit_regularized(
            method="l1", alpha=alpha, disp=0, acc=1e-15, maxiter=2000,
            trim_mode='auto')
        # Actually drop the last columnand do an unregularized fit
        exog_no_PSI = data.exog[:, :cls.m]
        cls.res_unreg = MNLogit(data.endog, exog_no_PSI).fit(
            disp=0, tol=1e-15)
#
    def test_t_test(self):
        m = self.m
        kvars = self.kvars
        t_unreg = self.res_unreg.t_test(np.eye(m))
        t_reg = self.res_reg.t_test(np.eye(kvars))
        assert_almost_equal(t_unreg.effect, t_reg.effect[:m], DECIMAL_3)
        assert_almost_equal(t_unreg.sd, t_reg.sd[:m], DECIMAL_3)
        assert_almost_equal(np.nan, t_reg.sd[m])
        assert_almost_equal(t_unreg.tvalue, t_reg.tvalue[:m, :m], DECIMAL_3)

    def test_f_test(self):
        raise SkipTest("Skipped test_f_test for MNLogit")


class TestProbitL1Compatability(CheckL1Compatability):
    @classmethod
    def setupClass(cls):
        cls.kvars = 4 # Number of variables
        cls.m = 3 # Number of unregularized parameters
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog, prepend=True)
        alpha = np.array([0, 0, 0, 10])
        cls.res_reg = Probit(data.endog, data.exog).fit_regularized(
            method="l1", alpha=alpha, disp=0, acc=1e-15, maxiter=2000,
            trim_mode='auto')
        # Actually drop the last columnand do an unregularized fit
        exog_no_PSI = data.exog[:, :cls.m]
        cls.res_unreg = Probit(data.endog, exog_no_PSI).fit(disp=0, tol=1e-15)


class CompareL1(object):
    """
    For checking results for l1 regularization.
    Assumes self.res1 and self.res2 are two legitimate models to be compared.
    """
    def test_basic_results(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_4)
        assert_almost_equal(self.res1.cov_params(), self.res2.cov_params(), DECIMAL_4)
        assert_almost_equal(self.res1.conf_int(), self.res2.conf_int(), DECIMAL_4)
        assert_almost_equal(self.res1.pvalues, self.res2.pvalues, DECIMAL_4)
        assert_almost_equal(self.res1.pred_table(), self.res2.pred_table(), DECIMAL_4)
        assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL_4)
        assert_almost_equal(self.res1.llf, self.res2.llf, DECIMAL_4)
        assert_almost_equal(self.res1.aic, self.res2.aic, DECIMAL_4)
        assert_almost_equal(self.res1.bic, self.res2.bic, DECIMAL_4)
        assert_almost_equal(self.res1.pvalues, self.res2.pvalues, DECIMAL_4)


class CompareL11D(CompareL1):
    """
    Check t and f tests.  This only works for 1-d results
    """
    def test_tests(self):
        restrictmat = np.eye(len(self.res1.params.ravel()))
        assert_almost_equal(self.res1.t_test(restrictmat).pvalue,
                            self.res2.t_test(restrictmat).pvalue, DECIMAL_4)
        assert_almost_equal(self.res1.f_test(restrictmat).pvalue,
                            self.res2.f_test(restrictmat).pvalue, DECIMAL_4)


class TestL1AlphaZeroLogit(CompareL11D):
    """
    Compares l1 model with alpha = 0 to the unregularized model.
    """
    @classmethod
    def setupClass(cls):
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog, prepend=True)
        cls.res1 = Logit(data.endog, data.exog).fit_regularized(
                method="l1", alpha=0, disp=0, acc=1e-15, maxiter=1000,
                trim_mode='auto', auto_trim_tol=0.01)
        cls.res2 = Logit(data.endog, data.exog).fit(disp=0, tol=1e-15)


class TestL1AlphaZeroProbit(CompareL11D):
    """
    Compares l1 model with alpha = 0 to the unregularized model.
    """
    @classmethod
    def setupClass(cls):
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog, prepend=True)
        cls.res1 = Probit(data.endog, data.exog).fit_regularized(
                method="l1", alpha=0, disp=0, acc=1e-15, maxiter=1000,
                trim_mode='auto', auto_trim_tol=0.01)
        cls.res2 = Probit(data.endog, data.exog).fit(disp=0, tol=1e-15)


class TestL1AlphaZeroMNLogit(CompareL1):
    @classmethod
    def setupClass(cls):
        data = sm.datasets.anes96.load()
        data.exog = sm.add_constant(data.exog, prepend=False)
        cls.res1 = MNLogit(data.endog, data.exog).fit_regularized(
                method="l1", alpha=0, disp=0, acc=1e-15, maxiter=1000,
                trim_mode='auto', auto_trim_tol=0.01)
        cls.res2 = MNLogit(data.endog, data.exog).fit(disp=0, tol=1e-15)


class TestLogitNewton(CheckBinaryResults, CheckMargEff):
    @classmethod
    def setupClass(cls):
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog, prepend=False)
        cls.res1 = Logit(data.endog, data.exog).fit(method="newton", disp=0)
        res2 = Spector()
        res2.logit()
        cls.res2 = res2

    def test_nodummy_exog1(self):
        me = self.res1.get_margeff(atexog={0 : 2.0, 2 : 1.})
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_atexog1, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_atexog1_se, DECIMAL_4)

    def test_nodummy_exog2(self):
        me = self.res1.get_margeff(atexog={1 : 21., 2 : 0}, at='mean')
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_atexog2, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_atexog2_se, DECIMAL_4)

    def test_dummy_exog1(self):
        me = self.res1.get_margeff(atexog={0 : 2.0, 2 : 1.}, dummy=True)
        assert_almost_equal(me.margeff,
                self.res2.margeff_dummy_atexog1, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_dummy_atexog1_se, DECIMAL_4)

    def test_dummy_exog2(self):
        me = self.res1.get_margeff(atexog={1 : 21., 2 : 0}, at='mean',
                dummy=True)
        assert_almost_equal(me.margeff,
                self.res2.margeff_dummy_atexog2, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_dummy_atexog2_se, DECIMAL_4)

class TestLogitBFGS(CheckBinaryResults, CheckMargEff):
    @classmethod
    def setupClass(cls):
#        import scipy
#        major, minor, micro = scipy.__version__.split('.')[:3]
#        if int(minor) < 9:
#            raise SkipTest
        #Skip this unconditionally for release 0.3.0
        #since there are still problems with scipy 0.9.0 on some machines
        #Ralf on mailing list 2011-03-26
        raise SkipTest

        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog, prepend=False)
        res2 = Spector()
        res2.logit()
        cls.res2 = res2
        cls.res1 = Logit(data.endog, data.exog).fit(method="bfgs",
            disp=0)

class TestPoissonNewton(CheckModelResults):
    @classmethod
    def setupClass(cls):
        from results.results_discrete import RandHIE
        data = sm.datasets.randhie.load()
        exog = sm.add_constant(data.exog, prepend=False)
        cls.res1 = Poisson(data.endog, exog).fit(method='newton', disp=0)
        res2 = RandHIE()
        res2.poisson()
        cls.res2 = res2

    def test_margeff_overall(self):
        me = self.res1.get_margeff()
        assert_almost_equal(me.margeff,
                self.res2.margeff_nodummy_overall, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_nodummy_overall_se, DECIMAL_4)

    def test_margeff_dummy_overall(self):
        me = self.res1.get_margeff(dummy=True)
        assert_almost_equal(me.margeff,
                self.res2.margeff_dummy_overall, DECIMAL_4)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_dummy_overall_se, DECIMAL_4)

class TestMNLogitNewtonBaseZero(CheckModelResults):
    @classmethod
    def setupClass(cls):
        from results.results_discrete import Anes
        data = sm.datasets.anes96.load()
        cls.data = data
        exog = data.exog
        exog = sm.add_constant(exog, prepend=False)
        cls.res1 = MNLogit(data.endog, exog).fit(method="newton", disp=0)
        res2 = Anes()
        res2.mnlogit_basezero()
        cls.res2 = res2

    def test_margeff_overall(self):
        me = self.res1.get_margeff()
        assert_almost_equal(me.margeff, self.res2.margeff_dydx_overall, 6)
        assert_almost_equal(me.margeff_se, self.res2.margeff_dydx_overall_se, 6)

    def test_margeff_mean(self):
        me = self.res1.get_margeff(at='mean')
        assert_almost_equal(me.margeff, self.res2.margeff_dydx_mean, 7)
        assert_almost_equal(me.margeff_se, self.res2.margeff_dydx_mean_se, 7)

    def test_margeff_dummy(self):
        data = self.data
        vote = data.data['vote']
        exog = np.column_stack((data.exog, vote))
        exog = sm.add_constant(exog, prepend=False)
        res = MNLogit(data.endog, exog).fit(method="newton", disp=0)
        me = res.get_margeff(dummy=True)
        assert_almost_equal(me.margeff, self.res2.margeff_dydx_dummy_overall,
                6)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_dydx_dummy_overall_se, 6)
        me = res.get_margeff(dummy=True, method="eydx")
        assert_almost_equal(me.margeff, self.res2.margeff_eydx_dummy_overall,
                5)
        assert_almost_equal(me.margeff_se,
                self.res2.margeff_eydx_dummy_overall_se, 6)

    def test_j(self):
        assert_equal(self.res1.model.J, self.res2.J)

    def test_k(self):
        assert_equal(self.res1.model.K, self.res2.K)

    def test_endog_names(self):
        assert_equal(self.res1._get_endog_name(None,None)[1],
                     ['y=1', 'y=2', 'y=3', 'y=4', 'y=5', 'y=6'])

    def test_pred_table(self):
        # fitted results taken from gretl
        pred = [6, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 6, 0, 1, 6, 0, 0,
                1, 1, 6, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 6, 0, 0, 6, 6, 0, 0, 1,
                1, 6, 1, 6, 0, 0, 0, 1, 0, 1, 0, 0, 0, 6, 0, 0, 6, 0, 0, 0, 1,
                1, 0, 0, 6, 6, 6, 6, 1, 0, 5, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,
                6, 0, 6, 6, 1, 0, 1, 1, 6, 5, 1, 0, 0, 0, 5, 0, 0, 6, 0, 1, 0,
                0, 0, 0, 0, 1, 1, 0, 6, 6, 6, 6, 5, 0, 1, 1, 0, 1, 0, 6, 6, 0,
                0, 0, 6, 0, 0, 0, 6, 6, 0, 5, 1, 0, 0, 0, 0, 6, 0, 5, 6, 6, 0,
                0, 0, 0, 6, 1, 0, 0, 1, 0, 1, 6, 1, 1, 1, 1, 1, 0, 0, 0, 6, 0,
                5, 1, 0, 6, 6, 6, 0, 0, 0, 0, 1, 6, 6, 0, 0, 0, 1, 1, 5, 6, 0,
                6, 1, 0, 0, 1, 6, 0, 0, 1, 0, 6, 6, 0, 5, 6, 6, 0, 0, 6, 1, 0,
                6, 0, 1, 0, 1, 6, 0, 1, 1, 1, 6, 0, 5, 0, 0, 6, 1, 0, 6, 5, 5,
                0, 6, 1, 1, 1, 0, 0, 6, 0, 0, 5, 0, 0, 6, 6, 6, 6, 6, 0, 1, 0,
                0, 6, 6, 0, 0, 1, 6, 0, 0, 6, 1, 6, 1, 1, 1, 0, 1, 6, 5, 0, 0,
                1, 5, 0, 1, 6, 6, 1, 0, 0, 1, 6, 1, 5, 6, 1, 0, 0, 1, 1, 0, 6,
                1, 6, 0, 1, 1, 5, 6, 6, 5, 1, 1, 1, 0, 6, 1, 6, 1, 0, 1, 0, 0,
                1, 5, 0, 1, 1, 0, 5, 6, 0, 5, 1, 1, 6, 5, 0, 6, 0, 0, 0, 0, 0,
                0, 1, 6, 1, 0, 5, 1, 0, 0, 1, 6, 0, 0, 6, 6, 6, 0, 2, 1, 6, 5,
                6, 1, 1, 0, 5, 1, 1, 1, 6, 1, 6, 6, 5, 6, 0, 1, 0, 1, 6, 0, 6,
                1, 6, 0, 0, 6, 1, 0, 6, 1, 0, 0, 0, 0, 6, 6, 6, 6, 5, 6, 6, 0,
                0, 6, 1, 1, 6, 0, 0, 6, 6, 0, 6, 6, 0, 0, 6, 0, 0, 6, 6, 6, 1,
                0, 6, 0, 0, 0, 6, 1, 1, 0, 1, 5, 0, 0, 5, 0, 0, 0, 1, 1, 6, 1,
                0, 0, 0, 6, 6, 1, 1, 6, 5, 5, 0, 6, 6, 0, 1, 1, 0, 6, 6, 0, 6,
                5, 5, 6, 5, 1, 0, 6, 0, 6, 1, 0, 1, 6, 6, 6, 1, 0, 6, 0, 5, 6,
                6, 5, 0, 5, 1, 0, 6, 0, 6, 1, 5, 5, 0, 1, 5, 5, 2, 6, 6, 6, 5,
                0, 0, 1, 6, 1, 0, 1, 6, 1, 0, 0, 1, 5, 6, 6, 0, 0, 0, 5, 6, 6,
                6, 1, 5, 6, 1, 0, 0, 6, 5, 0, 1, 1, 1, 6, 6, 0, 1, 0, 0, 0, 5,
                0, 0, 6, 1, 6, 0, 6, 1, 5, 5, 6, 5, 0, 0, 0, 0, 1, 1, 0, 5, 5,
                0, 0, 0, 0, 1, 0, 6, 6, 1, 1, 6, 6, 0, 5, 5, 0, 0, 0, 6, 6, 1,
                6, 0, 0, 5, 0, 1, 6, 5, 6, 6, 5, 5, 6, 6, 1, 0, 1, 6, 6, 1, 6,
                0, 6, 0, 6, 5, 0, 6, 6, 0, 5, 6, 0, 6, 6, 5, 0, 1, 6, 6, 1, 0,
                1, 0, 6, 6, 1, 0, 6, 6, 6, 0, 1, 6, 0, 1, 5, 1, 1, 5, 6, 6, 0,
                1, 6, 6, 1, 5, 0, 5, 0, 6, 0, 1, 6, 1, 0, 6, 1, 6, 0, 6, 1, 0,
                0, 0, 6, 6, 0, 1, 1, 6, 6, 6, 1, 6, 0, 5, 6, 0, 5, 6, 6, 5, 5,
                5, 6, 0, 6, 0, 0, 0, 5, 0, 6, 1, 2, 6, 6, 6, 5, 1, 6, 0, 6, 0,
                0, 0, 0, 6, 5, 0, 5, 1, 6, 5, 1, 6, 5, 1, 1, 0, 0, 6, 1, 1, 5,
                6, 6, 0, 5, 2, 5, 5, 0, 5, 5, 5, 6, 5, 6, 6, 5, 2, 6, 5, 6, 0,
                0, 6, 5, 0, 6, 0, 0, 6, 6, 6, 0, 5, 1, 1, 6, 6, 5, 2, 1, 6, 5,
                6, 0, 6, 6, 1, 1, 5, 1, 6, 6, 6, 0, 0, 6, 1, 0, 5, 5, 1, 5, 6,
                1, 6, 0, 1, 6, 5, 0, 0, 6, 1, 5, 1, 0, 6, 0, 6, 6, 5, 5, 6, 6,
                6, 6, 2, 6, 6, 6, 5, 5, 5, 0, 1, 0, 0, 0, 6, 6, 1, 0, 6, 6, 6,
                6, 6, 1, 0, 6, 1, 5, 5, 6, 6, 6, 6, 6, 5, 6, 1, 6, 2, 5, 5, 6,
                5, 6, 6, 5, 6, 6, 5, 5, 6, 1, 5, 1, 6, 0, 2, 5, 0, 5, 0, 2, 1,
                6, 0, 0, 6, 6, 1, 6, 0, 5, 5, 6, 6, 1, 6, 6, 6, 5, 6, 6, 1, 6,
                5, 6, 1, 1, 0, 6, 6, 5, 1, 0, 0, 6, 6, 5, 6, 0, 1, 6, 0, 5, 6,
                5, 2, 5, 2, 0, 0, 1, 6, 6, 1, 5, 6, 6, 0, 6, 6, 6, 6, 6, 5]
        assert_array_equal(self.res1.predict().argmax(1), pred)

        # the rows should add up for pred table
        assert_array_equal(self.res1.pred_table().sum(0), np.bincount(pred))

        # note this is just a regression test, gretl doesn't have a prediction
        # table
        pred = [[ 126.,   41.,    2.,    0.,    0.,   12.,   19.],
                [  77.,   73.,    3.,    0.,    0.,   15.,   12.],
                [  37.,   43.,    2.,    0.,    0.,   19.,    7.],
                [  12.,    9.,    1.,    0.,    0.,    9.,    6.],
                [  19.,   10.,    2.,    0.,    0.,   20.,   43.],
                [  22.,   25.,    1.,    0.,    0.,   31.,   71.],
                [   9.,    7.,    1.,    0.,    0.,   18.,  140.]]
        assert_array_equal(self.res1.pred_table(), pred)

def test_perfect_prediction():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    iris_dir = os.path.join(cur_dir, '..', '..', 'genmod', 'tests', 'results')
    iris_dir = os.path.abspath(iris_dir)
    iris = np.genfromtxt(os.path.join(iris_dir, 'iris.csv'), delimiter=",",
                            skip_header=1)
    y = iris[:,-1]
    X = iris[:,:-1]
    X = X[y != 2]
    y = y[y != 2]
    X = sm.add_constant(X, prepend=True)
    mod = Logit(y,X)
    assert_raises(PerfectSeparationError, mod.fit)
    #turn off raise PerfectSeparationError
    mod.raise_on_perfect_prediction = False
    mod.fit(disp=False)  #should not raise

def test_poisson_predict():
    #GH: 175, make sure poisson predict works without offset and exposure
    data = sm.datasets.randhie.load()
    exog = sm.add_constant(data.exog, prepend=True)
    res = sm.Poisson(data.endog, exog).fit(method='newton', disp=0)
    pred1 = res.predict()
    pred2 = res.predict(exog)
    assert_almost_equal(pred1, pred2)
    #exta options
    pred3 = res.predict(exog, offset=0, exposure=1)
    assert_almost_equal(pred1, pred3)
    pred3 = res.predict(exog, offset=0, exposure=2)
    assert_almost_equal(2*pred1, pred3)
    pred3 = res.predict(exog, offset=np.log(2), exposure=1)
    assert_almost_equal(2*pred1, pred3)

def test_poisson_newton():
    #GH: 24, Newton doesn't work well sometimes
    nobs = 10000
    np.random.seed(987689)
    x = np.random.randn(nobs, 3)
    x = sm.add_constant(x, prepend=True)
    y_count = np.random.poisson(np.exp(x.sum(1)))
    mod = sm.Poisson(y_count, x)
    res = mod.fit(start_params=-np.ones(4), method='newton', disp=0)
    assert_(not res.mle_retvals['converged'])

def test_issue_339():
    # make sure MNLogit summary works for J != K.
    data = sm.datasets.anes96.load()
    exog = data.exog
    # leave out last exog column
    exog = exog[:,:-1]
    exog = sm.add_constant(exog, prepend=True)
    res1 = sm.MNLogit(data.endog, exog).fit(method="newton", disp=0)
    # strip the header from the test
    smry = "\n".join(res1.summary().as_text().split('\n')[9:])
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    test_case_file = os.path.join(cur_dir, 'results', 'mn_logit_summary.txt')
    test_case = open(test_case_file, 'r').read()
    np.testing.assert_(smry == test_case[:-1])

def test_issue_341():
    data = sm.datasets.anes96.load()
    exog = data.exog
    # leave out last exog column
    exog = exog[:,:-1]
    exog = sm.add_constant(exog, prepend=True)
    res1 = sm.MNLogit(data.endog, exog).fit(method="newton", disp=0)
    x = exog[0]
    np.testing.assert_equal(res1.predict(x).shape, (1,7))
    np.testing.assert_equal(res1.predict(x[None]).shape, (1,7))

def test_iscount():
    X = np.random.random((50, 10))
    X[:,2] = np.random.randint(1, 10, size=50)
    X[:,6] = np.random.randint(1, 10, size=50)
    X[:,4] = np.random.randint(0, 2, size=50)
    X[:,1] = np.random.randint(-10, 10, size=50) # not integers
    count_ind = _iscount(X)
    assert_equal(count_ind, [2, 6])

def test_isdummy():
    X = np.random.random((50, 10))
    X[:,2] = np.random.randint(1, 10, size=50)
    X[:,6] = np.random.randint(0, 2, size=50)
    X[:,4] = np.random.randint(0, 2, size=50)
    X[:,1] = np.random.randint(-10, 10, size=50) # not integers
    count_ind = _isdummy(X)
    assert_equal(count_ind, [4, 6])


if __name__ == "__main__":
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb'],
            exit=False)
