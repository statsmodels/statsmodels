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
from statsmodels.discrete.discrete_model import *
import statsmodels.api as sm
from sys import platform
from nose import SkipTest
from results.results_discrete import Spector, DiscreteL1
from statsmodels.tools.sm_exceptions import PerfectSeparationError
import pdb  # pdb.set_trace
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

    def test_margeff(self):
        pass
    # this probably needs it's own test class?

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
        assert_almost_equal(self.res1.margeff(),
                self.res2.margeff_nodummy_dydx, DECIMAL_4)

    def test_nodummy_dydxmean(self):
        assert_almost_equal(self.res1.margeff(at='mean'),
                self.res2.margeff_nodummy_dydxmean, DECIMAL_4)

    def test_nodummy_dydxmedian(self):
        assert_almost_equal(self.res1.margeff(at='median'),
                self.res2.margeff_nodummy_dydxmedian, DECIMAL_4)

    def test_nodummy_dydxzero(self):
        assert_almost_equal(self.res1.margeff(at='zero'),
                self.res2.margeff_nodummy_dydxzero, DECIMAL_4)

    def test_nodummy_dyexoverall(self):
        assert_almost_equal(self.res1.margeff(method='dyex'),
                self.res2.margeff_nodummy_dyex, DECIMAL_4)

    def test_nodummy_dyexmean(self):
        assert_almost_equal(self.res1.margeff(at='mean', method='dyex'),
                self.res2.margeff_nodummy_dyexmean, DECIMAL_4)

    def test_nodummy_dyexmedian(self):
        assert_almost_equal(self.res1.margeff(at='median', method='dyex'),
                self.res2.margeff_nodummy_dyexmedian, DECIMAL_4)

    def test_nodummy_dyexzero(self):
        assert_almost_equal(self.res1.margeff(at='zero', method='dyex'),
                self.res2.margeff_nodummy_dyexzero, DECIMAL_4)

    def test_nodummy_eydxoverall(self):
        assert_almost_equal(self.res1.margeff(method='eydx'),
                self.res2.margeff_nodummy_eydx, DECIMAL_4)

    def test_nodummy_eydxmean(self):
        assert_almost_equal(self.res1.margeff(at='mean', method='eydx'),
                self.res2.margeff_nodummy_eydxmean, DECIMAL_4)

    def test_nodummy_eydxmedian(self):
        assert_almost_equal(self.res1.margeff(at='median', method='eydx'),
                self.res2.margeff_nodummy_eydxmedian, DECIMAL_4)

    def test_nodummy_eydxzero(self):
        assert_almost_equal(self.res1.margeff(at='zero', method='eydx'),
                self.res2.margeff_nodummy_eydxzero, DECIMAL_4)

    def test_nodummy_eyexoverall(self):
        assert_almost_equal(self.res1.margeff(method='eyex'),
                self.res2.margeff_nodummy_eyex, DECIMAL_4)

    def test_nodummy_eyexmean(self):
        assert_almost_equal(self.res1.margeff(at='mean', method='eyex'),
                self.res2.margeff_nodummy_eyexmean, DECIMAL_4)

    def test_nodummy_eyexmedian(self):
        assert_almost_equal(self.res1.margeff(at='median', method='eyex'),
                self.res2.margeff_nodummy_eyexmedian, DECIMAL_4)

    def test_nodummy_eyexzero(self):
        assert_almost_equal(self.res1.margeff(at='zero', method='eyex'),
                self.res2.margeff_nodummy_eyexzero, DECIMAL_4)

    def test_dummy_dydxoverall(self):
        assert_almost_equal(self.res1.margeff(dummy=True),
                self.res2.margeff_dummy_dydx, DECIMAL_4)

    def test_dummy_dydxmean(self):
        assert_almost_equal(self.res1.margeff(at='mean', dummy=True),
                self.res2.margeff_dummy_dydxmean, DECIMAL_4)

    def test_dummy_eydxoverall(self):
        assert_almost_equal(self.res1.margeff(method='eydx', dummy=True),
                self.res2.margeff_dummy_eydx, DECIMAL_4)

    def test_dummy_eydxmean(self):
        assert_almost_equal(self.res1.margeff(at='mean', method='eydx',
            dummy=True), self.res2.margeff_dummy_eydxmean, DECIMAL_4)

class TestProbitNewton(CheckBinaryResults):

    @classmethod
    def setupClass(cls):
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog)
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
        data.exog = sm.add_constant(data.exog)
        cls.res1 = Probit(data.endog, data.exog).fit(method="bfgs",
            disp=0)
        res2 = Spector()
        res2.probit()
        cls.res2 = res2


class TestProbitNM(CheckBinaryResults):
    @classmethod
    def setupClass(cls):
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog)
        res2 = Spector()
        res2.probit()
        cls.res2 = res2
        cls.res1 = Probit(data.endog, data.exog).fit(method="nm",
            disp=0, maxiter=500)

class TestProbitPowell(CheckBinaryResults):
    @classmethod
    def setupClass(cls):
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog)
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
        data.exog = sm.add_constant(data.exog)
        res2 = Spector()
        res2.probit()
        cls.res2 = res2
        cls.res1 = Probit(data.endog, data.exog).fit(method="cg",
            disp=0, maxiter=500)

class TestProbitNCG(CheckBinaryResults):
    @classmethod
    def setupClass(cls):
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog)
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


class TestsCVXOPT(object):
    @classmethod
    def setupClass(self):
        self.data = sm.datasets.spector.load()
        self.data.exog = sm.add_constant(self.data.exog, prepend=True)
        self.alpha = 3 * np.array([0, 1, 1, 1])
        self.res1 = Logit(self.data.endog, self.data.exog).fit(
            method="l1", alpha=self.alpha, disp=0, acc=1e-10)

    def test_cvxopt(self):
        """
        Compares resutls from csxopt to the standard slsqp
        """
        if has_cvxopt:
            res3 = Logit(self.data.endog, self.data.exog).fit(
                method="l1_cvxopt_cp", alpha=self.alpha, disp=0, abstol=1e-10)
            assert_almost_equal(self.res1.params, res3.params, DECIMAL_4)
        else:
            raise SkipTest("Skipped test_cvxopt since cvxopt is not available")


class TestLogitL1(CheckLikelihoodModelL1):
    @classmethod
    def setupClass(cls):
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog, prepend=True)
        cls.alpha = 3 * np.array([0, 1, 1, 1])
        cls.res1 = Logit(data.endog, data.exog).fit(
            method="l1", alpha=cls.alpha, disp=0, trim_params=True, 
            trim_tol=1e-5, acc=1e-10)
        res2 = DiscreteL1()
        res2.logit()
        cls.res2 = res2


class TestLogitL1AlphaZero(object):
    @classmethod
    def setupClass(cls):
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog, prepend=True)
        cls.res1 = Logit(data.endog, data.exog).fit(
                method="l1", alpha=0, disp=0, acc=1e-15)
        cls.res2 = Logit(data.endog, data.exog).fit(disp=0, tol=1e-15)

    def test_logit_l1_alpha_zero(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_4)


class TestLogitNewton(CheckBinaryResults, CheckMargEff):
    @classmethod
    def setupClass(cls):
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog)
        cls.res1 = Logit(data.endog, data.exog).fit(method="newton", disp=0)
        res2 = Spector()
        res2.logit()
        cls.res2 = res2

    def test_nodummy_exog1(self):
        assert_almost_equal(self.res1.margeff(atexog={0 : 2.0, 2 : 1.}),
                self.res2.margeff_nodummy_atexog1, DECIMAL_4)

    def test_nodummy_exog2(self):
        assert_almost_equal(self.res1.margeff(atexog={1 : 21., 2 : 0}, at='mean'),
                self.res2.margeff_nodummy_atexog2, DECIMAL_4)

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
        data.exog = sm.add_constant(data.exog)
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
        exog = sm.add_constant(data.exog)
        cls.res1 = Poisson(data.endog, exog).fit(method='newton', disp=0)
        res2 = RandHIE()
        res2.poisson()
        cls.res2 = res2

class TestMNLogitNewtonBaseZero(CheckModelResults):
    @classmethod
    def setupClass(cls):
        from results.results_discrete import Anes
        data = sm.datasets.anes96.load()
        exog = data.exog
        exog[:,0] = np.log(exog[:,0] + .1)
        exog = np.column_stack((exog[:,0],exog[:,2],
            exog[:,5:8]))
        exog = sm.add_constant(exog)
        cls.res1 = MNLogit(data.endog, exog).fit(method="newton", disp=0)
        res2 = Anes()
        res2.mnlogit_basezero()
        cls.res2 = res2

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
    mod.fit()  #should not raise

def test_poisson_predict():
    #GH: 175, make sure poisson predict works without offset and exposure
    data = sm.datasets.randhie.load()
    exog = sm.add_constant(data.exog)
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
    exog[:,0] = np.log(exog[:,0] + .1)
    # leave out last exog column
    exog = np.column_stack((exog[:,0],exog[:,2],
        exog[:,5:7]))
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
    exog[:,0] = np.log(exog[:,0] + .1)
    # leave out last exog column
    exog = np.column_stack((exog[:,0],exog[:,2],
        exog[:,5:7]))
    exog = sm.add_constant(exog, prepend=True)
    res1 = sm.MNLogit(data.endog, exog).fit(method="newton", disp=0)
    x = exog[0]
    np.testing.assert_equal(res1.predict(x).shape, (1,7))
    np.testing.assert_equal(res1.predict(x[None]).shape, (1,7))


if __name__ == "__main__":
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb'],
            exit=False)
