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
from scikits.statsmodels.discrete.discrete_model import *
import scikits.statsmodels.api as sm
from sys import platform
from nose import SkipTest
from results.results_discrete import Spector
from scikits.statsmodels.tools.sm_exceptions import PerfectSeparationError

DECIMAL_14 = 14
DECIMAL_10 = 10
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
        assert_almost_equal(jacsum, score, DECIMAL_10) #Poisson has low precision ?


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

class TestProbitNewton(CheckModelResults):

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


class TestProbitBFGS(CheckModelResults):

    @classmethod
    def setupClass(cls):
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog)
        cls.res1 = Probit(data.endog, data.exog).fit(method="bfgs",
            disp=0)
        res2 = Spector()
        res2.probit()
        cls.res2 = res2


class TestProbitNM(CheckModelResults):
    @classmethod
    def setupClass(cls):
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog)
        res2 = Spector()
        res2.probit()
        cls.res2 = res2
        cls.res1 = Probit(data.endog, data.exog).fit(method="nm",
            disp=0, maxiter=500)

class TestProbitPowell(CheckModelResults):
    @classmethod
    def setupClass(cls):
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog)
        res2 = Spector()
        res2.probit()
        cls.res2 = res2
        cls.res1 = Probit(data.endog, data.exog).fit(method="powell",
            disp=0, ftol=1e-8)

class TestProbitCG(CheckModelResults):
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

class TestProbitNCG(CheckModelResults):
    @classmethod
    def setupClass(cls):
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog)
        res2 = Spector()
        res2.probit()
        cls.res2 = res2
        cls.res1 = Probit(data.endog, data.exog).fit(method="ncg",
            disp=0, avextol=1e-8)

class TestLogitNewton(CheckModelResults, CheckMargEff):
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

class TestLogitBFGS(CheckModelResults, CheckMargEff):
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



if __name__ == "__main__":
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb'],
            exit=False)
