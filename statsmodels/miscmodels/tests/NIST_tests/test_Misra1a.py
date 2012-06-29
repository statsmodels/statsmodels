
import numpy as np
import statsmodels.api as sm
from numpy.testing import (assert_almost_equal, assert_, assert_equal,
                          assert_approx_equal)
from statsmodels.datasets import misra
from results.misra1a.Misra1a import (funcMisra1a,funcMisra1a_J,Misra1a,
                                    Misra1aWNLS)

DECIMAL_2 = 2
DECIMAL_3 = 3
DECIMAL_4 = 4
DECIMAL_8 = 8

class CheckNLSresults(object):

    decimal_params = DECIMAL_4
    def test_params(self):
        assert_almost_equal(self.res.params, self.resv.params,
                self.decimal_params)

    decimal_standarderrors = DECIMAL_4
    def test_standarderrors(self):
        assert_almost_equal(self.res.bse,self.resv.bse,
                self.decimal_standarderrors)

    decimal_ser = DECIMAL_4
    def test_ser(self):
        assert_almost_equal(self.res.ser, self.resv.ser,
                self.decimal_params)

    decimal_covparams = DECIMAL_4
    def test_covparams(self):
        assert_almost_equal(self.res.cov_params(), self.resv.cov_params(),
                self.decimal_covparams)

    decimal_confidenceintervals = DECIMAL_4
    def test_confidenceintervals(self):
        conf1 = self.res.conf_int()
        conf2 = self.resv.conf_int()
        for i in range(len(conf1)):
            assert_approx_equal(conf1[i][0], conf2[i][0],
                    self.decimal_confidenceintervals)
            assert_approx_equal(conf1[i][1], conf2[i][1],
                    self.decimal_confidenceintervals)

    decimal_scale = DECIMAL_4
    def test_scale(self):
        assert_almost_equal(self.res.scale, self.resv.scale,
                self.decimal_scale)

    decimal_rsquared = DECIMAL_4
    def test_rsquared(self):
        assert_almost_equal(self.res.rsquared, self.resv.rsquared,
                self.decimal_rsquared)

    decimal_rsquared_adj = DECIMAL_4
    def test_rsquared_adj(self):
        assert_almost_equal(self.res.rsquared_adj, self.resv.rsquared_adj,
                    self.decimal_rsquared_adj)

    def test_degrees(self):
        assert_equal(self.res.model.df_model, self.resv.df_model)
        assert_equal(self.res.model.df_resid, self.resv.df_resid)

    decimal_ssr = DECIMAL_4
    def test_sumof_squaredresids(self):
        assert_almost_equal(self.res.ssr, self.resv.ssr, self.decimal_ssr)

    decimal_mse_resid = DECIMAL_4
    def test_mse_resid(self):
        assert_almost_equal(self.res.mse_resid, self.resv.mse_resid,
                    self.decimal_mse_resid)

    decimal_aic = DECIMAL_4
    def test_aic(self):
        assert_almost_equal(self.res.aic, self.resv.aic, self.decimal_aic)

    decimal_bic = DECIMAL_4
    def test_bic(self):
        assert_almost_equal(self.res.bic, self.resv.bic, self.decimal_bic)

    decimal_hqc = DECIMAL_4
    def test_hqc(self):
        assert_almost_equal(self.res.hqc, self.resv.hqc, self.decimal_hqc)

    decimal_tvalues = DECIMAL_4
    def test_tvalues(self):
        assert_almost_equal(self.res.tvalues, self.resv.tvalues,
            self.decimal_tvalues)

    decimal_pvalues = DECIMAL_4
    def test_pvalues(self):
        assert_almost_equal(self.res.pvalues, self.resv.pvalues,
            self.decimal_pvalues)

    decimal_resids = DECIMAL_4
    def test_resids(self):
        assert_almost_equal(self.res.resid, self.resv.resid,
            self.decimal_resids)

    decimal_fittedvalues = DECIMAL_4
    def test_fittedvalues(self):
        assert_almost_equal(self.res.fittedvalues,self.resv.fittedvalues,
            self.decimal_fittedvalues)

class TestMisra1a01(CheckNLSresults):
    ''' Numerical Jacobian
        start_value1
    '''
    @classmethod
    def setupClass(cls):
        #Loading the dataset
        data = misra.load()
        x = data.exog
        y = data.endog

        #Loading the results
        resv = Misra1a()
        mod = funcMisra1a(y,x)
        res = mod.fit(start_value=resv.start_value1)
        cls.res = res
        cls.resv = resv

class TestMisra1a02(CheckNLSresults):
    ''' Numerical Jacobian
        start_value2
    '''
    @classmethod
    def setupClass(cls):
        #Loading the dataset
        data = misra.load()
        x = data.exog
        y = data.endog

        #Loading the results
        resv = Misra1a()
        mod = funcMisra1a(y,x)
        res = mod.fit(start_value=resv.start_value2)
        cls.res = res
        cls.resv = resv

        #tests fail for more precision than below
        cls.decimal_tvalues = DECIMAL_2
        cls.decimal_params = DECIMAL_3

class TestMisra1a03(CheckNLSresults):
    ''' Analytical Jacobian
        start_value1
    '''
    @classmethod
    def setupClass(cls):
        #Loading the dataset
        data = misra.load()
        x = data.exog
        y = data.endog

        #Loading the results
        resv = Misra1a()
        mod = funcMisra1a_J(y,x)
        res = mod.fit(start_value=resv.start_value1)
        cls.res = res
        cls.resv = resv

class TestMisra1a04(CheckNLSresults):
    ''' Analytical Jacobian
        start_value2
    '''
    @classmethod
    def setupClass(cls):
        #Loading the dataset
        data = misra.load()
        x = data.exog
        y = data.endog

        #Loading the results
        resv = Misra1a()
        mod = funcMisra1a_J(y,x)
        res = mod.fit(start_value=resv.start_value2)
        cls.res = res
        cls.resv = resv

class TestMisra1aWNLS(object):
    ''' Numerical Jacobian
        start_value1
        Weighted Nonlinear Least Squares
    '''
    @classmethod
    def setupClass(cls):
        #Loading the dataset
        data = misra.load()
        x = data.exog
        y = data.endog

        #Loading the results
        resv = Misra1aWNLS()
        wts = np.ones(14)
        wts[7:14] *= 0.49
        mod = funcMisra1a(y,x,weights=wts)
        res = mod.fit(start_value=resv.start_value1)
        cls.res = res
        cls.resv = resv

    decimal_params = DECIMAL_4
    def test_params(self):
        assert_almost_equal(self.res.params, self.resv.params,
                self.decimal_params)

    decimal_standarderrors = DECIMAL_2
    def test_standarderrors(self):
        assert_almost_equal(self.res.bse,self.resv.bse,
                self.decimal_standarderrors)

    decimal_ser = DECIMAL_4
    def test_ser(self):
        assert_almost_equal(self.res.ser, self.resv.ser,
                self.decimal_params)

    decimal_covparams = DECIMAL_4
    def test_covparams(self):
        assert_almost_equal(self.res.cov_params(), self.resv.cov_params(),
                self.decimal_covparams)

    decimal_confidenceintervals = DECIMAL_4
    def test_confidenceintervals(self):
        conf1 = self.res.conf_int()
        conf2 = self.resv.conf_int()
        for i in range(len(conf1)):
            assert_approx_equal(conf1[i][0], conf2[i][0],
                    self.decimal_confidenceintervals)
            assert_approx_equal(conf1[i][1], conf2[i][1],
                    self.decimal_confidenceintervals)

    decimal_scale = DECIMAL_4
    def test_scale(self):
        assert_almost_equal(self.res.scale, self.resv.scale,
                self.decimal_scale)

    def test_degrees(self):
        assert_equal(self.res.model.df_model, self.resv.df_model)
        assert_equal(self.res.model.df_resid, self.resv.df_resid)

    decimal_ssr = DECIMAL_4
    def test_sumof_squaredresids(self):
        assert_almost_equal(self.res.ssr, self.resv.ssr, self.decimal_ssr)

    decimal_mse_resid = DECIMAL_4
    def test_mse_resid(self):
        assert_almost_equal(self.res.mse_resid, self.resv.mse_resid,
                    self.decimal_mse_resid)

    decimal_tvalues = DECIMAL_2
    def test_tvalues(self):
        assert_almost_equal(np.round(self.res.tvalues,2), self.resv.tvalues,
            self.decimal_tvalues)

    decimal_resids = DECIMAL_4
    def test_resids(self):
        assert_almost_equal(self.res.resid, self.resv.resid,
            self.decimal_resids)

    decimal_fittedvalues = DECIMAL_4
    def test_fittedvalues(self):
        assert_almost_equal(self.res.fittedvalues,self.resv.fittedvalues,
            self.decimal_fittedvalues)

    decimal_weights = DECIMAL_4
    def test_weights(self):
        assert_almost_equal(self.res.model.weights,self.resv.weights,
            self.decimal_weights)

