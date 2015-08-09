"""
Tests for VARMAX models

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
import os

import warnings
from statsmodels.datasets import webuse
from statsmodels.tsa.statespace import varmax
from .results import results_varmax
from numpy.testing import assert_equal, assert_almost_equal, assert_raises, assert_allclose
from nose.exc import SkipTest

current_path = os.path.dirname(os.path.abspath(__file__))

var_path = 'results' + os.sep + 'results_var_stata.csv'
var_results = pd.read_csv(current_path + os.sep + var_path)

varmax_path = 'results' + os.sep + 'results_varmax_stata.csv'
varmax_results = pd.read_csv(current_path + os.sep + varmax_path)


class CheckVARMAX(object):
    """
    Test Vector Autoregression against Stata's `dfactor` code (Stata's
    `var` function uses OLS and not state space / MLE, so we can't get
    equivalent log-likelihoods)
    """

    def test_mle(self):
        # Fit with all transformations
        # results = self.model.fit(method='powell', disp=-1)
        results = self.model.fit(maxiter=100, disp=False)
        # Fit now without transformations
        self.model.enforce_stationarity = False
        self.model.enforce_invertibility = False
        results = self.model.fit(results.params, method='nm', maxiter=1000,
                                 disp=False)
        assert_allclose(results.llf, self.results.llf, rtol=1e-5)

    def test_loglike(self):
        assert_allclose(self.results.llf, self.true['loglike'], rtol=1e-6)

    def test_bse_oim(self):
        assert_allclose(self.results.bse**2, self.true['var_oim'], atol=1e-2)

    def test_aic(self):
        # We only get 3 digits from Stata
        assert_allclose(self.results.aic, self.true['aic'], atol=3)

    def test_bic(self):
        # We only get 3 digits from Stata
        assert_allclose(self.results.bic, self.true['bic'], atol=3)

    def test_predict(self, end, atol=1e-6):
        # Tests predict + forecast
        assert_allclose(
            self.results.predict(end=end),
            self.true['predict'].T,
            atol=atol)

    def test_dynamic_predict(self, end, dynamic, atol=1e-6):
        # Tests predict + dynamic predict + forecast
        assert_allclose(
            self.results.predict(end=end, dynamic=dynamic),
            self.true['dynamic_predict'].T,
            atol=atol)


class CheckLutkepohl(CheckVARMAX):
    def __init__(self, true, order, trend, error_cov_type, cov_type='oim',
             included_vars=['dln_inv', 'dln_inc', 'dln_consump'],
             **kwargs):
        self.true = true
        # 1960:Q1 - 1982:Q4
        dta = pd.DataFrame(
            results_varmax.lutkepohl_data, columns=['inv', 'inc', 'consump'],
            index=pd.date_range('1960-01-01', '1982-10-01', freq='QS'))

        dta['dln_inv'] = np.log(dta['inv']).diff()
        dta['dln_inc'] = np.log(dta['inc']).diff()
        dta['dln_consump'] = np.log(dta['consump']).diff()

        endog = dta.ix['1960-04-01':'1978-10-01', included_vars]

        self.model = varmax.VARMAX(endog, order=order, trend=trend,
                                   error_cov_type=error_cov_type, **kwargs)

        self.results = self.model.filter(true['params'], cov_type=cov_type)

    def test_predict(self):
        super(CheckLutkepohl, self).test_predict(end='1982-10-01')

    def test_dynamic_predict(self):
        super(CheckLutkepohl, self).test_dynamic_predict(end='1982-10-01', dynamic='1961-01-01')


class TestVAR(CheckLutkepohl):
    def __init__(self):
        true = results_varmax.lutkepohl_var1.copy()
        true['predict'] = var_results.ix[1:, ['predict_1', 'predict_2', 'predict_3']]
        true['dynamic_predict'] = var_results.ix[1:, ['dyn_predict_1', 'dyn_predict_2', 'dyn_predict_3']]
        super(TestVAR, self).__init__(
            true,  order=(1,0), trend='nc',
            error_cov_type="unstructured")


class TestVAR_diagonal(CheckLutkepohl):
    def __init__(self):
        true = results_varmax.lutkepohl_var1_diag.copy()
        true['predict'] = var_results.ix[1:, ['predict_diag1', 'predict_diag2', 'predict_diag3']]
        true['dynamic_predict'] = var_results.ix[1:, ['dyn_predict_diag1', 'dyn_predict_diag2', 'dyn_predict_diag3']]
        super(TestVAR_diagonal, self).__init__(
            true,  order=(1,0), trend='nc',
            error_cov_type="diagonal")


class TestVAR_obs_intercept(CheckLutkepohl):
    def __init__(self):
        true = results_varmax.lutkepohl_var1_obs_intercept.copy()
        true['predict'] = var_results.ix[1:, ['predict_int1', 'predict_int2', 'predict_int3']]
        true['dynamic_predict'] = var_results.ix[1:, ['dyn_predict_int1', 'dyn_predict_int2', 'dyn_predict_int3']]
        super(TestVAR_obs_intercept, self).__init__(
            true, order=(1,0), trend='nc',
            error_cov_type="diagonal", obs_intercept=true['obs_intercept'])

    def test_aic(self):
        # Since the obs_intercept is added in in an ad-hoc way here, the number
        # of parameters, and hence the aic and bic, will be off
        pass

    def test_bic(self):
        # Since the obs_intercept is added in in an ad-hoc way here, the number
        # of parameters, and hence the aic and bic, will be off
        pass

class TestVAR2(CheckLutkepohl):
    def __init__(self):
        true = results_varmax.lutkepohl_var2.copy()
        true['predict'] = var_results.ix[1:, ['predict_var2_1', 'predict_var2_2']]
        true['dynamic_predict'] = var_results.ix[1:, ['dyn_predict_var2_1', 'dyn_predict_var2_2']]
        super(TestVAR2, self).__init__(
            true, order=(2,0), trend='nc', error_cov_type='unstructured',
            included_vars=['dln_inv', 'dln_inc'])

    def test_bse_oim(self):
        # Exclude the covariance cholesky terms
        assert_allclose(
            self.results.bse[:-3]**2, self.true['var_oim'][:-3], atol=1e-2)


# class TestVARX(CheckVARMAX):
#     pass
# pandas, numpy; use 1-dim


class CheckFREDManufacturing(CheckVARMAX):
    def __init__(self, true, order, trend, error_cov_type, cov_type='oim',
                 **kwargs):
        self.true = true
        # 1960:Q1 - 1982:Q4
        dta = webuse('manufac', 'http://www.stata-press.com/data/r12/')

        dta.index = dta.month
        dta['dlncaputil'] = dta['lncaputil'].diff()
        dta['dlnhours'] = dta['lnhours'].diff()

        endog = dta.ix['1972-02-01':, ['dlncaputil', 'dlnhours']]

        self.model = varmax.VARMAX(endog, order=order, trend=trend,
                                   error_cov_type=error_cov_type, **kwargs)

        self.results = self.model.filter(true['params'], cov_type=cov_type)


class TestVARMA(CheckFREDManufacturing):
    """
    Test against the sspace VARMA example with some params set to zeros.
    """

    def __init__(self):
        true = results_varmax.fred_varma11.copy()
        true['predict'] = varmax_results.ix[1:, ['predict_varma11_1', 'predict_varma11_2']]
        true['dynamic_predict'] = varmax_results.ix[1:, ['dyn_predict_varma11_1', 'dyn_predict_varma11_2']]
        super(TestVARMA, self).__init__(
            true, order=(1,1), trend='nc', error_cov_type='diagonal')

    def test_mle(self):
        # Since the VARMA model here is generic (we're just forcing zeros
        # in some params) whereas Stata's is restricted, the MLE test isn't
        # meaninful
        pass

    def test_bse_oim(self):
        # Standard errors do not match Stata's
        raise SkipTest('Known failure: standard errors do not match.')

    def test_aic(self):
        # Since the VARMA model here is generic (we're just putting in zeros
        # for some params), Stata assumes a different estimated number of
        # parameters; hence the aic and bic, will be off
        pass

    def test_bic(self):
        # Since the VARMA model here is generic (we're just putting in zeros
        # for some params), Stata assumes a different estimated number of
        # parameters; hence the aic and bic, will be off
        pass

    def test_predict(self):
        super(TestVARMA, self).test_predict(end='2009-05-01', atol=1e-4)

    def test_dynamic_predict(self):
        super(TestVARMA, self).test_dynamic_predict(end='2009-05-01', dynamic='2000-01-01')


def test_specifications():
    # Tests for model specification and state space creation
    endog = np.arange(20).reshape(10,2)
    exog = np.arange(10)
    exog2 = pd.Series(exog, index=pd.date_range('2000-01-01', '2009-01-01', freq='AS'))

    # Test successful model creation
    mod = varmax.VARMAX(endog, exog=exog, order=(1,0))

    # Test successful model creation with pandas exog
    mod = varmax.VARMAX(endog, exog=exog2, order=(1,0))


def test_misspecifications():
    # Tests for model specification and misspecification exceptions
    endog = np.arange(20).reshape(10,2)

    # Bad trend specification
    assert_raises(ValueError, varmax.VARMAX, endog, order=(1,0), trend='')

    # Bad error_cov_type specification
    assert_raises(ValueError, varmax.VARMAX, endog, order=(1,0), error_cov_type='')

    # Bad order specification
    assert_raises(ValueError, varmax.VARMAX, endog, order=(0,0), trend='')

    # Warning with VARMA specification
    warnings.simplefilter("always")
    with warnings.catch_warnings(record=True) as w:
        varmax.VARMAX(endog, order=(1,1))
        message = ('Estimation of VARMA(p,q) models is not generically robust,'
                   ' due especially to identification issues.')
        assert_equal(str(w[0].message), message)

