import numpy as np
import pandas as pd
import pytest

from numpy.testing import assert_, assert_equal, assert_allclose, assert_raises

from statsmodels.tsa.arima import specification, params


def test_init():
    # Test initialization of the params

    # Basic test, with 1 of each parameter
    exog = pd.DataFrame([[0]], columns=['a'])
    spec = specification.SARIMAXSpecification(
        exog=exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
    p = params.SARIMAXParams(spec=spec)

    # Test things copied over from spec
    assert_equal(p.spec, spec)
    assert_equal(p.exog_names, ['a'])
    assert_equal(p.ar_names, ['ar.L1'])
    assert_equal(p.ma_names, ['ma.L1'])
    assert_equal(p.seasonal_ar_names, ['ar.S.L4'])
    assert_equal(p.seasonal_ma_names, ['ma.S.L4'])
    assert_equal(p.param_names, ['a', 'ar.L1', 'ma.L1', 'ar.S.L4', 'ma.S.L4',
                                 'sigma2'])

    assert_equal(p.k_exog_params, 1)
    assert_equal(p.k_ar_params, 1)
    assert_equal(p.k_ma_params, 1)
    assert_equal(p.k_seasonal_ar_params, 1)
    assert_equal(p.k_seasonal_ma_params, 1)
    assert_equal(p.k_params, 6)

    # Initial parameters should all be NaN
    assert_equal(p.params, np.nan)
    assert_equal(p.ar_params, [np.nan])
    assert_equal(p.ma_params, [np.nan])
    assert_equal(p.seasonal_ar_params, [np.nan])
    assert_equal(p.seasonal_ma_params, [np.nan])
    assert_equal(p.sigma2, np.nan)
    assert_equal(p.ar_poly.coef, np.r_[1, np.nan])
    assert_equal(p.ma_poly.coef, np.r_[1, np.nan])
    assert_equal(p.seasonal_ar_poly.coef, np.r_[1, 0, 0, 0, np.nan])
    assert_equal(p.seasonal_ma_poly.coef, np.r_[1, 0, 0, 0, np.nan])
    assert_equal(p.reduced_ar_poly.coef, np.r_[1, [np.nan] * 5])
    assert_equal(p.reduced_ma_poly.coef, np.r_[1, [np.nan] * 5])

    # Test other properties, methods
    assert_(not p.is_complete)
    assert_(not p.is_valid)
    assert_raises(ValueError, p.__getattribute__, 'is_stationary')
    assert_raises(ValueError, p.__getattribute__, 'is_invertible')
    desired = {
        'exog_params': [np.nan],
        'ar_params': [np.nan],
        'ma_params': [np.nan],
        'seasonal_ar_params': [np.nan],
        'seasonal_ma_params': [np.nan],
        'sigma2': np.nan}
    assert_equal(p.to_dict(), desired)
    desired = pd.Series([np.nan] * spec.k_params, index=spec.param_names)
    assert_allclose(p.to_pandas(), desired)

    # Test with different numbers of parameters for each
    exog = pd.DataFrame([[0, 0]], columns=['a', 'b'])
    spec = specification.SARIMAXSpecification(
        exog=exog, order=(3, 1, 2), seasonal_order=(5, 1, 6, 4))
    p = params.SARIMAXParams(spec=spec)
    # No real need to test names here, since they are already tested above for
    # the 1-param case, and tested more extensively in test for
    # SARIMAXSpecification
    assert_equal(p.k_exog_params, 2)
    assert_equal(p.k_ar_params, 3)
    assert_equal(p.k_ma_params, 2)
    assert_equal(p.k_seasonal_ar_params, 5)
    assert_equal(p.k_seasonal_ma_params, 6)
    assert_equal(p.k_params, 2 + 3 + 2 + 5 + 6 + 1)


def test_set_params_single():
    # Test setting parameters directly (i.e. we test setting the AR/MA
    # parameters by setting the lag polynomials elsewhere)
    # Here each type has only a single parameters
    exog = pd.DataFrame([[0]], columns=['a'])
    spec = specification.SARIMAXSpecification(
        exog=exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
    p = params.SARIMAXParams(spec=spec)

    def check(is_stationary='raise', is_invertible='raise'):
        assert_(not p.is_complete)
        assert_(not p.is_valid)
        if is_stationary == 'raise':
            assert_raises(ValueError, p.__getattribute__, 'is_stationary')
        else:
            assert_equal(p.is_stationary, is_stationary)
        if is_invertible == 'raise':
            assert_raises(ValueError, p.__getattribute__, 'is_invertible')
        else:
            assert_equal(p.is_invertible, is_invertible)

    # Set params one at a time, as scalars
    p.exog_params = -6.
    check()
    p.ar_params = -5.
    check()
    p.ma_params = -4.
    check()
    p.seasonal_ar_params = -3.
    check(is_stationary=False)
    p.seasonal_ma_params = -2.
    check(is_stationary=False, is_invertible=False)
    p.sigma2 = -1.
    # Finally, we have a complete set.
    assert_(p.is_complete)
    # But still not valid
    assert_(not p.is_valid)

    assert_equal(p.params, [-6, -5, -4, -3, -2, -1])
    assert_equal(p.exog_params, [-6])
    assert_equal(p.ar_params, [-5])
    assert_equal(p.ma_params, [-4])
    assert_equal(p.seasonal_ar_params, [-3])
    assert_equal(p.seasonal_ma_params, [-2])
    assert_equal(p.sigma2, -1.)

    # Lag polynomials
    assert_equal(p.ar_poly.coef, np.r_[1, 5])
    assert_equal(p.ma_poly.coef, np.r_[1, -4])
    assert_equal(p.seasonal_ar_poly.coef, np.r_[1, 0, 0, 0, 3])
    assert_equal(p.seasonal_ma_poly.coef, np.r_[1, 0, 0, 0, -2])
    # (1 - a L) (1 - b L^4) = (1 - a L - b L^4 + a b L^5)
    assert_equal(p.reduced_ar_poly.coef, np.r_[1, 5, 0, 0, 3, 15])
    # (1 + a L) (1 + b L^4) = (1 + a L + b L^4 + a b L^5)
    assert_equal(p.reduced_ma_poly.coef, np.r_[1, -4, 0, 0, -2, 8])

    # Override again, one at a time, now using lists
    p.exog_params = [1.]
    p.ar_params = [2.]
    p.ma_params = [3.]
    p.seasonal_ar_params = [4.]
    p.seasonal_ma_params = [5.]
    p.sigma2 = [6.]

    p.params = [1, 2, 3, 4, 5, 6]
    assert_equal(p.params, [1, 2, 3, 4, 5, 6])
    assert_equal(p.exog_params, [1])
    assert_equal(p.ar_params, [2])
    assert_equal(p.ma_params, [3])
    assert_equal(p.seasonal_ar_params, [4])
    assert_equal(p.seasonal_ma_params, [5])
    assert_equal(p.sigma2, 6.)

    # Override again, one at a time, now using arrays
    p.exog_params = np.array(6.)
    p.ar_params = np.array(5.)
    p.ma_params = np.array(4.)
    p.seasonal_ar_params = np.array(3.)
    p.seasonal_ma_params = np.array(2.)
    p.sigma2 = np.array(1.)

    assert_equal(p.params, [6, 5, 4, 3, 2, 1])
    assert_equal(p.exog_params, [6])
    assert_equal(p.ar_params, [5])
    assert_equal(p.ma_params, [4])
    assert_equal(p.seasonal_ar_params, [3])
    assert_equal(p.seasonal_ma_params, [2])
    assert_equal(p.sigma2, 1.)

    # Override again, now setting params all at once
    p.params = [1, 2, 3, 4, 5, 6]
    assert_equal(p.params, [1, 2, 3, 4, 5, 6])
    assert_equal(p.exog_params, [1])
    assert_equal(p.ar_params, [2])
    assert_equal(p.ma_params, [3])
    assert_equal(p.seasonal_ar_params, [4])
    assert_equal(p.seasonal_ma_params, [5])
    assert_equal(p.sigma2, 6.)

    # Lag polynomials
    assert_equal(p.ar_poly.coef, np.r_[1, -2])
    assert_equal(p.ma_poly.coef, np.r_[1, 3])
    assert_equal(p.seasonal_ar_poly.coef, np.r_[1, 0, 0, 0, -4])
    assert_equal(p.seasonal_ma_poly.coef, np.r_[1, 0, 0, 0, 5])
    # (1 - a L) (1 - b L^4) = (1 - a L - b L^4 + a b L^5)
    assert_equal(p.reduced_ar_poly.coef, np.r_[1, -2, 0, 0, -4, 8])
    # (1 + a L) (1 + b L^4) = (1 + a L + b L^4 + a b L^5)
    assert_equal(p.reduced_ma_poly.coef, np.r_[1, 3, 0, 0, 5, 15])


def test_set_params_single_nonconsecutive():
    # Test setting parameters directly (i.e. we test setting the AR/MA
    # parameters by setting the lag polynomials elsewhere)
    # Here each type has only a single parameters but has non-consecutive
    # lag orders
    exog = pd.DataFrame([[0]], columns=['a'])
    spec = specification.SARIMAXSpecification(
        exog=exog, order=([0, 1], 1, [0, 1]),
        seasonal_order=([0, 1], 1, [0, 1], 4))
    p = params.SARIMAXParams(spec=spec)

    def check(is_stationary='raise', is_invertible='raise'):
        assert_(not p.is_complete)
        assert_(not p.is_valid)
        if is_stationary == 'raise':
            assert_raises(ValueError, p.__getattribute__, 'is_stationary')
        else:
            assert_equal(p.is_stationary, is_stationary)
        if is_invertible == 'raise':
            assert_raises(ValueError, p.__getattribute__, 'is_invertible')
        else:
            assert_equal(p.is_invertible, is_invertible)

    # Set params one at a time, as scalars
    p.exog_params = -6.
    check()
    p.ar_params = -5.
    check()
    p.ma_params = -4.
    check()
    p.seasonal_ar_params = -3.
    check(is_stationary=False)
    p.seasonal_ma_params = -2.
    check(is_stationary=False, is_invertible=False)
    p.sigma2 = -1.
    # Finally, we have a complete set.
    assert_(p.is_complete)
    # But still not valid
    assert_(not p.is_valid)

    assert_equal(p.params, [-6, -5, -4, -3, -2, -1])
    assert_equal(p.exog_params, [-6])
    assert_equal(p.ar_params, [-5])
    assert_equal(p.ma_params, [-4])
    assert_equal(p.seasonal_ar_params, [-3])
    assert_equal(p.seasonal_ma_params, [-2])
    assert_equal(p.sigma2, -1.)

    # Lag polynomials
    assert_equal(p.ar_poly.coef, [1, 0, 5])
    assert_equal(p.ma_poly.coef, [1, 0, -4])
    assert_equal(p.seasonal_ar_poly.coef, [1, 0, 0, 0, 0, 0, 0, 0, 3])
    assert_equal(p.seasonal_ma_poly.coef, [1, 0, 0, 0, 0, 0, 0, 0, -2])
    # (1 - a L^2) (1 - b L^8) = (1 - a L^2 - b L^8 + a b L^10)
    assert_equal(p.reduced_ar_poly.coef, [1, 0, 5, 0, 0, 0, 0, 0, 3, 0, 15])
    # (1 + a L^2) (1 + b L^4) = (1 + a L^2 + b L^8 + a b L^10)
    assert_equal(p.reduced_ma_poly.coef, [1, 0, -4, 0, 0, 0, 0, 0, -2, 0, 8])

    # Override again, now setting params all at once
    p.params = [1, 2, 3, 4, 5, 6]
    assert_equal(p.params, [1, 2, 3, 4, 5, 6])
    assert_equal(p.exog_params, [1])
    assert_equal(p.ar_params, [2])
    assert_equal(p.ma_params, [3])
    assert_equal(p.seasonal_ar_params, [4])
    assert_equal(p.seasonal_ma_params, [5])
    assert_equal(p.sigma2, 6.)

    # Lag polynomials
    assert_equal(p.ar_poly.coef, np.r_[1, 0, -2])
    assert_equal(p.ma_poly.coef, np.r_[1, 0, 3])
    assert_equal(p.seasonal_ar_poly.coef, [1, 0, 0, 0, 0, 0, 0, 0, -4])
    assert_equal(p.seasonal_ma_poly.coef, [1, 0, 0, 0, 0, 0, 0, 0, 5])
    # (1 - a L^2) (1 - b L^8) = (1 - a L^2 - b L^8 + a b L^10)
    assert_equal(p.reduced_ar_poly.coef, [1, 0, -2, 0, 0, 0, 0, 0, -4, 0, 8])
    # (1 + a L^2) (1 + b L^4) = (1 + a L^2 + b L^8 + a b L^10)
    assert_equal(p.reduced_ma_poly.coef, [1, 0, 3, 0, 0, 0, 0, 0, 5, 0, 15])


def test_set_params_multiple():
    # Test setting parameters directly (i.e. we test setting the AR/MA
    # parameters by setting the lag polynomials elsewhere)
    # Here each type has multiple a single parameters
    exog = pd.DataFrame([[0, 0]], columns=['a', 'b'])
    spec = specification.SARIMAXSpecification(
        exog=exog, order=(2, 1, 2), seasonal_order=(2, 1, 2, 4))
    p = params.SARIMAXParams(spec=spec)

    p.params = [-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11]
    assert_equal(p.params,
                 [-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11])
    assert_equal(p.exog_params, [-1, 2])
    assert_equal(p.ar_params, [-3, 4])
    assert_equal(p.ma_params, [-5, 6])
    assert_equal(p.seasonal_ar_params, [-7, 8])
    assert_equal(p.seasonal_ma_params, [-9, 10])
    assert_equal(p.sigma2, -11)

    # Lag polynomials
    assert_equal(p.ar_poly.coef, np.r_[1, 3, -4])
    assert_equal(p.ma_poly.coef, np.r_[1, -5, 6])
    assert_equal(p.seasonal_ar_poly.coef, np.r_[1, 0, 0, 0, 7, 0, 0, 0, -8])
    assert_equal(p.seasonal_ma_poly.coef, np.r_[1, 0, 0, 0, -9, 0, 0, 0, 10])
    # (1 - a_1 L - a_2 L^2) (1 - b_1 L^4 - b_2 L^8) =
    #     (1 - b_1 L^4 - b_2 L^8) +
    #     (-a_1 L + a_1 b_1 L^5 + a_1 b_2 L^9) +
    #     (-a_2 L^2 + a_2 b_1 L^6 + a_2 b_2 L^10) =
    #     1 - a_1 L - a_2 L^2 - b_1 L^4 + a_1 b_1 L^5 +
    #     a_2 b_1 L^6 - b_2 L^8 + a_1 b_2 L^9 + a_2 b_2 L^10
    assert_equal(p.reduced_ar_poly.coef,
                 [1, 3, -4, 0, 7, (-3 * -7), (4 * -7), 0, -8, (-3 * 8), 4 * 8])
    # (1 + a_1 L + a_2 L^2) (1 + b_1 L^4 + b_2 L^8) =
    #     (1 + b_1 L^4 + b_2 L^8) +
    #     (a_1 L + a_1 b_1 L^5 + a_1 b_2 L^9) +
    #     (a_2 L^2 + a_2 b_1 L^6 + a_2 b_2 L^10) =
    #     1 + a_1 L + a_2 L^2 + b_1 L^4 + a_1 b_1 L^5 +
    #     a_2 b_1 L^6 + b_2 L^8 + a_1 b_2 L^9 + a_2 b_2 L^10
    assert_equal(p.reduced_ma_poly.coef,
                 [1, -5, 6, 0, -9, (-5 * -9), (6 * -9),
                  0, 10, (-5 * 10), (6 * 10)])


def test_set_poly_short_lags():
    # Basic example (short lag orders)
    exog = pd.DataFrame([[0, 0]], columns=['a', 'b'])
    spec = specification.SARIMAXSpecification(
        exog=exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
    p = params.SARIMAXParams(spec=spec)

    # Valid polynomials
    p.ar_poly = [1, -0.5]
    assert_equal(p.ar_params, [0.5])
    p.ar_poly = np.polynomial.Polynomial([1, -0.55])
    assert_equal(p.ar_params, [0.55])
    p.ma_poly = [1, 0.3]
    assert_equal(p.ma_params, [0.3])
    p.ma_poly = np.polynomial.Polynomial([1, 0.35])
    assert_equal(p.ma_params, [0.35])

    p.seasonal_ar_poly = [1, 0, 0, 0, -0.2]
    assert_equal(p.seasonal_ar_params, [0.2])
    p.seasonal_ar_poly = np.polynomial.Polynomial([1, 0, 0, 0, -0.25])
    assert_equal(p.seasonal_ar_params, [0.25])
    p.seasonal_ma_poly = [1, 0, 0, 0, 0.1]
    assert_equal(p.seasonal_ma_params, [0.1])
    p.seasonal_ma_poly = np.polynomial.Polynomial([1, 0, 0, 0, 0.15])
    assert_equal(p.seasonal_ma_params, [0.15])

    # Invalid polynomials
    # Must have 1 in the initial position
    assert_raises(ValueError, p.__setattr__, 'ar_poly', [2, -0.5])
    assert_raises(ValueError, p.__setattr__, 'ma_poly', [2, 0.3])
    assert_raises(ValueError, p.__setattr__, 'seasonal_ar_poly',
                  [2, 0, 0, 0, -0.2])
    assert_raises(ValueError, p.__setattr__, 'seasonal_ma_poly',
                  [2, 0, 0, 0, 0.1])
    # Too short
    assert_raises(ValueError, p.__setattr__, 'ar_poly', 1)
    assert_raises(ValueError, p.__setattr__, 'ar_poly', [1])
    assert_raises(ValueError, p.__setattr__, 'ma_poly', 1)
    assert_raises(ValueError, p.__setattr__, 'ma_poly', [1])
    assert_raises(ValueError, p.__setattr__, 'seasonal_ar_poly', 1)
    assert_raises(ValueError, p.__setattr__, 'seasonal_ar_poly', [1])
    assert_raises(ValueError, p.__setattr__, 'seasonal_ar_poly', [1, 0, 0, 0])
    assert_raises(ValueError, p.__setattr__, 'seasonal_ma_poly', 1)
    assert_raises(ValueError, p.__setattr__, 'seasonal_ma_poly', [1])
    assert_raises(ValueError, p.__setattr__, 'seasonal_ma_poly', [1, 0, 0, 0])
    # Too long
    assert_raises(ValueError, p.__setattr__, 'ar_poly', [1, -0.5, 0.2])
    assert_raises(ValueError, p.__setattr__, 'ma_poly', [1, 0.3, 0.2])
    assert_raises(ValueError, p.__setattr__, 'seasonal_ar_poly',
                  [1, 0, 0, 0, 0.1, 0])
    assert_raises(ValueError, p.__setattr__, 'seasonal_ma_poly',
                  [1, 0, 0, 0, 0.1, 0])
    # Number in invalid location (only for seasonal polynomials)
    assert_raises(ValueError, p.__setattr__, 'seasonal_ar_poly',
                  [1, 1, 0, 0, 0, -0.2])
    assert_raises(ValueError, p.__setattr__, 'seasonal_ma_poly',
                  [1, 1, 0, 0, 0, 0.1])


def test_set_poly_short_lags_nonconsecutive():
    # Short but non-consecutive lag orders
    exog = pd.DataFrame([[0, 0]], columns=['a', 'b'])
    spec = specification.SARIMAXSpecification(
        exog=exog, order=([0, 1], 1, [0, 1]),
        seasonal_order=([0, 1], 1, [0, 1], 4))
    p = params.SARIMAXParams(spec=spec)

    # Valid polynomials
    p.ar_poly = [1, 0, -0.5]
    assert_equal(p.ar_params, [0.5])
    p.ar_poly = np.polynomial.Polynomial([1, 0, -0.55])
    assert_equal(p.ar_params, [0.55])
    p.ma_poly = [1, 0, 0.3]
    assert_equal(p.ma_params, [0.3])
    p.ma_poly = np.polynomial.Polynomial([1, 0, 0.35])
    assert_equal(p.ma_params, [0.35])

    p.seasonal_ar_poly = [1, 0, 0, 0, 0, 0, 0, 0, -0.2]
    assert_equal(p.seasonal_ar_params, [0.2])
    p.seasonal_ar_poly = (
        np.polynomial.Polynomial([1, 0, 0, 0, 0, 0, 0, 0, -0.25]))
    assert_equal(p.seasonal_ar_params, [0.25])
    p.seasonal_ma_poly = [1, 0, 0, 0, 0, 0, 0, 0, 0.1]
    assert_equal(p.seasonal_ma_params, [0.1])
    p.seasonal_ma_poly = (
        np.polynomial.Polynomial([1, 0, 0, 0, 0, 0, 0, 0, 0.15]))
    assert_equal(p.seasonal_ma_params, [0.15])

    # Invalid polynomials
    # Number in invalid (i.e. an excluded lag) location
    # (now also for non-seasonal polynomials)
    assert_raises(ValueError, p.__setattr__, 'ar_poly', [1, 1, -0.5])
    assert_raises(ValueError, p.__setattr__, 'ma_poly', [1, 1, 0.3])
    assert_raises(ValueError, p.__setattr__, 'seasonal_ar_poly',
                  [1, 0, 0, 0, 1., 0, 0, 0, -0.2])
    assert_raises(ValueError, p.__setattr__, 'seasonal_ma_poly',
                  [1, 0, 0, 0, 1., 0, 0, 0, 0.1])


def test_set_poly_longer_lags():
    # Test with higher order polynomials
    exog = pd.DataFrame([[0, 0]], columns=['a', 'b'])
    spec = specification.SARIMAXSpecification(
        exog=exog, order=(2, 1, 2), seasonal_order=(2, 1, 2, 4))
    p = params.SARIMAXParams(spec=spec)

    # Setup the non-AR/MA values
    p.exog_params = [-1, 2]
    p.sigma2 = -11

    # Lag polynomials
    p.ar_poly = np.r_[1, 3, -4]
    p.ma_poly = np.r_[1, -5, 6]
    p.seasonal_ar_poly = np.r_[1, 0, 0, 0, 7, 0, 0, 0, -8]
    p.seasonal_ma_poly = np.r_[1, 0, 0, 0, -9, 0, 0, 0, 10]

    # Test that parameters were set correctly
    assert_equal(p.params,
                 [-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11])
    assert_equal(p.exog_params, [-1, 2])
    assert_equal(p.ar_params, [-3, 4])
    assert_equal(p.ma_params, [-5, 6])
    assert_equal(p.seasonal_ar_params, [-7, 8])
    assert_equal(p.seasonal_ma_params, [-9, 10])
    assert_equal(p.sigma2, -11)


def test_is_stationary():
    # Tests for the `is_stationary` property
    spec = specification.SARIMAXSpecification(
        order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
    p = params.SARIMAXParams(spec=spec)

    # Test stationarity
    assert_raises(ValueError, p.__getattribute__, 'is_stationary')
    p.ar_params = [0.5]
    p.seasonal_ar_params = [0]
    assert_(p.is_stationary)
    p.ar_params = [1.0]
    assert_(not p.is_stationary)

    p.ar_params = [0]
    p.seasonal_ar_params = [0.5]
    assert_(p.is_stationary)
    p.seasonal_ar_params = [1.0]
    assert_(not p.is_stationary)

    p.ar_params = [0.2]
    p.seasonal_ar_params = [0.2]
    assert_(p.is_stationary)
    p.ar_params = [0.99]
    p.seasonal_ar_params = [0.99]
    assert_(p.is_stationary)
    p.ar_params = [1.]
    p.seasonal_ar_params = [1.]
    assert_(not p.is_stationary)


def test_is_invertible():
    # Tests for the `is_invertible` property
    spec = specification.SARIMAXSpecification(
        order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
    p = params.SARIMAXParams(spec=spec)

    # Test invertibility
    assert_raises(ValueError, p.__getattribute__, 'is_invertible')
    p.ma_params = [0.5]
    p.seasonal_ma_params = [0]
    assert_(p.is_invertible)
    p.ma_params = [1.0]
    assert_(not p.is_invertible)

    p.ma_params = [0]
    p.seasonal_ma_params = [0.5]
    assert_(p.is_invertible)
    p.seasonal_ma_params = [1.0]
    assert_(not p.is_invertible)

    p.ma_params = [0.2]
    p.seasonal_ma_params = [0.2]
    assert_(p.is_invertible)
    p.ma_params = [0.99]
    p.seasonal_ma_params = [0.99]
    assert_(p.is_invertible)
    p.ma_params = [1.]
    p.seasonal_ma_params = [1.]
    assert_(not p.is_invertible)


def test_is_valid():
    # Additional tests for the `is_valid` property (tests for NaN checks were
    # already done in `test_set_params_single`).
    spec = specification.SARIMAXSpecification(
        order=(1, 1, 1), seasonal_order=(1, 1, 1, 4),
        enforce_stationarity=True, enforce_invertibility=True)
    p = params.SARIMAXParams(spec=spec)

    # Doesn't start out as valid
    assert_(not p.is_valid)
    # Given stationary / invertible values, it is valid
    p.params = [0.5, 0.5, 0.5, 0.5, 1.]
    assert_(p.is_valid)
    # With either non-stationary or non-invertible values, not valid
    p.params = [1., 0.5, 0.5, 0.5, 1.]
    assert_(not p.is_valid)
    p.params = [0.5, 1., 0.5, 0.5, 1.]
    assert_(not p.is_valid)
    p.params = [0.5, 0.5, 1., 0.5, 1.]
    assert_(not p.is_valid)
    p.params = [0.5, 0.5, 0.5, 1., 1.]
    assert_(not p.is_valid)


def test_repr_str():
    exog = pd.DataFrame([[0, 0]], columns=['a', 'b'])
    spec = specification.SARIMAXSpecification(
        exog=exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
    p = params.SARIMAXParams(spec=spec)

    # Check when we haven't given any parameters
    assert_equal(repr(p), 'SARIMAXParams(exog=[nan nan], ar=[nan], ma=[nan],'
                          ' seasonal_ar=[nan], seasonal_ma=[nan], sigma2=nan)')
    # assert_equal(str(p), '[nan nan nan nan nan nan nan]')

    p.exog_params = [1, 2]
    assert_equal(repr(p), 'SARIMAXParams(exog=[1. 2.], ar=[nan], ma=[nan],'
                          ' seasonal_ar=[nan], seasonal_ma=[nan], sigma2=nan)')
    # assert_equal(str(p), '[ 1.  2. nan nan nan nan nan]')

    p.ar_params = [0.5]
    assert_equal(repr(p), 'SARIMAXParams(exog=[1. 2.], ar=[0.5], ma=[nan],'
                          ' seasonal_ar=[nan], seasonal_ma=[nan], sigma2=nan)')
    # assert_equal(str(p), '[1.  2.  0.5 nan nan nan nan]')

    p.ma_params = [0.2]
    assert_equal(repr(p), 'SARIMAXParams(exog=[1. 2.], ar=[0.5], ma=[0.2],'
                          ' seasonal_ar=[nan], seasonal_ma=[nan], sigma2=nan)')
    # assert_equal(str(p), '[1.  2.  0.5 0.2 nan nan nan]')

    p.seasonal_ar_params = [0.001]
    assert_equal(repr(p), 'SARIMAXParams(exog=[1. 2.], ar=[0.5], ma=[0.2],'
                          ' seasonal_ar=[0.001], seasonal_ma=[nan],'
                          ' sigma2=nan)')
    # assert_equal(str(p),
    #              '[1.e+00 2.e+00 5.e-01 2.e-01 1.e-03    nan    nan]')

    p.seasonal_ma_params = [-0.001]
    assert_equal(repr(p), 'SARIMAXParams(exog=[1. 2.], ar=[0.5], ma=[0.2],'
                          ' seasonal_ar=[0.001], seasonal_ma=[-0.001],'
                          ' sigma2=nan)')
    # assert_equal(str(p), '[ 1.e+00  2.e+00  5.e-01  2.e-01  1.e-03'
    #                      ' -1.e-03     nan]')

    p.sigma2 = 10.123
    assert_equal(repr(p), 'SARIMAXParams(exog=[1. 2.], ar=[0.5], ma=[0.2],'
                          ' seasonal_ar=[0.001], seasonal_ma=[-0.001],'
                          ' sigma2=10.123)')
    # assert_equal(str(p), '[ 1.0000e+00  2.0000e+00  5.0000e-01  2.0000e-01'
    #                      '  1.0000e-03 -1.0000e-03\n  1.0123e+01]')


@pytest.mark.parametrize(
    'fixed_params, expected_params',
    [
        # empty fixed_params
        (
            {},
            np.full(9, np.nan)
        ),
        # fix partial parameters for each parameter type
        (
            {'x1': 0, 'ar.L1': 1, 'ma.L2': -1, 'ar.S.L10': 10},
            np.array([0, np.nan, 1, np.nan, np.nan, -1, np.nan, 10, np.nan])
        ),
        # fix all parameters for select parameter types
        (
            {'x1': 0, 'x2': 4, 'ma.L1': 0, 'ma.L2': 3},
            np.array([0, 4, np.nan, np.nan, 0, 3, np.nan, np.nan, np.nan])
        ),
        # all parameters are fixed
        (
            {
                'x1': 0, 'x2': 1,
                'ar.L1': 2, 'ar.L3': 3,
                'ma.L1': 4, 'ma.L2': 5,
                'ar.S.L5': 6, 'ar.S.L10': 7,
                'sigma2': 8
            },
            np.array(list(range(9)))
        )
    ]
)
def test_set_fixed_params(fixed_params, expected_params):
    spec = specification.SARIMAXSpecification(
        exog=np.random.random(size=(20, 2)),
        ar_order=[1, 0, 1], ma_order=2,
        seasonal_order=[2, 1, 0, 5],
    )
    p = params.SARIMAXParams(spec=spec)
    p.set_fixed_params(fixed_params)

    # use spec.split_params to speed up simplify the code
    expected_params_split = spec.split_params(
        expected_params, allow_infnan=True
    )

    # test fixed + free params
    assert_equal(p.params, expected_params)
    assert_equal(p.exog_params, expected_params_split['exog_params'])
    assert_equal(p.ar_params, expected_params_split['ar_params'])
    assert_equal(p.ma_params, expected_params_split['ma_params'])
    assert_equal(
        p.seasonal_ar_params, expected_params_split['seasonal_ar_params']
    )
    assert_equal(
        p.seasonal_ma_params, expected_params_split['seasonal_ma_params']
    )
    assert_equal(p.sigma2, expected_params_split['sigma2'])

    # test is_fixed bool
    assert_equal(p.is_param_fixed, ~np.isnan(expected_params))
    assert_equal(
        p.is_exog_param_fixed,
        ~np.isnan(expected_params_split['exog_params'])
    )
    assert_equal(
        p.is_ar_param_fixed,
        ~np.isnan(expected_params_split['ar_params'])
    )
    assert_equal(
        p.is_ma_param_fixed,
        ~np.isnan(expected_params_split['ma_params'])
    )
    assert_equal(
        p.is_seasonal_ar_param_fixed,
        ~np.isnan(expected_params_split['seasonal_ar_params'])
    )
    assert_equal(
        p.is_seasonal_ma_param_fixed,
        ~np.isnan(expected_params_split['seasonal_ma_params'])
    )

    # test fixed params
    assert_equal(p.fixed_params, expected_params[p.is_param_fixed])
    assert_equal(
        p.fixed_exog_params,
        expected_params_split['exog_params'][p.is_exog_param_fixed]
    )
    assert_equal(
        p.fixed_ar_params,
        expected_params_split['ar_params'][p.is_ar_param_fixed]
    )
    assert_equal(
        p.fixed_ma_params,
        expected_params_split['ma_params'][p.is_ma_param_fixed]
    )
    assert_equal(
        p.fixed_seasonal_ar_params,
        expected_params_split['seasonal_ar_params'][
            p.is_seasonal_ar_param_fixed
        ]
    )
    assert_equal(
        p.fixed_seasonal_ma_params,
        expected_params_split['seasonal_ma_params'][
            p.is_seasonal_ma_param_fixed
        ]
    )

    # test free params
    assert_equal(p.free_params, expected_params[~p.is_param_fixed])
    assert_equal(
        p.free_exog_params,
        expected_params_split['exog_params'][~p.is_exog_param_fixed]
    )
    assert_equal(
        p.free_ar_params,
        expected_params_split['ar_params'][~p.is_ar_param_fixed]
    )
    assert_equal(
        p.free_ma_params,
        expected_params_split['ma_params'][~p.is_ma_param_fixed]
    )
    assert_equal(
        p.free_seasonal_ar_params,
        expected_params_split['seasonal_ar_params'][
            ~p.is_seasonal_ar_param_fixed
        ]
    )
    assert_equal(
        p.free_seasonal_ma_params,
        expected_params_split['seasonal_ma_params'][
            ~p.is_seasonal_ma_param_fixed
        ]
    )


def test_set_invalid_fixed_params():
    spec = specification.SARIMAXSpecification(
        exog=np.random.random(size=(20, 2)),
        ar_order=[1, 0, 1], ma_order=2,
        seasonal_order=[2, 1, 0, 5],
        concentrate_scale=True
    )
    p = params.SARIMAXParams(spec=spec)
    invalid_fixed_params = {'sigma2': 1}
    with pytest.raises(ValueError, match='Invalid fixed parameter'):
        p.set_fixed_params(invalid_fixed_params, validate=True)


def test_overwriting_fixed_params():
    # 1. directly test `_warn_fixed_overwrite` private method
    with pytest.warns(UserWarning, match='Overwriting 1'):
        params.SARIMAXParams._warn_fixed_overwrite(
            original_value=np.array([1, 1./7., np.nan, np.nan]),
            new_value=np.array([2, 1./7, np.nan, np.nan]),
            is_fixed_bool=np.array([True, True, True, False])
        )

    # 2. test with param setters (e.g. `set_fixed_params`, `params`,
    # `free_params`)
    spec = specification.SARIMAXSpecification(
        exog=np.random.random(size=(20, 2)),
        ar_order=[1, 0, 1], ma_order=2,
        seasonal_order=[2, 1, 0, 5],
    )
    p = params.SARIMAXParams(spec=spec)

    # 2.1 set fixed params once, and check results
    fixed_params_1 = {'x1': 10, 'ar.L1': 5}
    p.set_fixed_params(fixed_params_1)
    assert_equal(
        p.is_param_fixed,
        [True, False, True, False, False, False, False, False, False]
    )
    assert_equal(
        p.params,
        [10, np.nan, 5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    )
    assert_equal(p.fixed_params, [10, 5])
    assert_equal(p.free_params, [np.nan] * 7)

    # 2.2 call `set_fixed_params` a second time with different parameters, and
    # check that no warnings are raised
    fixed_params_2 = {'ma.L2': -0.1, 'sigma2': 2}
    with pytest.warns(None) as warning_record:
        p.set_fixed_params(fixed_params_2)
    assert len(warning_record) == 0   # no warnings
    assert_equal(
        p.is_param_fixed,
        [True, False, True, False, False, True, False, False, True]
    )
    assert_equal(
        p.params,
        [10, np.nan, 5, np.nan, np.nan, -0.1, np.nan, np.nan, 2]
    )
    assert_equal(p.fixed_params, [10, 5, -0.1, 2])
    assert_equal(p.free_params, [np.nan] * 5)

    # 2.3 call `set_fixed_params` a third time which overwrites some previously
    # fixed parameters with different values and check that no warnings are
    # raised
    fixed_params_3 = {'x1': 10, 'x2': 99, 'ar.L1': np.e}
    with pytest.warns(None) as warning_record:
        p.set_fixed_params(fixed_params_3)
    assert len(warning_record) == 0   # no warnings
    assert_equal(
        p.is_param_fixed,
        [True, True, True, False, False, True, False, False, True]
    )
    assert_equal(
        p.params,
        [10, 99, np.e, np.nan, np.nan, -0.1, np.nan, np.nan, 2]
    )
    assert_equal(p.fixed_params, [10, 99, np.e, -0.1, 2])
    assert_equal(p.free_params, [np.nan] * 4)

    # 2.4 call a few param setters, and check that no warnings are raised
    with pytest.warns(None) as warning_record:
        p.free_ar_params = [1]
        p.free_seasonal_ar_params = [1. / 7., 0]
    assert len(warning_record) == 0   # no warnings
    assert_equal(
        p.is_param_fixed,
        [True, True, True, False, False, True, False, False, True]
    )  # is_fixed status does not change
    assert_equal(p.params, [10, 99, np.e, 1, np.nan, -0.1, 1./7., 0, 2])
    assert_equal(p.fixed_params, [10, 99, np.e, -0.1, 2])
    assert_equal(p.free_params, [1, np.nan, 1./7., 0])

    # 2.5 call sigma2 setter to overwrite previously fixed sigma2
    with pytest.warns(UserWarning, match='Overwriting 1'):
        p.sigma2 = 4
    assert_equal(
        p.is_param_fixed,
        [True, True, True, False, False, True, False, False, True]
    )  # is_fixed status does not change
    assert_equal(p.params, [10, 99, np.e, 1, np.nan, -0.1, 1./7., 0, 4])
    assert_equal(p.fixed_params, [10, 99, np.e, -0.1, 4])
    assert_equal(p.free_params, [1, np.nan, 1./7., 0])

    # 2.6 call full param setter to overwrite some fixed params with different
    # values
    with pytest.warns(UserWarning, match='Overwriting 3'):
        p.params = [10, 1, 2, 3, 4, -0.1, 5, 6, 7]
    assert_equal(
        p.is_param_fixed,
        [True, True, True, False, False, True, False, False, True]
    )  # is_fixed status does not change
    assert_equal(p.params, [10, 1, 2, 3, 4, -0.1, 5, 6, 7])
    assert_equal(p.fixed_params, [10, 1, 2, -0.1, 7])
    assert_equal(p.free_params, [3, 4, 5, 6])


def test_reset_fixed_params():
    spec = specification.SARIMAXSpecification(
        exog=np.random.random(size=(20, 2)),
        ar_order=[1, 0, 1], ma_order=2,
        seasonal_order=[2, 1, 0, 5],
    )
    fixed_params = {'x1': 0, 'ar.L1': 1, 'ma.L2': -1, 'ar.S.L10': 10,
                    'sigma2': 1}

    # check that with `keep_param_value=True`, the previously fixed params'
    # statuses are correctly reset, while the values are kept
    p = params.SARIMAXParams(spec=spec)
    p.set_fixed_params(fixed_params, validate=False)
    p.reset_fixed_params(keep_param_value=True)
    assert_equal(p.is_param_fixed, [False] * p.k_params)
    assert_equal(
        p.params,
        [0, np.nan, 1, np.nan, np.nan, -1, np.nan, 10, 1]
    )

    # check that with `keep_param_value=False`, the previously fixed params'
    # are also correctly set back to np.nan
    p = params.SARIMAXParams(spec=spec)
    p.set_fixed_params(fixed_params, validate=False)
    p.reset_fixed_params(keep_param_value=False)
    assert_equal(p.is_param_fixed, [False] * 9)
    assert_equal(p.params, [np.nan] * 9)


def test_set_free_params():
    spec = specification.SARIMAXSpecification(
        exog=np.random.random(size=(20, 2)),
        ar_order=[1, 0, 1], ma_order=2,
        seasonal_order=[2, 1, 3, 5],
    )
    full_params = list(range(12))
    full_param_split = spec.split_params(full_params)

    def check_params_equal(actual_p, expected_p):
        # exog params
        assert_equal(actual_p.free_exog_params, expected_p.free_exog_params)
        assert_equal(actual_p.exog_params, expected_p.exog_params)
        assert_equal(actual_p.exog_params, full_param_split['exog_params'])
        # ar params
        assert_equal(actual_p.free_ar_params, expected_p.free_ar_params)
        assert_equal(actual_p.ar_params, expected_p.ar_params)
        assert_equal(actual_p.ar_params, full_param_split['ar_params'])
        # ma params
        assert_equal(actual_p.free_ma_params, expected_p.free_ma_params)
        assert_equal(actual_p.ma_params, expected_p.ma_params)
        assert_equal(actual_p.ma_params, full_param_split['ma_params'])
        # seasonal ar params
        assert_equal(
            actual_p.free_seasonal_ar_params,
            expected_p.free_seasonal_ar_params
        )
        assert_equal(
            actual_p.seasonal_ar_params,
            expected_p.seasonal_ar_params
        )
        assert_equal(
            actual_p.seasonal_ar_params,
            full_param_split['seasonal_ar_params']
        )
        # seasonal ma params
        assert_equal(
            actual_p.free_seasonal_ma_params,
            expected_p.free_seasonal_ma_params)
        assert_equal(
            actual_p.seasonal_ma_params,
            expected_p.seasonal_ma_params
        )
        assert_equal(
            actual_p.seasonal_ma_params,
            full_param_split['seasonal_ma_params']
        )
        # full params
        assert_equal(actual_p.free_params, expected_p.free_params)
        assert_equal(actual_p.params, expected_p.params)
        assert_equal(actual_p.params, full_params)

    # 1. before setting any fixed params, using free param setters and using
    # full param setters should be equivalent

    # 1.1 setting entire param array
    p_1 = params.SARIMAXParams(spec=spec)   # to be set with free param setters
    p_2 = params.SARIMAXParams(spec=spec)   # to be set with full param setters

    p_1.free_params = full_params
    p_2.params = full_params

    check_params_equal(p_1, p_2)

    # 1.2 setting specific param arrays
    p_1 = params.SARIMAXParams(spec=spec)  # to be set with free param setters
    p_2 = params.SARIMAXParams(spec=spec)  # to be set with full param setters

    p_1.free_exog_params = full_param_split['exog_params']
    p_1.free_ar_params = full_param_split['ar_params']
    p_1.free_ma_params = full_param_split['ma_params']
    p_1.free_seasonal_ar_params = full_param_split['seasonal_ar_params']
    p_1.free_seasonal_ma_params = full_param_split['seasonal_ma_params']
    p_1.sigma2 = full_param_split['sigma2']

    p_2.exog_params = full_param_split['exog_params']
    p_2.ar_params = full_param_split['ar_params']
    p_2.ma_params = full_param_split['ma_params']
    p_2.seasonal_ar_params = full_param_split['seasonal_ar_params']
    p_2.seasonal_ma_params = full_param_split['seasonal_ma_params']
    p_2.sigma2 = full_param_split['sigma2']

    check_params_equal(p_1, p_2)

    # 2. setting some fixed params before setting free params
    fixed_params = {
        'x1': 0, 'ar.L1': 2, 'ma.L1': 4,
        'ar.S.L5': 6, 'ma.S.L5': 8, 'ma.S.L15': 10
    }

    p_1 = params.SARIMAXParams(spec=spec)  # to be set with free param setters
    p_2 = params.SARIMAXParams(spec=spec)  # to be set with full param setters

    p_1.set_fixed_params(fixed_params)
    p_1.free_exog_params = full_param_split['exog_params'][1:]
    p_1.free_ar_params = full_param_split['ar_params'][1:]
    p_1.free_ma_params = full_param_split['ma_params'][1:]
    p_1.free_seasonal_ar_params = full_param_split['seasonal_ar_params'][1:]
    p_1.free_seasonal_ma_params = full_param_split['seasonal_ma_params'][1:-1]
    p_1.sigma2 = full_param_split['sigma2']

    p_2.set_fixed_params(fixed_params)
    p_2.params = full_params

    check_params_equal(p_1, p_2)
