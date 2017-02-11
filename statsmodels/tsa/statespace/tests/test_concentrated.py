"""
Tests for concentrating the scale out of the loglikelihood function

Note: the univariate cases is well tested in test_sarimax.py

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
from .results import results_varmax
from statsmodels.tsa.statespace import varmax
from numpy.testing import assert_equal, assert_raises, assert_allclose


def check_concentrated_scale(filter_univariate=False, **kwargs):
    # Test that concentrating the scale out of the likelihood function works
    index = pd.date_range('1960-01-01', '1982-10-01', freq='QS')
    dta = pd.DataFrame(results_varmax.lutkepohl_data,
                       columns=['inv', 'inc', 'consump'], index=index)
    dta['dln_inv'] = np.log(dta['inv']).diff()
    dta['dln_inc'] = np.log(dta['inc']).diff()
    dta['dln_consump'] = np.log(dta['consump']).diff()

    endog = dta.ix['1960-04-01':'1978-10-01', ['dln_inv', 'dln_inc']]

    # Sometimes we can have slight differences if the Kalman filters
    # converge at different observations, so disable convergence.
    kwargs.update({'tolerance': 0})

    mod_orig = varmax.VARMAX(endog, **kwargs)
    mod_conc = varmax.VARMAX(endog, concentrate_scale=True, **kwargs)

    mod_orig.ssm.filter_univariate = filter_univariate
    mod_conc.ssm.filter_univariate = filter_univariate

    # Since VARMAX doesn't explicitly allow concentrating out the scale, for
    # now we will simulate it by setting the first variance to be 1.
    # Note that start_scale will not be the scale used for the non-concentrated
    # model, because we need to use the MLE scale estimated by the
    # concentrated model.
    conc_params = mod_conc.start_params
    start_scale = conc_params[mod_conc._params_state_cov][0]
    if kwargs.get('error_cov_type', 'unstructured') == 'diagonal':
        conc_params[mod_conc._params_state_cov] /= start_scale
    else:
        conc_params[mod_conc._params_state_cov] /= start_scale**0.5

    # Concentrated model smoothing
    res_conc = mod_conc.smooth(conc_params)
    scale = res_conc.scale

    # Map the concentrated parameters to the non-concentrated model
    orig_params = conc_params.copy()
    if kwargs.get('error_cov_type', 'unstructured') == 'diagonal':
        orig_params[mod_orig._params_state_cov] *= scale
    else:
        orig_params[mod_orig._params_state_cov] *= scale**0.5

    # Measurement error variances also get rescaled
    orig_params[mod_orig._params_obs_cov] *= scale

    # Non-oncentrated model smoothing
    res_orig = mod_orig.smooth(orig_params)

    # Test loglike
    # Need to reduce the tolerance when we have measurement error.
    assert_allclose(res_conc.llf, res_orig.llf)

    # Test state space representation matrices
    for name in mod_conc.ssm.shapes:
        if name == 'obs':
            continue
        assert_allclose(getattr(res_conc.filter_results, name),
                        getattr(res_orig.filter_results, name))

    # Test filter / smoother output
    scale = res_conc.scale
    d = res_conc.loglikelihood_burn

    filter_attr = ['predicted_state', 'filtered_state', 'forecasts',
                   'forecasts_error', 'kalman_gain']

    for name in filter_attr:
        actual = getattr(res_conc.filter_results, name)
        desired = getattr(res_orig.filter_results, name)
        assert_allclose(actual, desired)

    # Note: don't want to compare the elements from any diffuse
    # initialization for things like covariances, so only compare for
    # periods past the loglikelihood_burn period
    filter_attr_burn = ['standardized_forecasts_error',
                        'predicted_state_cov', 'filtered_state_cov',
                        'tmp1', 'tmp2', 'tmp3', 'tmp4']

    for name in filter_attr_burn:
        actual = getattr(res_conc.filter_results, name)[..., d:]
        desired = getattr(res_orig.filter_results, name)[..., d:]
        assert_allclose(actual, desired)

    smoothed_attr = ['smoothed_state', 'smoothed_state_cov',
                     'smoothed_state_autocov',
                     'smoothed_state_disturbance',
                     'smoothed_state_disturbance_cov',
                     'smoothed_measurement_disturbance',
                     'smoothed_measurement_disturbance_cov',
                     'scaled_smoothed_estimator',
                     'scaled_smoothed_estimator_cov', 'smoothing_error',
                     'smoothed_forecasts', 'smoothed_forecasts_error',
                     'smoothed_forecasts_error_cov']

    for name in smoothed_attr:
        actual = getattr(res_conc.filter_results, name)
        desired = getattr(res_orig.filter_results, name)
        assert_allclose(actual, desired)


def test_concentrated_scale_conventional():
    check_concentrated_scale(filter_univariate=False)
    check_concentrated_scale(filter_univariate=False, measurement_error=True)
    check_concentrated_scale(filter_univariate=False,
                             error_cov_type='diagonal')


def test_concentrated_scale_univariate():
    check_concentrated_scale(filter_univariate=True)
    check_concentrated_scale(filter_univariate=True, measurement_error=False)
    check_concentrated_scale(filter_univariate=True, error_cov_type='diagonal')
