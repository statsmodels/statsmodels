"""
Markov switching models

Author: Chad Fulton
License: BSD-3
"""
from __future__ import division, absolute_import, print_function
from statsmodels.compat.scipy import logsumexp

import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd

from statsmodels.tools.tools import Bunch
from statsmodels.tools.numdiff import approx_fprime_cs, approx_hess_cs
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, bic, hqic
from statsmodels.tools.tools import pinv_extended
from statsmodels.tools.sm_exceptions import EstimationWarning

import statsmodels.base.wrapper as wrap
from statsmodels.base.data import PandasData

import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.statespace.tools import find_best_blas_type, prepare_exog

from statsmodels.tsa.regime_switching._hamilton_filter import (
    shamilton_filter, dhamilton_filter, chamilton_filter, zhamilton_filter)
from statsmodels.tsa.regime_switching._kim_smoother import (
    skim_smoother, dkim_smoother, ckim_smoother, zkim_smoother)

prefix_hamilton_filter_map = {
    's': shamilton_filter, 'd': dhamilton_filter,
    'c': chamilton_filter, 'z': zhamilton_filter
}

prefix_kim_smoother_map = {
    's': skim_smoother, 'd': dkim_smoother,
    'c': ckim_smoother, 'z': zkim_smoother
}


def _logistic(x):
    """
    Note that this is not a vectorized function
    """
    x = np.array(x)
    # np.exp(x) / (1 + np.exp(x))
    if x.ndim == 0:
        y = np.reshape(x, (1, 1, 1))
    # np.exp(x[i]) / (1 + np.sum(np.exp(x[:])))
    elif x.ndim == 1:
        y = np.reshape(x, (len(x), 1, 1))
    # np.exp(x[i,t]) / (1 + np.sum(np.exp(x[:,t])))
    elif x.ndim == 2:
        y = np.reshape(x, (x.shape[0], 1, x.shape[1]))
    # np.exp(x[i,j,t]) / (1 + np.sum(np.exp(x[:,j,t])))
    elif x.ndim == 3:
        y = x
    else:
        raise NotImplementedError

    tmp = np.c_[np.zeros((y.shape[-1], y.shape[1], 1)), y.T].T
    evaluated = np.reshape(np.exp(y - logsumexp(tmp, axis=0)), x.shape)

    return evaluated


def _partials_logistic(x):
    """
    Note that this is not a vectorized function
    """
    tmp = _logistic(x)

    # k
    if tmp.ndim == 0:
        return tmp - tmp**2
    # k x k
    elif tmp.ndim == 1:
        partials = np.diag(tmp - tmp**2)
    # k x k x t
    elif tmp.ndim == 2:
        partials = [np.diag(tmp[:, t] - tmp[:, t]**2)
                    for t in range(tmp.shape[1])]
        shape = tmp.shape[1], tmp.shape[0], tmp.shape[0]
        partials = np.concatenate(partials).reshape(shape).transpose((1, 2, 0))
    # k x k x j x t
    else:
        partials = [[np.diag(tmp[:, j, t] - tmp[:, j, t]**2)
                     for t in range(tmp.shape[2])]
                    for j in range(tmp.shape[1])]
        shape = tmp.shape[1], tmp.shape[2], tmp.shape[0], tmp.shape[0]
        partials = np.concatenate(partials).reshape(shape).transpose(
            (2, 3, 0, 1))

    for i in range(tmp.shape[0]):
        for j in range(i):
            partials[i, j, ...] = -tmp[i, ...] * tmp[j, ...]
            partials[j, i, ...] = partials[i, j, ...]
    return partials


def py_hamilton_filter(initial_probabilities, regime_transition,
                       conditional_likelihoods, model_order):
    """
    Hamilton filter using pure Python

    Parameters
    ----------
    initial_probabilities : array
        Array of initial probabilities, shaped (k_regimes,) giving the
        distribution of the regime process at time t = -order where order
        is a nonnegative integer.
    regime_transition : array
        Matrix of regime transition probabilities, shaped either
        (k_regimes, k_regimes, 1) or if there are time-varying transition
        probabilities (k_regimes, k_regimes, nobs + order).  Entry [i, j,
        t] contains the probability of moving from j at time t-1 to i at
        time t, so each matrix regime_transition[:, :, t] should be left
        stochastic.  The first order entries and initial_probabilities are
        used to produce the initial joint distribution of dimension order +
        1 at time t=0.
    conditional_likelihoods : array
        Array of likelihoods conditional on the last `order+1` regimes,
        shaped (k_regimes,)*(order + 1) + (nobs,).

    Returns
    -------
    filtered_marginal_probabilities : array
        Array containing Pr[S_t=s_t | Y_t] - the probability of being in each
        regime conditional on time t information. Shaped (k_regimes, nobs).
    predicted_joint_probabilities : array
        Array containing Pr[S_t=s_t, ..., S_{t-order}=s_{t-order} | Y_{t-1}] -
        the joint probability of the current and previous `order` periods
        being in each combination of regimes conditional on time t-1
        information. Shaped (k_regimes,) * (order + 1) + (nobs,).
    joint_likelihoods : array
        Array of likelihoods condition on time t information, shaped (nobs,).
    filtered_joint_probabilities : array
        Array containing Pr[S_t=s_t, ..., S_{t-order}=s_{t-order} | Y_{t}] -
        the joint probability of the current and previous `order` periods
        being in each combination of regimes conditional on time t
        information. Shaped (k_regimes,) * (order + 1) + (nobs,).

    """

    # Dimensions
    k_regimes = len(initial_probabilities)
    nobs = conditional_likelihoods.shape[-1]
    order = conditional_likelihoods.ndim - 2
    dtype = conditional_likelihoods.dtype

    # Check for compatible shapes.
    incompatible_shapes = (
        regime_transition.shape[-1] not in (1, nobs + model_order)
        or regime_transition.shape[:2] != (k_regimes, k_regimes)
        or conditional_likelihoods.shape[0] != k_regimes)
    if incompatible_shapes:
        raise ValueError('Arguments do not have compatible shapes')

    # Storage
    # Pr[S_t = s_t | Y_t]
    filtered_marginal_probabilities = (
        np.zeros((k_regimes, nobs), dtype=dtype))
    # Pr[S_t = s_t, ... S_{t-r} = s_{t-r} | Y_{t-1}]
    predicted_joint_probabilities = np.zeros(
        (k_regimes,) * (order + 1) + (nobs,), dtype=dtype)
    # f(y_t | Y_{t-1})
    joint_likelihoods = np.zeros((nobs,), dtype)
    # Pr[S_t = s_t, ... S_{t-r} = s_{t-r} | Y_t]
    filtered_joint_probabilities = np.zeros(
        (k_regimes,) * (order + 1) + (nobs + 1,), dtype=dtype)

    # Initial probabilities
    filtered_marginal_probabilities[:, 0] = initial_probabilities
    tmp = np.copy(initial_probabilities)
    shape = (k_regimes, k_regimes)
    for i in range(order):
        tmp = np.reshape(regime_transition[..., i], shape + (1,) * i) * tmp
    filtered_joint_probabilities[..., 0] = tmp

    # Check that regime_transition is oriented correctly.
    if not np.allclose(np.sum(regime_transition, axis=0), 1):
        raise ValueError('regime_transition does not contain '
                         'left stochastic matrices.')

    # Reshape regime_transition so we can use broadcasting
    shape = (k_regimes, k_regimes)
    shape += (1,) * (order-1)
    shape += (regime_transition.shape[-1],)
    regime_transition = np.reshape(regime_transition, shape)

    # Get appropriate subset of transition matrix
    if regime_transition.shape[-1] > 1:
        regime_transition = regime_transition[..., model_order:]

    # Hamilton filter iterations
    transition_t = 0
    for t in range(nobs):
        if regime_transition.shape[-1] > 1:
            transition_t = t

        # S_t, S_{t-1}, ..., S_{t-r} | t-1, stored at zero-indexed location t
        if order > 0:
            predicted_joint_probabilities[..., t] = (
                # S_t | S_{t-1}
                regime_transition[..., transition_t] *
                # S_{t-1}, S_{t-2}, ..., S_{t-r} | t-1
                filtered_joint_probabilities[..., t].sum(axis=-1))
        else:
            predicted_joint_probabilities[..., t] = (
                np.dot(regime_transition[..., transition_t],
                       filtered_joint_probabilities[..., t]))

        # f(y_t, S_t, ..., S_{t-r} | t-1)
        tmp = (conditional_likelihoods[..., t] *
               predicted_joint_probabilities[..., t])
        # f(y_t | t-1)
        joint_likelihoods[t] = np.sum(tmp)

        # S_t, S_{t-1}, ..., S_{t-r} | t, stored at index t+1
        filtered_joint_probabilities[..., t+1] = (
            tmp / joint_likelihoods[t])

    # S_t | t
    filtered_marginal_probabilities = filtered_joint_probabilities[..., 1:]
    for i in range(1, filtered_marginal_probabilities.ndim - 1):
        filtered_marginal_probabilities = np.sum(
            filtered_marginal_probabilities, axis=-2)

    return (filtered_marginal_probabilities, predicted_joint_probabilities,
            joint_likelihoods, filtered_joint_probabilities[..., 1:])


def cy_hamilton_filter(initial_probabilities, regime_transition,
                       conditional_likelihoods, model_order):
    """
    Hamilton filter using Cython inner loop

    Parameters
    ----------
    initial_probabilities : array
        Array of initial probabilities, shaped (k_regimes,) giving the
        distribution of the regime process at time t = -order where order
        is a nonnegative integer.
    regime_transition : array
        Matrix of regime transition probabilities, shaped either
        (k_regimes, k_regimes, 1) or if there are time-varying transition
        probabilities (k_regimes, k_regimes, nobs + order).  Entry [i, j,
        t] contains the probability of moving from j at time t-1 to i at
        time t, so each matrix regime_transition[:, :, t] should be left
        stochastic.  The first order entries and initial_probabilities are
        used to produce the initial joint distribution of dimension order +
        1 at time t=0.
    conditional_likelihoods : array
        Array of likelihoods conditional on the last `order+1` regimes,
        shaped (k_regimes,)*(order + 1) + (nobs,).

    Returns
    -------
    filtered_marginal_probabilities : array
        Array containing Pr[S_t=s_t | Y_t] - the probability of being in each
        regime conditional on time t information. Shaped (k_regimes, nobs).
    predicted_joint_probabilities : array
        Array containing Pr[S_t=s_t, ..., S_{t-order}=s_{t-order} | Y_{t-1}] -
        the joint probability of the current and previous `order` periods
        being in each combination of regimes conditional on time t-1
        information. Shaped (k_regimes,) * (order + 1) + (nobs,).
    joint_likelihoods : array
        Array of likelihoods condition on time t information, shaped (nobs,).
    filtered_joint_probabilities : array
        Array containing Pr[S_t=s_t, ..., S_{t-order}=s_{t-order} | Y_{t}] -
        the joint probability of the current and previous `order` periods
        being in each combination of regimes conditional on time t
        information. Shaped (k_regimes,) * (order + 1) + (nobs,).
    """

    # Dimensions
    k_regimes = len(initial_probabilities)
    nobs = conditional_likelihoods.shape[-1]
    order = conditional_likelihoods.ndim - 2
    dtype = conditional_likelihoods.dtype

    # Check for compatible shapes.
    incompatible_shapes = (
        regime_transition.shape[-1] not in (1, nobs + model_order)
        or regime_transition.shape[:2] != (k_regimes, k_regimes)
        or conditional_likelihoods.shape[0] != k_regimes)
    if incompatible_shapes:
        raise ValueError('Arguments do not have compatible shapes')

    # Storage
    # Pr[S_t = s_t | Y_t]
    filtered_marginal_probabilities = (
        np.zeros((k_regimes, nobs), dtype=dtype))
    # Pr[S_t = s_t, ... S_{t-r} = s_{t-r} | Y_{t-1}]
    # Has k_regimes^(order+1) elements
    predicted_joint_probabilities = np.zeros(
        (k_regimes,) * (order + 1) + (nobs,), dtype=dtype)
    # f(y_t | Y_{t-1})
    joint_likelihoods = np.zeros((nobs,), dtype)
    # Pr[S_t = s_t, ... S_{t-r+1} = s_{t-r+1} | Y_t]
    # Has k_regimes^order elements
    filtered_joint_probabilities = np.zeros(
        (k_regimes,) * (order + 1) + (nobs + 1,), dtype=dtype)

    # Initial probabilities
    filtered_marginal_probabilities[:, 0] = initial_probabilities
    tmp = np.copy(initial_probabilities)
    shape = (k_regimes, k_regimes)
    transition_t = 0
    for i in range(order):
        if regime_transition.shape[-1] > 1:
            transition_t = i
        tmp = np.reshape(regime_transition[..., transition_t],
                         shape + (1,) * i) * tmp
    filtered_joint_probabilities[..., 0] = tmp

    # Get appropriate subset of transition matrix
    if regime_transition.shape[-1] > 1:
        regime_transition = regime_transition[..., model_order:]

    # Run Cython filter iterations
    prefix, dtype, _ = find_best_blas_type((
        regime_transition, conditional_likelihoods, joint_likelihoods,
        predicted_joint_probabilities, filtered_joint_probabilities))
    func = prefix_hamilton_filter_map[prefix]
    func(nobs, k_regimes, order, regime_transition,
         conditional_likelihoods.reshape(k_regimes**(order+1), nobs),
         joint_likelihoods,
         predicted_joint_probabilities.reshape(k_regimes**(order+1), nobs),
         filtered_joint_probabilities.reshape(k_regimes**(order+1), nobs+1))

    # S_t | t
    filtered_marginal_probabilities = filtered_joint_probabilities[..., 1:]
    for i in range(1, filtered_marginal_probabilities.ndim - 1):
        filtered_marginal_probabilities = np.sum(
            filtered_marginal_probabilities, axis=-2)

    return (filtered_marginal_probabilities, predicted_joint_probabilities,
            joint_likelihoods, filtered_joint_probabilities[..., 1:])


def py_kim_smoother(regime_transition, predicted_joint_probabilities,
                    filtered_joint_probabilities):
    """
    Kim smoother using pure Python

    Parameters
    ----------
    regime_transition : array
        Matrix of regime transition probabilities, shaped either
        (k_regimes, k_regimes, 1) or if there are time-varying transition
        probabilities (k_regimes, k_regimes, nobs).
    predicted_joint_probabilities : array
        Array containing Pr[S_t=s_t, ..., S_{t-order}=s_{t-order} | Y_{t-1}] -
        the joint probability of the current and previous `order` periods
        being in each combination of regimes conditional on time t-1
        information. Shaped (k_regimes,) * (order + 1) + (nobs,).
    filtered_joint_probabilities : array
        Array containing Pr[S_t=s_t, ..., S_{t-order}=s_{t-order} | Y_{t}] -
        the joint probability of the current and previous `order` periods
        being in each combination of regimes conditional on time t
        information. Shaped (k_regimes,) * (order + 1) + (nobs,).

    Returns
    -------
    smoothed_joint_probabilities : array
        Array containing Pr[S_t=s_t, ..., S_{t-order}=s_{t-order} | Y_T] -
        the joint probability of the current and previous `order` periods
        being in each combination of regimes conditional on all information.
        Shaped (k_regimes,) * (order + 1) + (nobs,).
    smoothed_marginal_probabilities : array
        Array containing Pr[S_t=s_t | Y_T] - the probability of being in each
        regime conditional on all information. Shaped (k_regimes, nobs).
    """

    # Dimensions
    k_regimes = filtered_joint_probabilities.shape[0]
    nobs = filtered_joint_probabilities.shape[-1]
    order = filtered_joint_probabilities.ndim - 2
    dtype = filtered_joint_probabilities.dtype

    # Storage
    smoothed_joint_probabilities = np.zeros(
        (k_regimes,) * (order + 1) + (nobs,), dtype=dtype)
    smoothed_marginal_probabilities = np.zeros((k_regimes, nobs), dtype=dtype)

    # S_T, S_{T-1}, ..., S_{T-r} | T
    smoothed_joint_probabilities[..., -1] = (
        filtered_joint_probabilities[..., -1])

    # Reshape transition so we can use broadcasting
    shape = (k_regimes, k_regimes)
    shape += (1,) * (order)
    shape += (regime_transition.shape[-1],)
    regime_transition = np.reshape(regime_transition, shape)

    # Get appropriate subset of transition matrix
    if regime_transition.shape[-1] == nobs + order:
        regime_transition = regime_transition[..., order:]

    # Kim smoother iterations
    transition_t = 0
    for t in range(nobs - 2, -1, -1):
        if regime_transition.shape[-1] > 1:
            transition_t = t + 1

        # S_{t+1}, S_t, ..., S_{t-r+1} | t
        # x = predicted_joint_probabilities[..., t]
        x = (filtered_joint_probabilities[..., t] *
             regime_transition[..., transition_t])
        # S_{t+1}, S_t, ..., S_{t-r+2} | T / S_{t+1}, S_t, ..., S_{t-r+2} | t
        y = (smoothed_joint_probabilities[..., t+1] /
             predicted_joint_probabilities[..., t+1])
        # S_t, S_{t-1}, ..., S_{t-r+1} | T
        smoothed_joint_probabilities[..., t] = (x * y[..., None]).sum(axis=0)

    # Get smoothed marginal probabilities S_t | T by integrating out
    # S_{t-k+1}, S_{t-k+2}, ..., S_{t-1}
    smoothed_marginal_probabilities = smoothed_joint_probabilities
    for i in range(1, smoothed_marginal_probabilities.ndim - 1):
        smoothed_marginal_probabilities = np.sum(
            smoothed_marginal_probabilities, axis=-2)

    return smoothed_joint_probabilities, smoothed_marginal_probabilities


def cy_kim_smoother(regime_transition, predicted_joint_probabilities,
                    filtered_joint_probabilities):
    """
    Kim smoother using Cython inner loop

    Parameters
    ----------
    regime_transition : array
        Matrix of regime transition probabilities, shaped either
        (k_regimes, k_regimes, 1) or if there are time-varying transition
        probabilities (k_regimes, k_regimes, nobs).
    predicted_joint_probabilities : array
        Array containing Pr[S_t=s_t, ..., S_{t-order}=s_{t-order} | Y_{t-1}] -
        the joint probability of the current and previous `order` periods
        being in each combination of regimes conditional on time t-1
        information. Shaped (k_regimes,) * (order + 1) + (nobs,).
    filtered_joint_probabilities : array
        Array containing Pr[S_t=s_t, ..., S_{t-order}=s_{t-order} | Y_{t}] -
        the joint probability of the current and previous `order` periods
        being in each combination of regimes conditional on time t
        information. Shaped (k_regimes,) * (order + 1) + (nobs,).

    Returns
    -------
    smoothed_joint_probabilities : array
        Array containing Pr[S_t=s_t, ..., S_{t-order}=s_{t-order} | Y_T] -
        the joint probability of the current and previous `order` periods
        being in each combination of regimes conditional on all information.
        Shaped (k_regimes,) * (order + 1) + (nobs,).
    smoothed_marginal_probabilities : array
        Array containing Pr[S_t=s_t | Y_T] - the probability of being in each
        regime conditional on all information. Shaped (k_regimes, nobs).
    """

    # Dimensions
    k_regimes = filtered_joint_probabilities.shape[0]
    nobs = filtered_joint_probabilities.shape[-1]
    order = filtered_joint_probabilities.ndim - 2
    dtype = filtered_joint_probabilities.dtype

    # Storage
    smoothed_joint_probabilities = np.zeros(
        (k_regimes,) * (order + 1) + (nobs,), dtype=dtype)

    # Get appropriate subset of transition matrix
    if regime_transition.shape[-1] == nobs + order:
        regime_transition = regime_transition[..., order:]

    # Run Cython smoother iterations
    prefix, dtype, _ = find_best_blas_type((
        regime_transition, predicted_joint_probabilities,
        filtered_joint_probabilities))
    func = prefix_kim_smoother_map[prefix]
    func(nobs, k_regimes, order, regime_transition,
         predicted_joint_probabilities.reshape(k_regimes**(order+1), nobs),
         filtered_joint_probabilities.reshape(k_regimes**(order+1), nobs),
         smoothed_joint_probabilities.reshape(k_regimes**(order+1), nobs))

    # Get smoothed marginal probabilities S_t | T by integrating out
    # S_{t-k+1}, S_{t-k+2}, ..., S_{t-1}
    smoothed_marginal_probabilities = smoothed_joint_probabilities
    for i in range(1, smoothed_marginal_probabilities.ndim - 1):
        smoothed_marginal_probabilities = np.sum(
            smoothed_marginal_probabilities, axis=-2)

    return smoothed_joint_probabilities, smoothed_marginal_probabilities


class MarkovSwitchingParams(object):
    """
    Class to hold parameters in Markov switching models

    Parameters
    ----------
    k_regimes : int
        The number of regimes between which parameters may switch.

    Notes
    -----

    The purpose is to allow selecting parameter indexes / slices based on
    parameter type, regime number, or both.

    Parameters are lexicographically ordered in the following way:

    1. Named type string (e.g. "autoregressive")
    2. Number (e.g. the first autoregressive parameter, then the second)
    3. Regime (if applicable)

    Parameter blocks are set using dictionary setter notation where the key
    is the named type string and the value is a list of boolean values
    indicating whether a given parameter is switching or not.

    For example, consider the following code:

        parameters = MarkovSwitchingParams(k_regimes=2)
        parameters['regime_transition'] = [1,1]
        parameters['exog'] = [0, 1]

    This implies the model has 7 parameters: 4 "regime_transition"-related
    parameters (2 parameters that each switch according to regimes) and 3
    "exog"-related parameters (1 parameter that does not switch, and one 1 that
    does).

    The order of parameters is then:

    1. The first "regime_transition" parameter, regime 0
    2. The first "regime_transition" parameter, regime 1
    3. The second "regime_transition" parameter, regime 1
    4. The second "regime_transition" parameter, regime 1
    5. The first "exog" parameter
    6. The second "exog" parameter, regime 0
    7. The second "exog" parameter, regime 1

    Retrieving indexes / slices is done through dictionary getter notation.
    There are three options for the dictionary key:

    - Regime number (zero-indexed)
    - Named type string (e.g. "autoregressive")
    - Regime number and named type string

    In the above example, consider the following getters:

    >>> parameters[0]
    array([0, 2, 4, 6])
    >>> parameters[1]
    array([1, 3, 5, 6])
    >>> parameters['exog']
    slice(4, 7, None)
    >>> parameters[0, 'exog']
    [4, 6]
    >>> parameters[1, 'exog']
    [4, 7]

    Notice that in the last two examples, both lists of indexes include 4.
    That's because that is the index of the the non-switching first "exog"
    parameter, which should be selected regardless of the regime.

    In addition to the getter, the `k_parameters` attribute is an OrderedDict
    with the named type strings as the keys. It can be used to get the total
    number of parameters of each type:

    >>> parameters.k_parameters['regime_transition']
    4
    >>> parameters.k_parameters['exog']
    3

    """
    def __init__(self, k_regimes):
        self.k_regimes = k_regimes

        self.k_params = 0
        self.k_parameters = OrderedDict()
        self.switching = OrderedDict()
        self.slices_purpose = OrderedDict()
        self.relative_index_regime_purpose = [
            OrderedDict() for i in range(self.k_regimes)]
        self.index_regime_purpose = [
            OrderedDict() for i in range(self.k_regimes)]
        self.index_regime = [[] for i in range(self.k_regimes)]

    def __getitem__(self, key):
        _type = type(key)

        # Get a slice for a block of parameters by purpose
        if _type is str:
            return self.slices_purpose[key]
        # Get a slice for a block of parameters by regime
        elif _type is int:
            return self.index_regime[key]
        elif _type is tuple:
            if not len(key) == 2:
                raise IndexError('Invalid index')
            if type(key[1]) == str and type(key[0]) == int:
                return self.index_regime_purpose[key[0]][key[1]]
            elif type(key[0]) == str and type(key[1]) == int:
                return self.index_regime_purpose[key[1]][key[0]]
            else:
                raise IndexError('Invalid index')
        else:
            raise IndexError('Invalid index')

    def __setitem__(self, key, value):
        _type = type(key)

        if _type is str:
            value = np.array(value, dtype=bool, ndmin=1)
            k_params = self.k_params
            self.k_parameters[key] = (
                value.size + np.sum(value) * (self.k_regimes - 1))
            self.k_params += self.k_parameters[key]
            self.switching[key] = value
            self.slices_purpose[key] = np.s_[k_params:self.k_params]

            for j in range(self.k_regimes):
                self.relative_index_regime_purpose[j][key] = []
                self.index_regime_purpose[j][key] = []

            offset = 0
            for i in range(value.size):
                switching = value[i]
                for j in range(self.k_regimes):
                    # Non-switching parameters
                    if not switching:
                        self.relative_index_regime_purpose[j][key].append(
                            offset)
                    # Switching parameters
                    else:
                        self.relative_index_regime_purpose[j][key].append(
                            offset + j)
                offset += 1 if not switching else self.k_regimes

            for j in range(self.k_regimes):
                offset = 0
                indices = []
                for k, v in self.relative_index_regime_purpose[j].items():
                    v = (np.r_[v] + offset).tolist()
                    self.index_regime_purpose[j][k] = v
                    indices.append(v)
                    offset += self.k_parameters[k]
                self.index_regime[j] = np.concatenate(indices).astype(int)
        else:
            raise IndexError('Invalid index')


class MarkovSwitching(tsbase.TimeSeriesModel):
    """
    First-order k-regime Markov switching model

    Parameters
    ----------
    endog : array_like
        The endogenous variable.
    k_regimes : integer
        The number of regimes.
    order : integer, optional
        The order of the model describes the dependence of the likelihood on
        previous regimes. This depends on the model in question and should be
        set appropriately by subclasses.
    exog_tvtp : array_like, optional
        Array of exogenous or lagged variables to use in calculating
        time-varying transition probabilities (TVTP). TVTP is only used if this
        variable is provided. If an intercept is desired, a column of ones must
        be explicitly included in this array.

    Notes
    -----
    This model is new and API stability is not guaranteed, although changes
    will be made in a backwards compatible way if possible.

    References
    ----------
    Kim, Chang-Jin, and Charles R. Nelson. 1999.
    "State-Space Models with Regime Switching:
    Classical and Gibbs-Sampling Approaches with Applications".
    MIT Press Books. The MIT Press.

    """

    def __init__(self, endog, k_regimes, order=0, exog_tvtp=None, exog=None,
                 dates=None, freq=None, missing='none'):

        # Properties
        self.k_regimes = k_regimes
        self.tvtp = exog_tvtp is not None
        # The order of the model may be overridden in subclasses
        self.order = order

        # Exogenous data
        # TODO add checks for exog_tvtp consistent shape and indices
        self.k_tvtp, self.exog_tvtp = prepare_exog(exog_tvtp)

        # Initialize the base model
        super(MarkovSwitching, self).__init__(endog, exog, dates=dates,
                                              freq=freq, missing=missing)

        # Dimensions
        self.nobs = self.endog.shape[0]

        # Sanity checks
        if self.endog.ndim > 1 and self.endog.shape[1] > 1:
            raise ValueError('Must have univariate endogenous data.')
        if self.k_regimes < 2:
            raise ValueError('Markov switching models must have at least two'
                             ' regimes.')
        if not(self.exog_tvtp is None or self.exog_tvtp.shape[0] == self.nobs):
            raise ValueError('Time-varying transition probabilities exogenous'
                             ' array must have the same number of observations'
                             ' as the endogenous array.')

        # Parameters
        self.parameters = MarkovSwitchingParams(self.k_regimes)
        k_transition = self.k_regimes - 1
        if self.tvtp:
            k_transition *= self.k_tvtp
        self.parameters['regime_transition'] = [1] * k_transition

        # Internal model properties: default is steady-state initialization
        self._initialization = 'steady-state'
        self._initial_probabilities = None

    @property
    def k_params(self):
        """
        (int) Number of parameters in the model
        """
        return self.parameters.k_params

    def initialize_steady_state(self):
        """
        Set initialization of regime probabilities to be steady-state values

        Notes
        -----
        Only valid if there are not time-varying transition probabilities.

        """
        if self.tvtp:
            raise ValueError('Cannot use steady-state initialization when'
                             ' the regime transition matrix is time-varying.')

        self._initialization = 'steady-state'
        self._initial_probabilities = None

    def initialize_known(self, probabilities, tol=1e-8):
        """
        Set initialization of regime probabilities to use known values
        """
        self._initialization = 'known'
        probabilities = np.array(probabilities, ndmin=1)
        if not probabilities.shape == (self.k_regimes,):
            raise ValueError('Initial probabilities must be a vector of shape'
                             ' (k_regimes,).')
        if not np.abs(np.sum(probabilities) - 1) < tol:
            raise ValueError('Initial probabilities vector must sum to one.')
        self._initial_probabilities = probabilities

    def initial_probabilities(self, params, regime_transition=None):
        """
        Retrieve initial probabilities
        """
        params = np.array(params, ndmin=1)
        if self._initialization == 'steady-state':
            if regime_transition is None:
                regime_transition = self.regime_transition_matrix(params)
            if regime_transition.ndim == 3:
                regime_transition = regime_transition[..., 0]
            m = regime_transition.shape[0]
            A = np.c_[(np.eye(m) - regime_transition).T, np.ones(m)].T
            try:
                probabilities = np.linalg.pinv(A)[:, -1]
            except np.linalg.LinAlgError:
                raise RuntimeError('Steady-state probabilities could not be'
                                   ' constructed.')
        elif self._initialization == 'known':
            probabilities = self._initial_probabilities
        else:
            raise RuntimeError('Invalid initialization method selected.')

        return probabilities

    def _regime_transition_matrix_tvtp(self, params, exog_tvtp=None):
        if exog_tvtp is None:
            exog_tvtp = self.exog_tvtp
        nobs = len(exog_tvtp)

        regime_transition_matrix = np.zeros(
            (self.k_regimes, self.k_regimes, nobs),
            dtype=np.promote_types(np.float64, params.dtype))

        # Compute the predicted values from the regression
        for i in range(self.k_regimes):
            coeffs = params[self.parameters[i, 'regime_transition']]
            regime_transition_matrix[:-1, i, :] = np.dot(
                exog_tvtp,
                np.reshape(coeffs, (self.k_regimes-1, self.k_tvtp)).T).T

        # Perform the logistic transformation
        tmp = np.c_[np.zeros((nobs, self.k_regimes, 1)),
                    regime_transition_matrix[:-1, :, :].T].T
        regime_transition_matrix[:-1, :, :] = np.exp(
            regime_transition_matrix[:-1, :, :] - logsumexp(tmp, axis=0))

        # Compute the last column of the transition matrix
        regime_transition_matrix[-1, :, :] = (
            1 - np.sum(regime_transition_matrix[:-1, :, :], axis=0))

        return regime_transition_matrix

    def regime_transition_matrix(self, params, exog_tvtp=None):
        """
        Construct the left-stochastic transition matrix

        Notes
        -----
        This matrix will either be shaped (k_regimes, k_regimes, 1) or if there
        are time-varying transition probabilities, it will be shaped
        (k_regimes, k_regimes, nobs).

        The (i,j)th element of this matrix is the probability of transitioning
        from regime j to regime i; thus the previous regime is represented in a
        column and the next regime is represented by a row.

        It is left-stochastic, meaning that each column sums to one (because
        it is certain that from one regime (j) you will transition to *some
        other regime*).

        """
        params = np.array(params, ndmin=1)
        if not self.tvtp:
            regime_transition_matrix = np.zeros(
                (self.k_regimes, self.k_regimes, 1),
                dtype=np.promote_types(np.float64, params.dtype))
            regime_transition_matrix[:-1, :, 0] = np.reshape(
                params[self.parameters['regime_transition']],
                (self.k_regimes-1, self.k_regimes))
            regime_transition_matrix[-1, :, 0] = (
                1 - np.sum(regime_transition_matrix[:-1, :, 0], axis=0))
        else:
            regime_transition_matrix = (
                self._regime_transition_matrix_tvtp(params, exog_tvtp))

        return regime_transition_matrix

    def predict(self, params, start=None, end=None, probabilities=None,
                conditional=False):
        """
        In-sample prediction and out-of-sample forecasting

        Parameters
        ----------
        params : array
            Parameters at which to form predictions
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast is end. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out of sample prediction. Default is the last observation in
            the sample.
        probabilities : string or array_like, optional
            Specifies the weighting probabilities used in constructing the
            prediction as a weighted average. If a string, can be 'predicted',
            'filtered', or 'smoothed'. Otherwise can be an array of
            probabilities to use. Default is smoothed.
        conditional: boolean or int, optional
            Whether or not to return predictions conditional on current or
            past regimes. If False, returns a single vector of weighted
            predictions. If True or 1, returns predictions conditional on the
            current regime. For larger integers, returns predictions
            conditional on the current regime and some number of past regimes.

        Returns
        -------
        predict : array
            Array of out of in-sample predictions and / or out-of-sample
            forecasts.
        """
        if start is None:
            start = self._index[0]

        # Handle start, end
        start, end, out_of_sample, prediction_index = (
            self._get_prediction_index(start, end))

        if out_of_sample > 0:
            raise NotImplementedError

        # Perform in-sample prediction
        predict = self.predict_conditional(params)
        squeezed = np.squeeze(predict)

        # Check if we need to do weighted averaging
        if squeezed.ndim - 1 > conditional:
            # Determine in-sample weighting probabilities
            if probabilities is None or probabilities == 'smoothed':
                results = self.smooth(params, return_raw=True)
                probabilities = results.smoothed_joint_probabilities
            elif probabilities == 'filtered':
                results = self.filter(params, return_raw=True)
                probabilities = results.filtered_joint_probabilities
            elif probabilities == 'predicted':
                results = self.filter(params, return_raw=True)
                probabilities = results.predicted_joint_probabilities

            # Compute weighted average
            predict = (predict * probabilities)
            for i in range(predict.ndim - 1 - int(conditional)):
                predict = np.sum(predict, axis=-2)
        else:
            predict = squeezed

        return predict[start:end + out_of_sample + 1]

    def predict_conditional(self, params):
        """
        In-sample prediction, conditional on the current, and possibly past,
        regimes

        Parameters
        ----------
        params : array_like
            Array of parameters at which to perform prediction.

        Returns
        -------
        predict : array_like
            Array of predictions conditional on current, and possibly past,
            regimes
        """
        raise NotImplementedError

    def _conditional_likelihoods(self, params):
        """
        Compute likelihoods conditional on the current period's regime (and
        the last self.order periods' regimes if self.order > 0).

        Must be implemented in subclasses.
        """
        raise NotImplementedError

    def _filter(self, params, regime_transition=None):
        # Get the regime transition matrix if not provided
        if regime_transition is None:
            regime_transition = self.regime_transition_matrix(params)
        # Get the initial probabilities
        initial_probabilities = self.initial_probabilities(
            params, regime_transition)

        # Compute the conditional likelihoods
        conditional_likelihoods = self._conditional_likelihoods(params)

        # Apply the filter
        return ((regime_transition, initial_probabilities,
                 conditional_likelihoods) +
                cy_hamilton_filter(initial_probabilities, regime_transition,
                                   conditional_likelihoods, self.order))

    def filter(self, params, transformed=True, cov_type=None, cov_kwds=None,
               return_raw=False, results_class=None,
               results_wrapper_class=None):
        """
        Apply the Hamilton filter

        Parameters
        ----------
        params : array_like
            Array of parameters at which to perform filtering.
        transformed : boolean, optional
            Whether or not `params` is already transformed. Default is True.
        cov_type : str, optional
            See `fit` for a description of covariance matrix types
            for results object.
        cov_kwds : dict or None, optional
            See `fit` for a description of required keywords for alternative
            covariance estimators
        return_raw : boolean,optional
            Whether or not to return only the raw Hamilton filter output or a
            full results object. Default is to return a full results object.
        results_class : type, optional
            A results class to instantiate rather than
            `MarkovSwitchingResults`. Usually only used internally by
            subclasses.
        results_wrapper_class : type, optional
            A results wrapper class to instantiate rather than
            `MarkovSwitchingResults`. Usually only used internally by
            subclasses.

        Returns
        -------
        MarkovSwitchingResults
        """
        params = np.array(params, ndmin=1)

        if not transformed:
            params = self.transform_params(params)

        # Save the parameter names
        self.data.param_names = self.param_names

        # Get the result
        names = ['regime_transition', 'initial_probabilities',
                 'conditional_likelihoods', 'filtered_marginal_probabilities',
                 'predicted_joint_probabilities', 'joint_likelihoods',
                 'filtered_joint_probabilities']
        result = HamiltonFilterResults(
            self, Bunch(**dict(zip(names, self._filter(params)))))

        # Wrap in a results object
        return self._wrap_results(params, result, return_raw, cov_type,
                                  cov_kwds, results_class,
                                  results_wrapper_class)

    def _smooth(self, params, filtered_marginal_probabilities,
                predicted_joint_probabilities,
                filtered_joint_probabilities, regime_transition=None):
        # Get the regime transition matrix
        if regime_transition is None:
            regime_transition = self.regime_transition_matrix(params)

        # Apply the smoother
        return cy_kim_smoother(regime_transition,
                               predicted_joint_probabilities,
                               filtered_joint_probabilities)

    @property
    def _res_classes(self):
        return {'fit': (MarkovSwitchingResults, MarkovSwitchingResultsWrapper)}

    def _wrap_results(self, params, result, return_raw, cov_type=None,
                      cov_kwds=None, results_class=None, wrapper_class=None):
        if not return_raw:
            # Wrap in a results object
            result_kwargs = {}
            if cov_type is not None:
                result_kwargs['cov_type'] = cov_type
            if cov_kwds is not None:
                result_kwargs['cov_kwds'] = cov_kwds

            if results_class is None:
                results_class = self._res_classes['fit'][0]
            if wrapper_class is None:
                wrapper_class = self._res_classes['fit'][1]

            res = results_class(self, params, result, **result_kwargs)
            result = wrapper_class(res)
        return result

    def smooth(self, params, transformed=True, cov_type=None, cov_kwds=None,
               return_raw=False, results_class=None,
               results_wrapper_class=None):
        """
        Apply the Kim smoother and Hamilton filter

        Parameters
        ----------
        params : array_like
            Array of parameters at which to perform filtering.
        transformed : boolean, optional
            Whether or not `params` is already transformed. Default is True.
        cov_type : str, optional
            See `fit` for a description of covariance matrix types
            for results object.
        cov_kwds : dict or None, optional
            See `fit` for a description of required keywords for alternative
            covariance estimators
        return_raw : boolean,optional
            Whether or not to return only the raw Hamilton filter output or a
            full results object. Default is to return a full results object.
        results_class : type, optional
            A results class to instantiate rather than
            `MarkovSwitchingResults`. Usually only used internally by
            subclasses.
        results_wrapper_class : type, optional
            A results wrapper class to instantiate rather than
            `MarkovSwitchingResults`. Usually only used internally by
            subclasses.

        Returns
        -------
        MarkovSwitchingResults
        """
        params = np.array(params, ndmin=1)

        if not transformed:
            params = self.transform_params(params)

        # Save the parameter names
        self.data.param_names = self.param_names

        # Hamilton filter
        names = ['regime_transition', 'initial_probabilities',
                 'conditional_likelihoods', 'filtered_marginal_probabilities',
                 'predicted_joint_probabilities', 'joint_likelihoods',
                 'filtered_joint_probabilities']
        result = Bunch(**dict(zip(names, self._filter(params))))

        # Kim smoother
        out = self._smooth(params, result.filtered_marginal_probabilities,
                           result.predicted_joint_probabilities,
                           result.filtered_joint_probabilities)
        result['smoothed_joint_probabilities'] = out[0]
        result['smoothed_marginal_probabilities'] = out[1]
        result = KimSmootherResults(self, result)

        # Wrap in a results object
        return self._wrap_results(params, result, return_raw, cov_type,
                                  cov_kwds, results_class,
                                  results_wrapper_class)

    def loglikeobs(self, params, transformed=True):
        """
        Loglikelihood evaluation for each period

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the loglikelihood
            function.
        transformed : boolean, optional
            Whether or not `params` is already transformed. Default is True.
        """
        params = np.array(params, ndmin=1)

        if not transformed:
            params = self.transform_params(params)

        results = self._filter(params)

        return np.log(results[5])

    def loglike(self, params, transformed=True):
        """
        Loglikelihood evaluation

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the loglikelihood
            function.
        transformed : boolean, optional
            Whether or not `params` is already transformed. Default is True.
        """
        return np.sum(self.loglikeobs(params, transformed))

    def score(self, params, transformed=True):
        """
        Compute the score function at params.

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the score
            function.
        transformed : boolean, optional
            Whether or not `params` is already transformed. Default is True.
        """
        params = np.array(params, ndmin=1)

        return approx_fprime_cs(params, self.loglike, args=(transformed,))

    def score_obs(self, params, transformed=True):
        """
        Compute the score per observation, evaluated at params

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the score
            function.
        transformed : boolean, optional
            Whether or not `params` is already transformed. Default is True.
        """
        params = np.array(params, ndmin=1)

        return approx_fprime_cs(params, self.loglikeobs, args=(transformed,))

    def hessian(self, params, transformed=True):
        """
        Hessian matrix of the likelihood function, evaluated at the given
        parameters

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the Hessian
            function.
        transformed : boolean, optional
            Whether or not `params` is already transformed. Default is True.
        """
        params = np.array(params, ndmin=1)

        return approx_hess_cs(params, self.loglike)

    def fit(self, start_params=None, transformed=True, cov_type='approx',
            cov_kwds=None, method='bfgs', maxiter=100, full_output=1, disp=0,
            callback=None, return_params=False, em_iter=5, search_reps=0,
            search_iter=5, search_scale=1., **kwargs):
        """
        Fits the model by maximum likelihood via Hamilton filter.

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            If None, the default is given by Model.start_params.
        transformed : boolean, optional
            Whether or not `start_params` is already transformed. Default is
            True.
        cov_type : str, optional
            The type of covariance matrix estimator to use. Can be one of
            'approx', 'opg', 'robust', or 'none'. Default is 'approx'.
        cov_kwds : dict or None, optional
            Keywords for alternative covariance estimators
        method : str, optional
            The `method` determines which solver from `scipy.optimize`
            is used, and it can be chosen from among the following strings:

            - 'newton' for Newton-Raphson, 'nm' for Nelder-Mead
            - 'bfgs' for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
            - 'lbfgs' for limited-memory BFGS with optional box constraints
            - 'powell' for modified Powell's method
            - 'cg' for conjugate gradient
            - 'ncg' for Newton-conjugate gradient
            - 'basinhopping' for global basin-hopping solver

            The explicit arguments in `fit` are passed to the solver,
            with the exception of the basin-hopping solver. Each
            solver has several optional arguments that are not the same across
            solvers. See the notes section below (or scipy.optimize) for the
            available arguments and for the list of explicit arguments that the
            basin-hopping solver supports.
        maxiter : int, optional
            The maximum number of iterations to perform.
        full_output : boolean, optional
            Set to True to have all available output in the Results object's
            mle_retvals attribute. The output is dependent on the solver.
            See LikelihoodModelResults notes section for more information.
        disp : boolean, optional
            Set to True to print convergence messages.
        callback : callable callback(xk), optional
            Called after each iteration, as callback(xk), where xk is the
            current parameter vector.
        return_params : boolean, optional
            Whether or not to return only the array of maximizing parameters.
            Default is False.
        em_iter : int, optional
            Number of initial EM iteration steps used to improve starting
            parameters.
        search_reps : int, optional
            Number of randomly drawn search parameters that are drawn around
            `start_params` to try and improve starting parameters. Default is
            0.
        search_iter : int, optional
            Number of initial EM iteration steps used to improve each of the
            search parameter repetitions.
        search_scale : float or array, optional.
            Scale of variates for random start parameter search.
        **kwargs
            Additional keyword arguments to pass to the optimizer.

        Returns
        -------
        MarkovSwitchingResults

        """

        if start_params is None:
            start_params = self.start_params
            transformed = True
        else:
            start_params = np.array(start_params, ndmin=1)

        # Random search for better start parameters
        if search_reps > 0:
            start_params = self._start_params_search(
                search_reps, start_params=start_params,
                transformed=transformed, em_iter=search_iter,
                scale=search_scale)
            transformed = True

        # Get better start params through EM algorithm
        if em_iter and not self.tvtp:
            start_params = self._fit_em(start_params, transformed=transformed,
                                        maxiter=em_iter, tolerance=0,
                                        return_params=True)
            transformed = True

        if transformed:
            start_params = self.untransform_params(start_params)

        # Maximum likelihood estimation by scoring
        fargs = (False,)
        mlefit = super(MarkovSwitching, self).fit(start_params, method=method,
                                                  fargs=fargs,
                                                  maxiter=maxiter,
                                                  full_output=full_output,
                                                  disp=disp, callback=callback,
                                                  skip_hessian=True, **kwargs)

        # Just return the fitted parameters if requested
        if return_params:
            result = self.transform_params(mlefit.params)
        # Otherwise construct the results class if desired
        else:
            result = self.smooth(mlefit.params, transformed=False,
                                 cov_type=cov_type, cov_kwds=cov_kwds)

            result.mlefit = mlefit
            result.mle_retvals = mlefit.mle_retvals
            result.mle_settings = mlefit.mle_settings

        return result

    def _fit_em(self, start_params=None, transformed=True, cov_type='none',
                cov_kwds=None, maxiter=50, tolerance=1e-6, full_output=True,
                return_params=False, **kwargs):
        """
        Fits the model using the Expectation-Maximization (EM) algorithm

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            If None, the default is given by `start_params`.
        transformed : boolean, optional
            Whether or not `start_params` is already transformed. Default is
            True.
        cov_type : str, optional
            The type of covariance matrix estimator to use. Can be one of
            'approx', 'opg', 'robust', or 'none'. Default is 'none'.
        cov_kwds : dict or None, optional
            Keywords for alternative covariance estimators
        maxiter : int, optional
            The maximum number of iterations to perform.
        tolerance : float, optional
            The iteration stops when the difference between subsequent
            loglikelihood values is less than this tolerance.
        full_output : bool, optional
            Set to True to have all available output in the Results object's
            mle_retvals attribute. This includes all intermediate values for
            parameters and loglikelihood values
        return_params : boolean, optional
            Whether or not to return only the array of maximizing parameters.
            Default is False.
        **kwargs
            Additional keyword arguments to pass to the optimizer.

        Notes
        -----
        This is a private method for finding good starting parameters for MLE
        by scoring. It has not been tested for a thoroughly correct EM
        implementation in all cases. It does not support TVTP transition
        probabilities.

        Returns
        -------
        MarkovSwitchingResults
        """

        if start_params is None:
            start_params = self.start_params
            transformed = True
        else:
            start_params = np.array(start_params, ndmin=1)

        if not transformed:
            start_params = self.transform_params(start_params)

        # Perform expectation-maximization
        llf = []
        params = [start_params]
        i = 0
        delta = 0
        while i < maxiter and (i < 2 or (delta > tolerance)):
            out = self._em_iteration(params[-1])
            llf.append(out[0].llf)
            params.append(out[1])
            if i > 0:
                delta = 2 * (llf[-1] - llf[-2]) / np.abs((llf[-1] + llf[-2]))
            i += 1

        # Just return the fitted parameters if requested
        if return_params:
            result = params[-1]
        # Otherwise construct the results class if desired
        else:
            result = self.filter(params[-1], transformed=True,
                                 cov_type=cov_type, cov_kwds=cov_kwds)

            # Save the output
            if full_output:
                em_retvals = Bunch(**{'params': np.array(params),
                                      'llf': np.array(llf),
                                      'iter': i})
                em_settings = Bunch(**{'tolerance': tolerance,
                                       'maxiter': maxiter})
            else:
                em_retvals = None
                em_settings = None

            result.mle_retvals = em_retvals
            result.mle_settings = em_settings

        return result

    def _em_iteration(self, params0):
        """
        EM iteration

        Notes
        -----
        The EM iteration in this base class only performs the EM step for
        non-TVTP transition probabilities.
        """
        params1 = np.zeros(params0.shape,
                           dtype=np.promote_types(np.float64, params0.dtype))

        # Smooth at the given parameters
        result = self.smooth(params0, transformed=True, return_raw=True)

        # The EM with TVTP is not yet supported, just return the previous
        # iteration parameters
        if self.tvtp:
            params1[self.parameters['regime_transition']] = (
                params0[self.parameters['regime_transition']])
        else:
            regime_transition = self._em_regime_transition(result)
            for i in range(self.k_regimes):
                params1[self.parameters[i, 'regime_transition']] = (
                    regime_transition[i])

        return result, params1

    def _em_regime_transition(self, result):
        """
        EM step for regime transition probabilities
        """

        # Marginalize the smoothed joint probabilites to just S_t, S_{t-1} | T
        tmp = result.smoothed_joint_probabilities
        for i in range(tmp.ndim - 3):
            tmp = np.sum(tmp, -2)
        smoothed_joint_probabilities = tmp

        # Transition parameters (recall we're not yet supporting TVTP here)
        k_transition = len(self.parameters[0, 'regime_transition'])
        regime_transition = np.zeros((self.k_regimes, k_transition))
        for i in range(self.k_regimes):  # S_{t_1}
            for j in range(self.k_regimes - 1):  # S_t
                regime_transition[i, j] = (
                    np.sum(smoothed_joint_probabilities[j, i]) /
                    np.sum(result.smoothed_marginal_probabilities[i]))

            # It may be the case that due to rounding error this estimates
            # transition probabilities that sum to greater than one. If so,
            # re-scale the probabilities and warn the user that something
            # is not quite right
            delta = np.sum(regime_transition[i]) - 1
            if delta > 0:
                warnings.warn('Invalid regime transition probabilities'
                              ' estimated in EM iteration; probabilities have'
                              ' been re-scaled to continue estimation.',
                              EstimationWarning)
                regime_transition[i] /= 1 + delta + 1e-6

        return regime_transition

    def _start_params_search(self, reps, start_params=None, transformed=True,
                             em_iter=5, scale=1.):
        """
        Search for starting parameters as random permutations of a vector

        Parameters
        ----------
        reps : int
            Number of random permutations to try.
        start_params : array, optional
            Starting parameter vector. If not given, class-level start
            parameters are used.
        transformed : boolean, optional
            If `start_params` was provided, whether or not those parameters
            are already transformed. Default is True.
        em_iter : int, optional
            Number of EM iterations to apply to each random permutation.
        scale : array or float, optional
            Scale of variates for random start parameter search. Can be given
            as an array of length equal to the number of parameters or as a
            single scalar.

        Notes
        -----
        This is a private method for finding good starting parameters for MLE
        by scoring, where the defaults have been set heuristically.

        """
        if start_params is None:
            start_params = self.start_params
            transformed = True
        else:
            start_params = np.array(start_params, ndmin=1)

        # Random search is over untransformed space
        if transformed:
            start_params = self.untransform_params(start_params)

        # Construct the standard deviations
        scale = np.array(scale, ndmin=1)
        if scale.size == 1:
            scale = np.ones(self.k_params) * scale
        if not scale.size == self.k_params:
            raise ValueError('Scale of variates for random start'
                             ' parameter search must be given for each'
                             ' parameter or as a single scalar.')

        # Construct the random variates
        variates = np.zeros((reps, self.k_params))
        for i in range(self.k_params):
            variates[:, i] = scale[i] * np.random.uniform(-0.5, 0.5, size=reps)

        llf = self.loglike(start_params, transformed=False)
        params = start_params
        for i in range(reps):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                try:
                    proposed_params = self._fit_em(
                        start_params + variates[i], transformed=False,
                        maxiter=em_iter, return_params=True)
                    proposed_llf = self.loglike(proposed_params)

                    if proposed_llf > llf:
                        llf = proposed_llf
                        params = self.untransform_params(proposed_params)
                except Exception:  # FIXME: catch something specific
                    pass

        # Return transformed parameters
        return self.transform_params(params)

    @property
    def start_params(self):
        """
        (array) Starting parameters for maximum likelihood estimation.
        """
        params = np.zeros(self.k_params, dtype=np.float64)

        # Transition probabilities
        if self.tvtp:
            params[self.parameters['regime_transition']] = 0.
        else:
            params[self.parameters['regime_transition']] = 1. / self.k_regimes

        return params

    @property
    def param_names(self):
        """
        (list of str) List of human readable parameter names (for parameters
        actually included in the model).
        """
        param_names = np.zeros(self.k_params, dtype=object)

        # Transition probabilities
        if self.tvtp:
            # TODO add support for exog_tvtp_names
            param_names[self.parameters['regime_transition']] = [
                'p[%d->%d].tvtp%d' % (j, i, k)
                for i in range(self.k_regimes-1)
                for k in range(self.k_tvtp)
                for j in range(self.k_regimes)
                ]
        else:
            param_names[self.parameters['regime_transition']] = [
                'p[%d->%d]' % (j, i)
                for i in range(self.k_regimes-1)
                for j in range(self.k_regimes)]

        return param_names.tolist()

    def transform_params(self, unconstrained):
        """
        Transform unconstrained parameters used by the optimizer to constrained
        parameters used in likelihood evaluation

        Parameters
        ----------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer, to be
            transformed.

        Returns
        -------
        constrained : array_like
            Array of constrained parameters which may be used in likelihood
            evalation.

        Notes
        -----
        In the base class, this only transforms the transition-probability-
        related parameters.
        """
        constrained = np.array(unconstrained, copy=True)
        constrained = constrained.astype(
            np.promote_types(np.float64, constrained.dtype))

        # Nothing to do for transition probabilities if TVTP
        if self.tvtp:
            constrained[self.parameters['regime_transition']] = (
                unconstrained[self.parameters['regime_transition']])
        # Otherwise do logistic transformation
        else:
            # Transition probabilities
            for i in range(self.k_regimes):
                tmp1 = unconstrained[self.parameters[i, 'regime_transition']]
                tmp2 = np.r_[0, tmp1]
                constrained[self.parameters[i, 'regime_transition']] = np.exp(
                    tmp1 - logsumexp(tmp2))

        # Do not do anything for the rest of the parameters

        return constrained

    def _untransform_logistic(self, unconstrained, constrained):
        """
        Function to allow using a numerical root-finder to reverse the
        logistic transform.
        """
        resid = np.zeros(unconstrained.shape, dtype=unconstrained.dtype)
        exp = np.exp(unconstrained)
        sum_exp = np.sum(exp)
        for i in range(len(unconstrained)):
            resid[i] = (unconstrained[i] -
                        np.log(1 + sum_exp - exp[i]) +
                        np.log(1 / constrained[i] - 1))
        return resid

    def untransform_params(self, constrained):
        """
        Transform constrained parameters used in likelihood evaluation
        to unconstrained parameters used by the optimizer

        Parameters
        ----------
        constrained : array_like
            Array of constrained parameters used in likelihood evalution, to be
            transformed.

        Returns
        -------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer.

        Notes
        -----
        In the base class, this only untransforms the transition-probability-
        related parameters.
        """
        unconstrained = np.array(constrained, copy=True)
        unconstrained = unconstrained.astype(
            np.promote_types(np.float64, unconstrained.dtype))

        # Nothing to do for transition probabilities if TVTP
        if self.tvtp:
            unconstrained[self.parameters['regime_transition']] = (
                constrained[self.parameters['regime_transition']])
        # Otherwise reverse logistic transformation
        else:
            for i in range(self.k_regimes):
                s = self.parameters[i, 'regime_transition']
                if self.k_regimes == 2:
                    unconstrained[s] = -np.log(1. / constrained[s] - 1)
                else:
                    from scipy.optimize import root
                    out = root(self._untransform_logistic,
                               np.zeros(unconstrained[s].shape,
                                        unconstrained.dtype),
                               args=(constrained[s],))
                    if not out['success']:
                        raise ValueError('Could not untransform parameters.')
                    unconstrained[s] = out['x']

        # Do not do anything for the rest of the parameters

        return unconstrained


class HamiltonFilterResults(object):
    """
    Results from applying the Hamilton filter to a state space model.

    Parameters
    ----------
    model : Representation
        A Statespace representation

    Attributes
    ----------
    nobs : int
        Number of observations.
    k_endog : int
        The dimension of the observation series.
    k_regimes : int
        The number of unobserved regimes.
    regime_transition : array
        The regime transition matrix.
    initialization : str
        Initialization method for regime probabilities.
    initial_probabilities : array
        Initial regime probabilities
    conditional_likelihoods : array
        The likelihood values at each time period, conditional on regime.
    predicted_joint_probabilities : array
        Predicted joint probabilities at each time period.
    filtered_marginal_probabilities : array
        Filtered marginal probabilities at each time period.
    filtered_joint_probabilities : array
        Filtered joint probabilities at each time period.
    joint_likelihoods : array
        The likelihood values at each time period.
    llf_obs : array
        The loglikelihood values at each time period.
    """
    def __init__(self, model, result):

        self.model = model

        self.nobs = model.nobs
        self.order = model.order
        self.k_regimes = model.k_regimes

        attributes = ['regime_transition', 'initial_probabilities',
                      'conditional_likelihoods',
                      'predicted_joint_probabilities',
                      'filtered_marginal_probabilities',
                      'filtered_joint_probabilities',
                      'joint_likelihoods']
        for name in attributes:
            setattr(self, name, getattr(result, name))

        self.initialization = model._initialization
        self.llf_obs = np.log(self.joint_likelihoods)
        self.llf = np.sum(self.llf_obs)

        # Subset transition if necessary (e.g. for Markov autoregression)
        if self.regime_transition.shape[-1] > 1 and self.order > 0:
            self.regime_transition = self.regime_transition[..., self.order:]

        # Cache for predicted marginal probabilities
        self._predicted_marginal_probabilities = None

    @property
    def predicted_marginal_probabilities(self):
        if self._predicted_marginal_probabilities is None:
            self._predicted_marginal_probabilities = (
                self.predicted_joint_probabilities)
            for i in range(self._predicted_marginal_probabilities.ndim - 2):
                self._predicted_marginal_probabilities = np.sum(
                    self._predicted_marginal_probabilities, axis=-2)
        return self._predicted_marginal_probabilities

    @property
    def expected_durations(self):
        """
        (array) Expected duration of a regime, possibly time-varying.
        """
        return 1. / (1 - np.diagonal(self.regime_transition).squeeze())


class KimSmootherResults(HamiltonFilterResults):
    """
    Results from applying the Kim smoother to a Markov switching model.

    Parameters
    ----------
    model : MarkovSwitchingModel
        The model object.
    result : dict
        A dictionary containing two keys: 'smoothd_joint_probabilities' and
        'smoothed_marginal_probabilities'.

    Attributes
    ----------
    nobs : int
        Number of observations.
    k_endog : int
        The dimension of the observation series.
    k_states : int
        The dimension of the unobserved state process.
    """
    def __init__(self, model, result):
        super(KimSmootherResults, self).__init__(model, result)

        attributes = ['smoothed_joint_probabilities',
                      'smoothed_marginal_probabilities']

        for name in attributes:
            setattr(self, name, getattr(result, name))


class MarkovSwitchingResults(tsbase.TimeSeriesModelResults):
    r"""
    Class to hold results from fitting a Markov switching model

    Parameters
    ----------
    model : MarkovSwitching instance
        The fitted model instance
    params : array
        Fitted parameters
    filter_results : HamiltonFilterResults or KimSmootherResults instance
        The underlying filter and, optionally, smoother output
    cov_type : string
        The type of covariance matrix estimator to use. Can be one of 'approx',
        'opg', 'robust', or 'none'.

    Attributes
    ----------
    model : Model instance
        A reference to the model that was fit.
    filter_results : HamiltonFilterResults or KimSmootherResults instance
        The underlying filter and, optionally, smoother output
    nobs : float
        The number of observations used to fit the model.
    params : array
        The parameters of the model.
    scale : float
        This is currently set to 1.0 and not used by the model or its results.

    """
    use_t = False

    def __init__(self, model, params, results, cov_type='opg', cov_kwds=None,
                 **kwargs):
        self.data = model.data

        tsbase.TimeSeriesModelResults.__init__(self, model, params,
                                               normalized_cov_params=None,
                                               scale=1.)

        # Save the filter / smoother output
        self.filter_results = results
        if isinstance(results, KimSmootherResults):
            self.smoother_results = results
        else:
            self.smoother_results = None

        # Dimensions
        self.nobs = model.nobs
        self.order = model.order
        self.k_regimes = model.k_regimes

        # Setup covariance matrix notes dictionary
        if not hasattr(self, 'cov_kwds'):
            self.cov_kwds = {}
        self.cov_type = cov_type

        # Setup the cache
        self._cache = {}

        # Handle covariance matrix calculation
        if cov_kwds is None:
            cov_kwds = {}
        self._cov_approx_complex_step = (
            cov_kwds.pop('approx_complex_step', True))
        self._cov_approx_centered = cov_kwds.pop('approx_centered', False)
        try:
            self._rank = None
            self._get_robustcov_results(cov_type=cov_type, use_self=True,
                                        **cov_kwds)
        except np.linalg.LinAlgError:
            self._rank = 0
            k_params = len(self.params)
            self.cov_params_default = np.zeros((k_params, k_params)) * np.nan
            self.cov_kwds['cov_type'] = (
                'Covariance matrix could not be calculated: singular.'
                ' information matrix.')

        # Copy over arrays
        attributes = ['regime_transition', 'initial_probabilities',
                      'conditional_likelihoods',
                      'predicted_marginal_probabilities',
                      'predicted_joint_probabilities',
                      'filtered_marginal_probabilities',
                      'filtered_joint_probabilities',
                      'joint_likelihoods', 'expected_durations']
        for name in attributes:
            setattr(self, name, getattr(self.filter_results, name))

        attributes = ['smoothed_joint_probabilities',
                      'smoothed_marginal_probabilities']
        for name in attributes:
            if self.smoother_results is not None:
                setattr(self, name, getattr(self.smoother_results, name))
            else:
                setattr(self, name, None)

        # Reshape some arrays to long-format
        self.predicted_marginal_probabilities = (
            self.predicted_marginal_probabilities.T)
        self.filtered_marginal_probabilities = (
            self.filtered_marginal_probabilities.T)
        if self.smoother_results is not None:
            self.smoothed_marginal_probabilities = (
                self.smoothed_marginal_probabilities.T)

        # Make into Pandas arrays if using Pandas data
        if isinstance(self.data, PandasData):
            index = self.data.row_labels
            if self.expected_durations.ndim > 1:
                self.expected_durations = pd.DataFrame(
                    self.expected_durations, index=index)
            self.predicted_marginal_probabilities = pd.DataFrame(
                self.predicted_marginal_probabilities, index=index)
            self.filtered_marginal_probabilities = pd.DataFrame(
                self.filtered_marginal_probabilities, index=index)
            if self.smoother_results is not None:
                self.smoothed_marginal_probabilities = pd.DataFrame(
                    self.smoothed_marginal_probabilities, index=index)

    def _get_robustcov_results(self, cov_type='opg', **kwargs):
        from statsmodels.base.covtype import descriptions

        use_self = kwargs.pop('use_self', False)
        if use_self:
            res = self
        else:
            raise NotImplementedError
            res = self.__class__(
                self.model, self.params,
                normalized_cov_params=self.normalized_cov_params,
                scale=self.scale)

        # Set the new covariance type
        res.cov_type = cov_type
        res.cov_kwds = {}

        approx_type_str = 'complex-step'

        # Calculate the new covariance matrix
        k_params = len(self.params)
        if k_params == 0:
            res.cov_params_default = np.zeros((0, 0))
            res._rank = 0
            res.cov_kwds['description'] = 'No parameters estimated.'
        elif cov_type == 'custom':
            res.cov_type = kwargs['custom_cov_type']
            res.cov_params_default = kwargs['custom_cov_params']
            res.cov_kwds['description'] = kwargs['custom_description']
            res._rank = np.linalg.matrix_rank(res.cov_params_default)
        elif cov_type == 'none':
            res.cov_params_default = np.zeros((k_params, k_params)) * np.nan
            res._rank = np.nan
            res.cov_kwds['description'] = descriptions['none']
        elif self.cov_type == 'approx':
            res.cov_params_default = res.cov_params_approx
            res.cov_kwds['description'] = descriptions['approx'].format(
                                                approx_type=approx_type_str)
        elif self.cov_type == 'opg':
            res.cov_params_default = res.cov_params_opg
            res.cov_kwds['description'] = descriptions['OPG'].format(
                                                approx_type=approx_type_str)
        elif self.cov_type == 'robust':
            res.cov_params_default = res.cov_params_robust
            res.cov_kwds['description'] = descriptions['robust'].format(
                                                approx_type=approx_type_str)
        else:
            raise NotImplementedError('Invalid covariance matrix type.')

        return res

    @cache_readonly
    def aic(self):
        """
        (float) Akaike Information Criterion
        """
        # return -2*self.llf + 2*self.params.shape[0]
        return aic(self.llf, self.nobs, self.params.shape[0])

    @cache_readonly
    def bic(self):
        """
        (float) Bayes Information Criterion
        """
        # return -2*self.llf + self.params.shape[0]*np.log(self.nobs)
        return bic(self.llf, self.nobs, self.params.shape[0])

    @cache_readonly
    def cov_params_approx(self):
        """
        (array) The variance / covariance matrix. Computed using the numerical
        Hessian approximated by complex step or finite differences methods.
        """
        evaluated_hessian = self.model.hessian(self.params, transformed=True)
        neg_cov, singular_values = pinv_extended(evaluated_hessian)

        if self._rank is None:
            self._rank = np.linalg.matrix_rank(np.diag(singular_values))

        return -neg_cov

    @cache_readonly
    def cov_params_opg(self):
        """
        (array) The variance / covariance matrix. Computed using the outer
        product of gradients method.
        """
        score_obs = self.model.score_obs(self.params, transformed=True).T
        cov_params, singular_values = pinv_extended(
            np.inner(score_obs, score_obs))

        if self._rank is None:
            self._rank = np.linalg.matrix_rank(np.diag(singular_values))

        return cov_params

    @cache_readonly
    def cov_params_robust(self):
        """
        (array) The QMLE variance / covariance matrix. Computed using the
        numerical Hessian as the evaluated hessian.
        """
        cov_opg = self.cov_params_opg
        evaluated_hessian = self.model.hessian(self.params, transformed=True)
        cov_params, singular_values = pinv_extended(
            np.dot(np.dot(evaluated_hessian, cov_opg), evaluated_hessian)
        )

        if self._rank is None:
            self._rank = np.linalg.matrix_rank(np.diag(singular_values))

        return cov_params

    @cache_readonly
    def fittedvalues(self):
        """
        (array) The predicted values of the model. An (nobs x k_endog) array.
        """
        return self.model.predict(self.params)

    @cache_readonly
    def hqic(self):
        """
        (float) Hannan-Quinn Information Criterion
        """
        # return -2*self.llf + 2*np.log(np.log(self.nobs))*self.params.shape[0]
        return hqic(self.llf, self.nobs, self.params.shape[0])

    @cache_readonly
    def llf_obs(self):
        """
        (float) The value of the log-likelihood function evaluated at `params`.
        """
        return self.model.loglikeobs(self.params)

    @cache_readonly
    def llf(self):
        """
        (float) The value of the log-likelihood function evaluated at `params`.
        """
        return self.model.loglike(self.params)

    @cache_readonly
    def resid(self):
        """
        (array) The model residuals. An (nobs x k_endog) array.
        """
        return self.model.endog - self.fittedvalues

    def predict(self, start=None, end=None, probabilities=None,
                conditional=False):
        """
        In-sample prediction and out-of-sample forecasting

        Parameters
        ----------
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast is end. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out of sample prediction. Default is the last observation in
            the sample.
        probabilities : string or array_like, optional
            Specifies the weighting probabilities used in constructing the
            prediction as a weighted average. If a string, can be 'predicted',
            'filtered', or 'smoothed'. Otherwise can be an array of
            probabilities to use. Default is smoothed.
        conditional: boolean or int, optional
            Whether or not to return predictions conditional on current or
            past regimes. If False, returns a single vector of weighted
            predictions. If True or 1, returns predictions conditional on the
            current regime. For larger integers, returns predictions
            conditional on the current regime and some number of past regimes.

        Returns
        -------
        predict : array
            Array of out of in-sample predictions and / or out-of-sample
            forecasts. An (npredict x k_endog) array.
        """
        return self.model.predict(self.params, start=start, end=end,
                                  probabilities=probabilities,
                                  conditional=conditional)

    def forecast(self, steps=1, **kwargs):
        """
        Out-of-sample forecasts

        Parameters
        ----------
        steps : int, str, or datetime, optional
            If an integer, the number of steps to forecast from the end of the
            sample. Can also be a date string to parse or a datetime type.
            However, if the dates index does not have a fixed frequency, steps
            must be an integer. Default
        **kwargs
            Additional arguments may required for forecasting beyond the end
            of the sample. See `FilterResults.predict` for more details.

        Returns
        -------
        forecast : array
            Array of out of sample forecasts. A (steps x k_endog) array.
        """
        raise NotImplementedError

    def summary(self, alpha=.05, start=None, title=None, model_name=None,
                display_params=True):
        """
        Summarize the Model

        Parameters
        ----------
        alpha : float, optional
            Significance level for the confidence intervals. Default is 0.05.
        start : int, optional
            Integer of the start observation. Default is 0.
        title : str, optional
            The title of the summary table.
        model_name : string
            The name of the model used. Default is to use model class name.
        display_params : boolean, optional
            Whether or not to display tables of estimated parameters. Default
            is True. Usually only used internally.

        Returns
        -------
        summary : Summary instance
            This holds the summary table and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary
        """
        from statsmodels.iolib.summary import Summary

        # Model specification results
        model = self.model
        if title is None:
            title = 'Markov Switching Model Results'

        if start is None:
            start = 0
        if self.data.dates is not None:
            dates = self.data.dates
            d = dates[start]
            sample = ['%02d-%02d-%02d' % (d.month, d.day, d.year)]
            d = dates[-1]
            sample += ['- ' + '%02d-%02d-%02d' % (d.month, d.day, d.year)]
        else:
            sample = [str(start), ' - ' + str(self.model.nobs)]

        # Standardize the model name as a list of str
        if model_name is None:
            model_name = model.__class__.__name__

        # Create the tables
        if not isinstance(model_name, list):
            model_name = [model_name]

        top_left = [('Dep. Variable:', None)]
        top_left.append(('Model:', [model_name[0]]))
        for i in range(1, len(model_name)):
            top_left.append(('', ['+ ' + model_name[i]]))
        top_left += [
            ('Date:', None),
            ('Time:', None),
            ('Sample:', [sample[0]]),
            ('', [sample[1]])
        ]

        top_right = [
            ('No. Observations:', [self.model.nobs]),
            ('Log Likelihood', ["%#5.3f" % self.llf]),
            ('AIC', ["%#5.3f" % self.aic]),
            ('BIC', ["%#5.3f" % self.bic]),
            ('HQIC', ["%#5.3f" % self.hqic])
        ]

        if hasattr(self, 'cov_type'):
            top_left.append(('Covariance Type:', [self.cov_type]))

        summary = Summary()
        summary.add_table_2cols(self, gleft=top_left, gright=top_right,
                                title=title)

        # Make parameters tables for each regime
        from statsmodels.iolib.summary import summary_params
        import re

        def make_table(self, mask, title, strip_end=True):
            res = (self, self.params[mask], self.bse[mask],
                   self.tvalues[mask], self.pvalues[mask],
                   self.conf_int(alpha)[mask])

            param_names = [
                re.sub(r'\[\d+\]$', '', name) for name in
                np.array(self.data.param_names)[mask].tolist()
            ]

            return summary_params(res, yname=None, xname=param_names,
                                  alpha=alpha, use_t=False, title=title)

        params = model.parameters
        regime_masks = [[] for i in range(model.k_regimes)]
        other_masks = {}
        for key, switching in params.switching.items():
            k_params = len(switching)
            if key == 'regime_transition':
                continue
            other_masks[key] = []

            for i in range(k_params):
                if switching[i]:
                    for j in range(self.k_regimes):
                        regime_masks[j].append(params[j, key][i])
                else:
                    other_masks[key].append(params[0, key][i])

        for i in range(self.k_regimes):
            mask = regime_masks[i]
            if len(mask) > 0:
                table = make_table(self, mask, 'Regime %d parameters' % i)
                summary.tables.append(table)

        mask = []
        for key, _mask in other_masks.items():
            mask.extend(_mask)
        if len(mask) > 0:
            table = make_table(self, mask, 'Non-switching parameters')
            summary.tables.append(table)

        # Transition parameters
        mask = params['regime_transition']
        table = make_table(self, mask, 'Regime transition parameters')
        summary.tables.append(table)

        # Add warnings/notes, added to text format only
        etext = []
        if hasattr(self, 'cov_type') and 'description' in self.cov_kwds:
            etext.append(self.cov_kwds['description'])
        if self._rank < len(self.params):
            etext.append("Covariance matrix is singular or near-singular,"
                         " with condition number %6.3g. Standard errors may be"
                         " unstable." % np.linalg.cond(self.cov_params()))

        if etext:
            etext = ["[{0}] {1}".format(i + 1, text)
                     for i, text in enumerate(etext)]
            etext.insert(0, "Warnings:")
            summary.add_extra_txt(etext)

        return summary


class MarkovSwitchingResultsWrapper(wrap.ResultsWrapper):
    _attrs = {
        'cov_params_approx': 'cov',
        'cov_params_default': 'cov',
        'cov_params_opg': 'cov',
        'cov_params_robust': 'cov',
    }
    _wrap_attrs = wrap.union_dicts(tsbase.TimeSeriesResultsWrapper._wrap_attrs,
                                   _attrs)
    _methods = {
        'forecast': 'dates',
    }
    _wrap_methods = wrap.union_dicts(
        tsbase.TimeSeriesResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(MarkovSwitchingResultsWrapper,  # noqa:E305
                      MarkovSwitchingResults)
