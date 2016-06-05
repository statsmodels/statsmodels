"""
Markov switching models

Author: Chad Fulton
License: BSD
"""

from __future__ import division, absolute_import, print_function

import warnings
import numpy as np
import pandas as pd
from collections import OrderedDict

from scipy.misc import logsumexp
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.tools import Bunch
from statsmodels.tools.numdiff import approx_fprime_cs, approx_hess_cs
from statsmodels.tools.decorators import cache_readonly, resettable_cache
from statsmodels.tools.eval_measures import aic, bic, hqic
from statsmodels.tools.tools import pinv_extended
import statsmodels.base.wrapper as wrap


from statsmodels.tsa.statespace.tools import find_best_blas_type
from statsmodels.tsa.regime_switching._hamilton_filter import (
    shamilton_filter, dhamilton_filter, chamilton_filter, zhamilton_filter)

prefix_hamilton_filter_map = {
    's': shamilton_filter, 'd': dhamilton_filter,
    'c': chamilton_filter, 'z': zhamilton_filter
}


def _prepare_exog(exog):
    k_exog = 0
    if exog is not None:
        exog_is_using_pandas = _is_using_pandas(exog, None)
        if not exog_is_using_pandas:
            exog = np.asarray(exog)

        # Make sure we have 2-dimensional array
        if exog.ndim == 1:
            if not exog_is_using_pandas:
                exog = exog[:, None]
            else:
                exog = pd.DataFrame(exog)

        k_exog = exog.shape[1]
    return k_exog, exog


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
        partials = np.concatenate(partials).reshape(shape).transpose((1,2,0))
    # k x k x j x t
    else:
        partials = [[np.diag(tmp[:, j, t] - tmp[:, j, t]**2)
                     for t in range(tmp.shape[2])]
                    for j in range(tmp.shape[1])]
        shape = tmp.shape[1], tmp.shape[2], tmp.shape[0], tmp.shape[0]
        partials = np.concatenate(partials).reshape(shape).transpose((2,3,0,1))

    for i in range(tmp.shape[0]):
        for j in range(i):
            partials[i, j, ...] = -tmp[i, ...] * tmp[j, ...]
            partials[j, i, ...] = partials[i, j, ...]
    return partials


def py_hamilton_filter(initial_probabilities, transition,
                       conditional_likelihoods):
    # Dimensions
    k_regimes = len(initial_probabilities)
    nobs = conditional_likelihoods.shape[-1]
    order = conditional_likelihoods.ndim - 2
    dtype = conditional_likelihoods.dtype

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
    for i in range(order):
        tmp = np.reshape(transition[..., i], shape + (1,) * i) * tmp
    filtered_joint_probabilities[..., 0] = tmp

    # Reshape transition so we can use broadcasting
    shape = (k_regimes, k_regimes)
    shape += (1,) * (order-1)
    shape += (transition.shape[-1],)
    transition = np.reshape(transition, shape)[..., order:]

    # Hamilton filter iterations
    transition_t = 0
    for t in range(nobs):
        if transition.shape[-1] > 1:
            transition_t = t

        # S_t, S_{t-1}, ..., S_{t-r} | t-1, stored at zero-indexed location t
        predicted_joint_probabilities[..., t] = (
            # S_t | S_{t-1}
            transition[..., transition_t] *
            # S_{t-1}, S_{t-2}, ..., S_{t-r} | t-1
            filtered_joint_probabilities[..., t].sum(axis=-1))

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


def cy_hamilton_filter(initial_probabilities, transition,
                       conditional_likelihoods):
    # Dimensions
    k_regimes = len(initial_probabilities)
    nobs = conditional_likelihoods.shape[-1]
    order = conditional_likelihoods.ndim - 2
    dtype = conditional_likelihoods.dtype

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
        if transition.shape[-1] > 1:
            transition_t = i
        tmp = np.reshape(transition[..., transition_t], shape + (1,) * i) * tmp
    filtered_joint_probabilities[..., 0] = tmp

    # Get appropriate subset of transition matrix
    if transition.shape[-1] > 1:
        transition = transition[..., order:]

    # Run Cython filter iterations
    prefix, dtype, _ = find_best_blas_type((
        transition, conditional_likelihoods, joint_likelihoods,
        predicted_joint_probabilities, filtered_joint_probabilities))
    func = prefix_hamilton_filter_map[prefix]
    func(nobs, k_regimes, order, transition,
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


def py_kim_smoother(transition, filtered_marginal_probabilities,
                    predicted_joint_probabilities,
                    filtered_joint_probabilities):
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
    shape += (transition.shape[-1],)
    transition = np.reshape(transition, shape)

    # Get appropriate subset of transition matrix
    if transition.shape[-1] > 1:
        transition = transition[..., order:]

    # Kim smoother iterations
    transition_t = 0
    for t in range(nobs - 2, -1, -1):
        if transition.shape[-1] > 1:
            transition_t = t+1

        # S_{t+1}, S_t, ..., S_{t-r+1} | t
        # x = predicted_joint_probabilities[..., t]
        x = (filtered_joint_probabilities[..., t] *
             transition[..., transition_t])
        # S_{t+1}, S_t, ..., S_{t-r+2} | T / S_{t+1}, S_t, ..., S_{t-r+2} | t
        y = (smoothed_joint_probabilities[..., t+1] /
             predicted_joint_probabilities[..., t+1])
        # S_{t+1}, S_t, ..., S_{t-r+1} | T
        smoothed_joint_probabilities[..., t] = (x * y[..., None]).sum(axis=0)

    # Get smoothed marginal probabilities S_t | T by integrating out
    # S_{t-k+1}, S_{t-k+2}, ..., S_{t-1}
    smoothed_marginal_probabilities = smoothed_joint_probabilities
    for i in range(1, smoothed_marginal_probabilities.ndim - 1):
        smoothed_marginal_probabilities = np.sum(
            smoothed_marginal_probabilities, axis=-2)

    return smoothed_joint_probabilities, smoothed_marginal_probabilities


class MarkovSwitchingParams(object):
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
                self.index_regime[j] = np.concatenate(indices)
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
    exog_tvtp : array_like, optional
        Array of exogenous or lagged variables to use in calculating
        time-varying transition probabilities (TVTP). TVTP is only used if this
        variable is provided. If an intercept is desired, a column of ones must
        be explicitly included in this array.

    References
    ----------
    Kim, Chang-Jin, and Charles R. Nelson. 1999.
    "State-Space Models with Regime Switching:
    Classical and Gibbs-Sampling Approaches with Applications".
    MIT Press Books. The MIT Press.

    """

    def __init__(self, endog, k_regimes, exog_tvtp=None, exog=None, dates=None,
                 freq=None, missing='none'):

        # Properties
        self.k_regimes = k_regimes
        self.tvtp = exog_tvtp is not None

        # Exogenous data
        # TODO add checks for exog_tvtp consistent shape and indices
        self.k_tvtp, self.exog_tvtp = _prepare_exog(exog_tvtp)

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
        self.parameters['transition'] = [1] * k_transition

        # Internal model properties: default is steady-state initialization
        self._initialization = 'steady-state'
        self._initial_probabilities = None

    @property
    def k_params(self):
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
                             ' the transition matrix is time-varying.')

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

    def initial_probabilities(self, params, transition=None):
        """
        Retrieve initial probabilities
        """
        params = np.array(params, ndmin=1)
        if self._initialization == 'steady-state':
            if transition is None:
                transition = self.transition_matrix(params)
            if transition.ndim == 3:
                transition = transition[..., 0]
            m = transition.shape[0]
            A = np.c_[(np.eye(m) - transition).T, np.ones(m)].T
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

    def _transition_matrix_tvtp(self, params):
        transition_matrix = np.zeros(
            (self.k_regimes, self.k_regimes, len(self.exog_tvtp)),
            dtype=np.promote_types(np.float64, params.dtype))

        # Compute the predicted values from the regression
        for i in range(self.k_regimes):
            coeffs = params[self.parameters[i, 'transition']]
            transition_matrix[:-1, i, :] = np.dot(
                self.exog_tvtp,
                np.reshape(coeffs, (self.k_regimes-1, self.k_tvtp)).T).T

        # Perform the logit transformation
        tmp = np.c_[np.zeros((len(self.exog_tvtp), self.k_regimes, 1)),
                    transition_matrix[:-1, :, :].T].T
        transition_matrix[:-1, :, :] = np.exp(transition_matrix[:-1, :, :] -
                                              logsumexp(tmp, axis=0))

        # Compute the last column of the transition matrix
        transition_matrix[-1, :, :] = (
            1 - np.sum(transition_matrix[:-1, :, :], axis=0))

        return transition_matrix

    def transition_matrix(self, params):
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
            transition_matrix = np.zeros((self.k_regimes, self.k_regimes, 1),
                                         dtype=np.promote_types(np.float64,
                                                                params.dtype))
            transition_matrix[:-1, :, 0] = np.reshape(
                params[self.parameters['transition']],
                (self.k_regimes-1, self.k_regimes))
            transition_matrix[-1, :, 0] = (
                1 - np.sum(transition_matrix[:-1, :, 0], axis=0))
        else:
            transition_matrix = self._transition_matrix_tvtp(params)

        return transition_matrix

    def _conditional_likelihoods(self, params):
        raise NotImplementedError

    def _filter(self, params, transition=None):
        # Get the transition matrix if not provided
        if transition is None:
            transition = self.transition_matrix(params)
        # Get the initial probabilities
        initial_probabilities = self.initial_probabilities(params, transition)

        # Compute the conditional likelihoods
        conditional_likelihoods = self._conditional_likelihoods(params)

        # Apply the filter
        return ((transition, initial_probabilities, conditional_likelihoods) +
                cy_hamilton_filter(initial_probabilities, transition,
                                   conditional_likelihoods))

    def filter(self, params, transformed=True, cov_type=None, cov_kwds=None,
               return_raw=False, results_class=None,
               results_wrapper_class=None):
        """
        Apply the Hamilton filter
        """
        params = np.array(params, ndmin=1)

        if not transformed:
            params = self.transform_params(params)

        # Save the parameter names
        self.data.param_names = self.param_names

        # Get the result
        names = ['transition', 'initial_probabilities',
                 'conditional_likelihoods', 'filtered_marginal_probabilities',
                 'predicted_joint_probabilities', 'joint_likelihoods',
                 'filtered_joint_probabilities']
        result = HamiltonFilterResults(
            self, Bunch(**dict(zip(names, self._filter(params)))))

        # Wrap in a results object
        if not return_raw:
            result_kwargs = {}
            if cov_type is not None:
                result_kwargs['cov_type'] = cov_type
            if cov_kwds is not None:
                result_kwargs['cov_kwds'] = cov_kwds

            if results_class is None:
                results_class = MarkovSwitchingResults
            if results_wrapper_class is None:
                results_wrapper_class = MarkovSwitchingResultsWrapper

            result = results_wrapper_class(
                results_class(self, params, result, **result_kwargs)
            )

        return result

    def _smooth(self, params, filtered_marginal_probabilities,
                predicted_joint_probabilities,
                filtered_joint_probabilities, transition=None):
        # Get the transition matrix
        if transition is None:
            transition = self.transition_matrix(params)

        # Apply the smoother
        return py_kim_smoother(transition, filtered_marginal_probabilities,
                               predicted_joint_probabilities,
                               filtered_joint_probabilities)

    def smooth(self, params, transformed=True, cov_type=None, cov_kwds=None,
               return_raw=False, results_class=None,
               results_wrapper_class=None):
        """
        Apply the Kim smoother
        """
        params = np.array(params, ndmin=1)

        if not transformed:
            params = self.transform_params(params)

        # Save the parameter names
        self.data.param_names = self.param_names

        # Hamilton filter
        names = ['transition', 'initial_probabilities',
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
        if not return_raw:
            result_kwargs = {}
            if cov_type is not None:
                result_kwargs['cov_type'] = cov_type
            if cov_kwds is not None:
                result_kwargs['cov_kwds'] = cov_kwds

            if results_class is None:
                results_class = MarkovSwitchingResults
            if results_wrapper_class is None:
                results_wrapper_class = MarkovSwitchingResultsWrapper

            result = results_wrapper_class(
                results_class(self, params, result, **result_kwargs)
            )

        return result

    def loglikeobs(self, params, transformed=True):
        """
        Compute the loglikelihood
        """
        params = np.array(params, ndmin=1)

        if not transformed:
            params = self.transform_params(params)

        results = self._filter(params)

        return np.log(results[5])

    def loglike(self, params, transformed=True):
        """
        Compute the loglikelihood
        """
        return np.sum(self.loglikeobs(params, transformed))

    def score(self, params, transformed=True):
        """
        Compute the score function at params.
        """
        params = np.array(params, ndmin=1)

        return approx_fprime_cs(params, self.loglike, args=(transformed,))

    def score_obs(self, params, transformed=True):
        """
        Compute the score per observation, evaluated at params
        """
        params = np.array(params, ndmin=1)

        return approx_fprime_cs(params, self.loglikeobs, args=(transformed,))

    def hessian(self, params, transformed=True):
        """
        Hessian matrix of the likelihood function, evaluated at the given
        parameters
        """
        params = np.array(params, ndmin=1)

        return approx_hess_cs(params, self.loglike)

    def fit(self, start_params=None, transformed=True, cov_type='opg',
            cov_kwds=None, method='bfgs', maxiter=100, full_output=1, disp=0,
            callback=None, return_params=False, em_iter=5, search_reps=0,
            search_iter=5, search_scale=1., **kwargs):

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

    def _fit_em(self, start_params=None, transformed=True, cov_type='opg',
                cov_kwds=None, maxiter=50, tolerance=1e-6, full_output=True,
                return_params=False, **kwargs):

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
        params1 = np.zeros(params0.shape,
                           dtype=np.promote_types(np.float64, params0.dtype))

        # Smooth at the given parameters
        result = self.smooth(params0, transformed=True, return_raw=True)

        # The EM with TVTP is not yet supported, just return the previous
        # iteration parameters
        if self.tvtp:
            params1[self.parameters['transition']] = (
                params0[self.parameters['transition']])
        else:
            transition = self._em_transition(result)
            for i in range(self.k_regimes):
                params1[self.parameters[i, 'transition']] = transition[i]

        return result, params1

    def _em_transition(self, result):
        # Marginalize the smoothed joint probabilites to just S_t, S_{t-1} | T
        tmp = result.smoothed_joint_probabilities
        for i in range(tmp.ndim - 3):
            tmp = np.sum(tmp, -2)
        smoothed_joint_probabilities = tmp

        # Transition parameters (recall we're not yet supporting TVTP here)
        k_transition = len(self.parameters[0, 'transition'])
        transition = np.zeros((self.k_regimes, k_transition))
        for i in range(self.k_regimes):  # S_{t_1}
            for j in range(self.k_regimes - 1):  # S_t
                transition[i, j] = (
                    np.sum(smoothed_joint_probabilities[j, i]) /
                    np.sum(result.smoothed_marginal_probabilities[i]))

            # It may be the case that due to rounding error this estimates
            # transition probabilities that sum to greater than one. If so,
            # re-scale the probabilities and warn the user that something
            # is not quite right
            delta = np.sum(transition[i]) - 1
            if delta > 0:
                warnings.warn('Invalid transition probabilities estimated in'
                              ' EM iteration; probabilities have been'
                              ' re-scaled to continue estimation.')
                transition[i] /= 1 + delta + 1e-6

        return transition

    def _start_params_search(self, reps, start_params=None, transformed=True,
                             em_iter=5, scale=1.):
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
                except:
                    pass

        # Return transformed parameters
        return self.transform_params(params)

    @property
    def start_params(self):
        params = np.zeros(self.k_params, dtype=np.float64)

        # Transition probabilities
        if self.tvtp:
            params[self.parameters['transition']] = 0.
        else:
            params[self.parameters['transition']] = 1. / self.k_regimes

        return params

    @property
    def param_names(self):
        param_names = np.zeros(self.k_params, dtype=object)

        # Transition probabilities
        if self.tvtp:
            # TODO add support for exog_tvtp_names
            param_names[self.parameters['transition']] = [
                'p[%d->%d].tvtp%d' % (j, i, k)
                for i in range(self.k_regimes-1)
                for k in range(self.k_tvtp)
                for j in range(self.k_regimes)
                ]
        else:
            param_names[self.parameters['transition']] = [
                'p[%d->%d]' % (j, i)
                for i in range(self.k_regimes-1)
                for j in range(self.k_regimes)]

        return param_names.tolist()

    def transform_params(self, unconstrained):
        constrained = np.array(unconstrained, copy=True)
        constrained = constrained.astype(
            np.promote_types(np.float64, constrained.dtype))

        # Nothing to do for transition probabilities if TVTP
        if self.tvtp:
            constrained[self.parameters['transition']] = (
                unconstrained[self.parameters['transition']])
        # Otherwise do logit transformation
        else:
            # Transition probabilities
            offset = 0
            for i in range(self.k_regimes):
                tmp1 = unconstrained[self.parameters[i, 'transition']]
                tmp2 = np.r_[0, tmp1]
                # Don't let transition probabilities be exactly equal to 1 or 0
                # (causes problems with the optimizer)
                constrained[self.parameters[i, 'transition']] = np.exp(
                    tmp1 - logsumexp(tmp2))

        # Do not do anything for the rest of the parameters

        return constrained

    def _untransform_logit(self, unconstrained, constrained):
        resid = np.zeros(unconstrained.shape, dtype=unconstrained.dtype)
        exp = np.exp(unconstrained)
        sum_exp = np.sum(exp)
        for i in range(len(unconstrained)):
            resid[i] = (unconstrained[i] -
                        np.log(1 + sum_exp - exp[i]) +
                        np.log(1 / constrained[i] - 1))
        return resid

    def untransform_params(self, constrained):
        unconstrained = np.array(constrained, copy=True)
        unconstrained = unconstrained.astype(
            np.promote_types(np.float64, unconstrained.dtype))

        # Nothing to do for transition probabilities if TVTP
        if self.tvtp:
            unconstrained[self.parameters['transition']] = (
                constrained[self.parameters['transition']])
        # Otherwise reverse logit transformation
        else:
            for i in range(self.k_regimes):
                s = self.parameters[i, 'transition']
                if self.k_regimes == 2:
                    unconstrained[s] = -np.log(1. / constrained[s] - 1)
                else:
                    from scipy.optimize import root
                    out = root(self._untransform_logit,
                               np.zeros(unconstrained[s].shape,
                                        unconstrained.dtype),
                               args=(constrained[s],))
                    if not out['success']:
                        raise ValueError('Could not untransform parameters.')
                    unconstrained[s] = out['x']

        # Do not do anything for the rest of the parameters

        return unconstrained


class HamiltonFilterResults(object):
    def __init__(self, model, result):

        self.model = model

        self.nobs = model.nobs
        self.k_regimes = model.k_regimes

        attributes = ['transition', 'initial_probabilities',
                      'conditional_likelihoods',
                      'predicted_joint_probabilities',
                      'filtered_marginal_probabilities',
                      'filtered_joint_probabilities',
                      'joint_likelihoods']
        for name in attributes:
            setattr(self, name, getattr(result, name))

        self.llf_obs = np.log(self.joint_likelihoods)
        self.llf = np.sum(self.llf_obs)

        # Subset transition if necessary (e.g. for Markov autoregression)
        diff = self.transition.shape[-1] - self.nobs
        if self.transition.shape[-1] > 1 and diff > 0:
            self.transition = self.transition[..., diff:]

    @property
    def expected_durations(self):
        return 1. / (1 - np.diagonal(self.transition).squeeze())


class KimSmootherResults(HamiltonFilterResults):
    def __init__(self, model, result):
        super(KimSmootherResults, self).__init__(model, result)

        attributes = ['smoothed_joint_probabilities',
                      'smoothed_marginal_probabilities']

        for name in attributes:
            setattr(self, name, getattr(result, name))


class MarkovSwitchingResults(tsbase.TimeSeriesModelResults):
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

        # Setup covariance matrix notes dictionary
        if not hasattr(self, 'cov_kwds'):
            self.cov_kwds = {}
        self.cov_type = cov_type

        # Setup the cache
        self._cache = resettable_cache()

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
        attributes = ['transition', 'initial_probabilities',
                      'conditional_likelihoods',
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

    def _get_robustcov_results(self, cov_type='opg', **kwargs):
        import statsmodels.stats.sandwich_covariance as sw

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

        # Calculate the new covariance matrix
        k_params = len(self.params)
        if k_params == 0:
            res.cov_params_default = np.zeros((0, 0))
            res._rank = 0
            res.cov_kwds['description'] = 'No parameters estimated.'
        elif cov_type == 'none':
            res.cov_params_default = np.zeros((k_params, k_params)) * np.nan
            res._rank = np.nan
            res.cov_kwds['description'] = 'Covariance matrix not calculated.'
        elif self.cov_type == 'approx':
            res.cov_params_default = res.cov_params_approx
            res.cov_kwds['description'] = (
                'Covariance matrix calculated using numerical'
                ' differentiation.')
        elif self.cov_type == 'opg':
            res.cov_params_default = res.cov_params_opg
            res.cov_kwds['description'] = (
                'Covariance matrix calculated using the outer product of'
                ' gradients.'
            )
        elif self.cov_type == 'robust':
            res.cov_params_default = res.cov_params_robust
            res.cov_kwds['description'] = (
                'Quasi-maximum likelihood covariance matrix used for'
                ' robustness to some misspecifications; calculated using'
                ' numerical differentiation.')
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
        score_obs = self.model.score_obs(self.params, transformed=True)
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
        # This is a (k_endog x nobs array; don't want to squeeze in case of
        # the corner case where nobs = 1 (mostly a concern in the predict or
        # forecast functions, but here also to maintain consistency)
        raise NotImplementedError

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
    def pvalues(self):
        """
        (array) The p-values associated with the z-statistics of the
        coefficients. Note that the coefficients are assumed to have a Normal
        distribution.
        """
        return norm.sf(np.abs(self.zvalues)) * 2

    @cache_readonly
    def resid(self):
        """
        (array) The model residuals. An (nobs x k_endog) array.
        """
        # This is a (k_endog x nobs array; don't want to squeeze in case of
        # the corner case where nobs = 1 (mostly a concern in the predict or
        # forecast functions, but here also to maintain consistency)
        raise NotImplementedError

    @cache_readonly
    def zvalues(self):
        """
        (array) The z-statistics for the coefficients.
        """
        return self.params / self.bse


class MarkovSwitchingResultsWrapper(wrap.ResultsWrapper):
    _attrs = {
        'zvalues': 'columns',
        'cov_params_approx': 'cov',
        'cov_params_default': 'cov',
        'cov_params_opg': 'cov',
        'cov_params_robust': 'cov',
    }
    _wrap_attrs = wrap.union_dicts(tsbase.TimeSeriesResultsWrapper._wrap_attrs,
                                   _attrs)
    _methods = {
        'forecast': 'dates',
        'simulate': 'ynames',
        'impulse_responses': 'ynames'
    }
    _wrap_methods = wrap.union_dicts(
        tsbase.TimeSeriesResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(MarkovSwitchingResultsWrapper, MarkovSwitchingResults)
