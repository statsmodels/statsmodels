"""
Markov Switching Autoregression

Author: Valery Likhosherstov
License: Simplified-BSD
"""
import numpy as np
from .switching_mlemodel import SwitchingMLEModel
from .kim_smoother import KimSmootherResults
from statsmodels.tsa.statespace.api import SARIMAX

def _em_iteration_for_markov_regression(dtype, k_regimes, endog, exog,
        smoothed_regime_probs, smoothed_curr_and_next_regime_probs):

    # Useful method, implementing EM-iteration of Markov switching regression
    # model, based on chapter 4.3.5 of
    # Kim, Chang-Jin and Nelson, Charles R. 1999.
    # State-space Models With Regime Switching : Classical and
    # Gibbs-sampling Approaches With Applications. MIT Press.

    # In Kim-Nelson notation `endog` represents y_t, `exog` - x_t,
    # `smoothed_regime_probs` - p(S_t | \bar{y}_T),
    # `smoothed_curr_and_next_regime_probs` - p(S_t, S_{t-1} | \bar{y}_T)

    # \beta_i storage
    coefs = np.zeros((k_regimes, exog.shape[1]), dtype=dtype)
    # \sigma^{2}_i storage
    variances = np.zeros((k_regimes,), dtype=dtype)

    for regime in range(k_regimes):
        # x_t \times \sqrt{p(S_t = i | \bar{y}_T)}
        regression_exog = exog * \
                np.sqrt(smoothed_regime_probs[regime, :].reshape(-1, 1))
        # y_t \times \sqrt{p(S_t = i | \bar{y}_T)}
        regression_endog = endog * np.sqrt(smoothed_regime_probs[regime, :])

        # Calculating regression. Probably may raise exceptions in case of
        # ill-conditioned data.
        coefs[regime, :] = np.linalg.lstsq(regression_exog, regression_endog)[0]

        # (y_t - x^{\prime}_t \beta_i)^2
        sqr_residuals = (endog - \
                exog.dot(coefs[regime, :].reshape(-1, 1)).ravel())**2

        # \sum_{t} p(S_t = i | \bar{y}_T)
        marginal_regime_prob_sum = np.sum(smoothed_regime_probs[regime, :])

        # Variance as weighted residuals sum
        if marginal_regime_prob_sum != 0:
            variances[regime] = \
                    np.sum(sqr_residuals * smoothed_regime_probs[regime, :]) / \
                    marginal_regime_prob_sum
        else:
            # Any value would be alright, because regime is unreachable
            variances[regime] = 1

    # \sum_{t} p(S_t = j, S_{t-1} = i | \bar{y}_T)
    joint_prob_sum = np.sum(smoothed_curr_and_next_regime_probs, axis=2)

    # \sum{t-1} p(S_{t-1} = i | \bar{y}_T)
    marginal_prob_sum = np.sum(smoothed_regime_probs[:, :-1], axis=1)

    # Regime transition matrix initialization
    regime_transition = np.zeros((k_regimes, k_regimes), dtype=dtype)

    # Switching probabilities from unreachable regime can take any value, but
    # still be a probability space
    regime_transition[marginal_prob_sum == 0, :] = 1.0 / k_regimes

    # EM-iteration for switching probabilities from reachable regimes
    regime_transition[marginal_prob_sum != 0, :] = \
            joint_prob_sum[marginal_prob_sum != 0] / \
            marginal_prob_sum[marginal_prob_sum != 0].reshape(-1, 1)

    # Make matrix left-stochastic
    regime_transition = regime_transition.transpose()

    return (coefs, variances, regime_transition)


class _NonswitchingAutoregression(SARIMAX):

    # This is a SARIMAX wrapper, simplifying it to AR(p) model.
    # It is used as non-switching model in `MarkovAutoregression` class

    def __init__(self, endog, order, exog=None, dtype=np.float64):

        # Use exog
        mle_regression = True

        # Constant exog variable
        intercept = np.ones((endog.shape[0], 1))

        if exog is not None:
            exog = np.hstack((intercept, exog))
        else:
            exog = intercept

        super(_NonswitchingAutoregression, self).__init__(endog, exog=exog,
                order=(order, 0, 0), mle_regression=mle_regression,
                enforce_stationarity=False)


class MarkovAutoregression(SwitchingMLEModel):
    r"""
    Markov switching autoregression

    Parameters
    ----------
    k_ar_regimes : int
        The number of Markov switching autoregression regimes. Note, that this
        is different from the number of regimes, used internally in state-space
        model.
    order : int
        The order of the autoregressive lag polynomial.
    endog : array_like
        The endogenous variable.
    switching_ar : bool or iterable, optional
        If a boolean, sets whether or not all autoregressive coefficients are
        switching across regimes. If an iterable, should be of length equal
        to `order`, where each element is a boolean describing whether the
        corresponding coefficient is switching. Default is `True`.
    switching_mean : bool, optional
        Sets whether or not mean of the process is switching. Default is
        `False`.
    switching_variance : bool, optional
        Whether or not there is regime-specific heteroskedasticity, i.e.
        whether or not the error term has a switching variance. Default is
        `False`.
    exog : array_like, optional
        Array of exogenous regressors, shaped nobs x k_exog.
    **kwargs
        Additional keyword arguments, passed to superclass initializer.

    Notes
    -----
    The general model can be written as:

    .. math::

        \phi_{S_t}(L)(y_t - \mu_{S_t} - x_t' \beta_t) = e_t \\
        e_t \sim N(0, \sigma_{S_t}^2)

    i.e. the model is an autoregression where the autoregressive
    coefficients, the mean of the process and the variance of the error term
    may be switching across regimes.

    The passed `exog` array should not have a column of constants, because
    constant regressors are expressed by mean value.

    References
    ----------
    Kim, Chang-Jin, and Charles R. Nelson. 1999.
    "State-Space Models with Regime Switching:
    Classical and Gibbs-Sampling Approaches with Applications".
    MIT Press Books. The MIT Press.
    """

    def __init__(self, k_ar_regimes, order, endog, switching_ar=False,
            switching_mean=False, switching_variance=False, exog=None,
            **kwargs):

        self.order = order
        self.k_ar_regimes = k_ar_regimes

        # Set `k_exog` and `exog` attributes
        if exog is not None:
            exog = np.asarray(exog)
            if exog.ndim == 1:
                exog = exog.reshape(-1, 1)
            self.k_exog = exog.shape[1]
        else:
            self.exog = None
            self.k_exog = None

        endog = endog.ravel()

        # In case of autoregression first `order` endog and exog values are
        # used to calculate initialization of state space model
        self.endog_head = endog[:order]
        endog = endog[order:]

        if exog is not None:
            self.exog_head = exog[:order]
            exog = exog[order:]

        # Broadcasting and saving `switching_ar` attribute
        if isinstance(switching_ar, bool):
            self.switching_ar = [switching_ar] * order
        else:
            if len(switching_ar) != order:
                raise ValueError('Invalid iterable passed to `switching_ar`.')
            self.switching_ar = switching_ar

        # Check if user needs switching model
        if k_ar_regimes == 1 or not (switching_mean or switching_variance or \
                any(self.switching_ar)):
            raise ValueError('Consider using SARIMAX model')

        self.switching_mean = switching_mean
        self.switching_variance = switching_variance

        # Error covariance in case of autoregression is a single number
        kwargs['k_posdef'] = 1

        # `k_ar_regimes**(order + 1)` is a number of regimes used internally in
        # the state space model
        super(MarkovAutoregression, self).__init__(k_ar_regimes**(order + 1),
                endog, order, param_k_regimes=k_ar_regimes, exog=exog, **kwargs)

        # Creating parameters slices

        if self.exog is not None:
            self.parameters['exog'] = [False] * self.k_exog
            # Add exog param names
            self._param_names += ['exog{0}'.format(i) for i in \
                    range(self.k_exog)]

        self.parameters['autoregressive'] = self.switching_ar
        # Add autoregressive param names
        for i, is_switching in zip(range(order), self.switching_ar):
            if is_switching:
                self._param_names += ['phi{0}_{1}'.format(i, j) for j in \
                        range(k_ar_regimes)]
            else:
                self._param_names += ['phi{0}'.format(i)]

        self.parameters['mean'] = [self.switching_mean]
        # Add mean param names
        if self.switching_mean:
            self._param_names += ['mu_{0}'.format(i) for i in \
                    range(k_ar_regimes)]
        else:
            self._param_names += ['mu']

        self.parameters['variance'] = [self.switching_variance]
        # Add variance param names
        if self.switching_variance:
            self._param_names += ['sigma^2_{0}'.format(i) for i in \
                    range(k_ar_regimes)]

    def get_nonswitching_model(self):

        exog = None
        if hasattr(self, 'exog'):
            exog = self.exog

        return _NonswitchingAutoregression(self.endog, self.order, exog=exog,
                dtype=self.ssm.dtype)

    def update_params(self, params, nonswitching_params, noise=0.1, seed=1):
        """
        Update constrained parameters of the model, using parameters of
        non-switching model.
 
        Parameters
        ----------
        noise : float
            Relative normal noise scale, added to switching parameters to break
            the symmetry of regimes.
        seed : int
            Random seed, used to generate the noise.

        Notes
        -----
        `noise` and `seed` parameters are defined in keyword arguments so that
        they can be changed in child class.

        See Also
        --------
        SwitchingMLEModel.update_params
        """

        dtype = self.ssm.dtype
        order = self.order
        k_ar_regimes = self.k_ar_regimes

        switching_mean = self.switching_mean
        switching_ar = self.switching_ar
        switching_variance = self.switching_variance

        # Set the seed
        np.random.seed(seed=seed)

        # Low-level `nonswitching_params` slices handling
        offset = 0

        # In case of non-switching mean just set it non-switching model mean
        # Otherwise, set all regimes means to one value
        params[self.parameters['mean']] = nonswitching_params[offset]

        # Add noise, if switching
        if switching_mean:
            noise_scale = np.absolute(nonswitching_params[offset]) * noise

            params[self.parameters['mean']] += np.random.normal(
                    scale=noise_scale, size=k_ar_regimes)

        # Setting exog coefficients to those from non-switching model
        if self.k_exog is not None:
            params[self.parameters['exog']] = \
                    nonswitching_params[offset:offset + self.k_exog]
            offset += self.k_exog

        offset += 1

        ar_coefs = nonswitching_params[offset:offset + order]
        # Setting all autoregressive coefficients to their non-switching analogs
        for i in range(k_ar_regimes):
            params[self.parameters[i, 'autoregressive']] = ar_coefs

        # Add noise, if switching
        if any(switching_ar):

            mask = np.array(switching_ar)

            # Get normal noise scale
            noise_scales = np.absolute(nonswitching_params[
                    offset:offset + order]) * noise

            for i in range(k_ar_regimes):
                params[self.parameters[i, 'autoregressive']][mask] += \
                        np.random.normal(scale=noise_scales[mask])

        offset += order

        # Setting variance
        params[self.parameters['variance']] = nonswitching_params[offset]

        if switching_variance:
            noise_scale = nonswitching_params[offset] * noise

            params[self.parameters['variance']] += np.random.normal(
                    scale=noise_scale, size=k_ar_regimes)

            # Keep variances non-negative
            params[self.parameters['variance']] = np.maximum(0,
                    params[self.parameters['variance']])

        return params

    def transform_model_params(self, unconstrained):

        k_ar_regimes = self.k_ar_regimes

        constrained = super(MarkovAutoregression,
                self).transform_model_params(unconstrained)

        # Keeping variance positive
        s = self.parameters['variance']
        constrained[s] = unconstrained[s]**2

        return constrained

    def untransform_model_params(self, constrained):

        k_ar_regimes = self.k_ar_regimes

        unconstrained = super(MarkovAutoregression,
                self).untransform_model_params(constrained)

        # Keeping variance positive
        s = self.parameters['variance']
        unconstrained[s] = constrained[s]**0.5

        return unconstrained

    def get_normal_regimes_permutation(self, params):

        k_ar_regimes = self.k_ar_regimes
        order = self.order

        # First, we construct comparison keys, which are just tuples of
        # regime-specific parameters

        regime_sort_keys = [() for _ in range(k_ar_regimes)]

        if self.switching_mean:
            for i in range(k_ar_regimes):
                regime_sort_keys[i] += (params[self.parameters[i, 'mean']],)

        if self.switching_variance:
            for i in range(k_ar_regimes):
                regime_sort_keys[i] += \
                        (params[self.parameters[i, 'variance']],)

        if any(self.switching_ar):
            for i in range(k_ar_regimes):
                regime_sort_keys[i] += tuple(params[self.parameters[i,
                        'autoregressive']])

        # Then, we sort regime indices according to these keys
        permutation = sorted(range(k_ar_regimes),
                key=lambda regime:regime_sort_keys[regime])

        return permutation

    def _get_ar_regimes(self, regime_index):

        # Regime, used in state space representation, encodes `order + 1`
        # consecutive MS-AR regimes using positional base-`k_ar_regimes` numeral
        # system:
        # S_{internal;t} = S_t + S_{t-1} * k + ... + S_{t-p} * k^p

        # This useful method takes S_{internal;t} and returns S_{t-1}, ...,
        # S_{t-p}

        order = self.order

        k_ar_regimes = self.k_ar_regimes
        regimes_suffix = regime_index
        for ar_coef_index in range(order + 1):
            yield regimes_suffix % k_ar_regimes
            regimes_suffix /= k_ar_regimes

    def _iterate_regimes(self):

        # Regime, used in state space representation, encodes `order + 1`
        # consecutive MS-AR regimes using positional base-`k_ar_regimes` numeral
        # system:
        # S_{internal;t} = S_t + S_{t-1} * k + ... + S_{t-p} * k^p

        # Internal regime (S_t = a_p, ..., S_{t-p} = a_1) can only switch to
        # regimes of form (S_t = b, S_{t-1} = a_p ..., S_{t-p} = a_2).
        # This method iterates through pairs of internal regimes among which
        # switching is possible and also yields corresponding MS-AR regimes
        # (lowest position in regime encoding)

        k_regimes = self.k_regimes
        k_ar_regimes = self.k_ar_regimes
        order = self.order

        # Iterating through all possible a_p + a_{p-1} * k + ... + a_1 * k^p
        for prev_regime_index in range(k_regimes):
            # a_p
            prev_ar_regime = prev_regime_index % k_ar_regimes
            # a_p * k + a_{p-1} * k^2 + ... + a_2 * k^p
            curr_regime_index_without_curr_ar_regime = k_ar_regimes * \
                    (prev_regime_index % (k_ar_regimes**order))
            # Iterating over all possible b
            for curr_ar_regime in range(k_ar_regimes):
                # b + a_p * k + a_{p-1} * k^2 + ... + a_2 * k^p
                curr_regime_index = \
                        curr_regime_index_without_curr_ar_regime + \
                        curr_ar_regime

                yield (prev_regime_index, curr_regime_index, prev_ar_regime,
                        curr_ar_regime)

    def update(self, params, **kwargs):

        # State space representation of autoregressive model is based on chapter
        # 3.3 of
        # [1] Durbin, James, and Siem Jan Koopman. 2012.
        # Time Series Analysis by State Space Methods: Second Edition.
        # Oxford University Press.

        order = self.order
        k_regimes = self.k_regimes
        k_ar_regimes = self.k_ar_regimes
        dtype = self.ssm.dtype

        # This does transformation of parameters
        params = super(MarkovAutoregression, self).update(params, **kwargs)

        # Transition matrix between MS-AR regimes
        ar_regime_transition = self._get_param_regime_transition(params)

        # Regime, used in state space representation, encodes `order + 1`
        # consecutive MS-AR regimes using positional base-`k_ar_regimes` numeral
        # system:
        # S_{internal;t} = S_t + S_{t-1} * k + ... + S_{t-p} * k^p

        # Internal regime (S_t = a_p, ..., S_{t-p} = a_1) can only switch to
        # regimes of form (S_t = b, S_{t-1} = a_p ..., S_{t-p} = a_2).
        # Moreover, probability of this transition is equal to
        # Pr[ S_t = b | S_{t-1} = a_p ].

        # Transition matrix between state space representation regimes
        regime_transition = np.zeros((k_regimes, k_regimes),
                dtype=dtype)

        # Filling state-space representation regime transition according to
        # MS-AR regime transition
        for prev_regime_index, curr_regime_index, prev_ar_regime, \
                        curr_ar_regime in self._iterate_regimes():
            regime_transition[curr_regime_index, prev_regime_index] = \
                    ar_regime_transition[curr_ar_regime, prev_ar_regime]

        self['regime_transition'] = regime_transition

        # Exogenous term of `obs_intercept`

        if self.exog is not None:
            exog_intercept_term = self.exog.dot(
                    params[self.parameters['exog']].reshape(-1, 1)
                    ).reshape(1, -1)

        # Autoregression coefficients for every MS-AR regime (equal among
        # regimes in case of non-switching AR coefficients)

        ar_coefs = np.zeros((k_ar_regimes, order), dtype=dtype)

        for i in range(k_ar_regimes):
            ar_coefs[i] = params[self.parameters[i, 'autoregressive']]

        # State transition matrix
        transition = np.zeros((k_regimes, order, order, 1), dtype=dtype)

        # Filling transition matrix, as proposed in Durbin and Koopman book
        for regime_index in range(k_regimes):
            curr_ar_regime = regime_index % k_ar_regimes
            transition[regime_index, :-1, 1:, 0] = np.identity(order - 1)
            transition[regime_index, :, 0, 0] = ar_coefs[curr_ar_regime, :]

        self['transition'] = transition

        # Switching means

        if self.exog is not None:
            # In case of exog data obs intercept is changing in time
            obs_intercept = np.zeros((k_regimes,) + \
                    exog_intercept_term.shape, dtype=dtype)
            # Obs intercept is a sum of exog term and the mean of the process
            for regime_index in range(k_regimes):
                obs_intercept[regime_index, :, :] = exog_intercept_term
                curr_ar_regime = regime_index % k_ar_regimes
                obs_intercept[regime_index, :, :] += \
                        params[self.parameters[curr_ar_regime, 'mean']]
        else:
            # In the case of no exog data obs intercept is time-invariant
            obs_intercept = np.zeros((k_regimes, 1, 1), dtype=dtype)
            # Filling obs intercept with mean values
            for regime_index in range(k_regimes):
                curr_ar_regime = regime_index % k_ar_regimes
                obs_intercept[regime_index, 0, 0] = \
                        params[self.parameters[curr_ar_regime, 'mean']]

        self['obs_intercept'] = obs_intercept

        # Switching variances

        state_cov = np.zeros((k_regimes, 1, 1, 1), dtype=dtype)
        for i in range(k_regimes):
            state_cov[i, :, :, :] = params[self.parameters[i % k_ar_regimes,
                    'variance']]

        self['state_cov'] = state_cov

        # Other matrices

        # Since variance is a matrix of order 1, `selection` is the following
        selection = np.zeros((order, 1, 1))
        selection[0, 0, 0] = 1

        self['selection'] = selection

        # This is due to Durbin and Koopman book
        design = np.zeros((1, order, 1))
        design[0, 0, 0] = 1

        self['design'] = design

        # `obs_cov` and `state_intercept` are zero by default

        # Initialization

        # Initial state is known a priori, so its covariance is zero
        initial_state_cov = np.zeros((order, order), dtype=dtype)

        initial_state = np.zeros((k_regimes, order), dtype=dtype)

        # Construct initial state according to Durbin and Koopman book,
        # encountering mean term and different regimes

        if self.exog is not None:
            exog_biases = self.exog_head.dot(
                    params[self.parameters['exog']].reshape(-1, 1)
                    ).ravel()
        else:
            exog_biases = np.zeros(order, dtype=dtype)

        for regime_index in range(k_regimes):

            curr_ar_regime = regime_index % k_ar_regimes

            curr_regime_ar_coefs = ar_coefs[curr_ar_regime, :]
            curr_regime_ar_means = np.zeros(order, dtype=dtype)

            for ar_lag_index, ar_regime in zip(range(order),
                    self._get_ar_regimes(regime_index)):
                curr_regime_ar_means[ar_lag_index] = \
                        params[self.parameters[ar_regime, 'mean']]

            curr_regime_biases = exog_biases + curr_regime_ar_means

            initial_state[regime_index, 0] = self.endog_head[-1] - \
                    curr_regime_biases[0]

            for i in range(1, order):
                for j in range(i, order):
                    initial_state[regime_index, i] += \
                            curr_regime_ar_coefs[j] * \
                            (self.endog_head[order - 2 + i - j] - \
                            curr_regime_biases[j])

        self.initialize_known(initial_state, initial_state_cov)

    def _em_iteration(self, params):

        # This method implements EM iteration.
        # Since no exact EM-algorithm is proposed, the idea is to express
        # lagged values and means as regressors and apply EM-iteration for
        # Markov switching regression.

        dtype = self.ssm.dtype
        order = self.order
        k_regimes = self.k_regimes
        k_ar_regimes = self.k_ar_regimes

        # EM-algorithm relies on smoothed regime probabilitie
        # Obtaining them for the case of MS-AR model

        msar_results = self.smooth(params, return_ssm=True,
                return_extended_probs=False)

        smoothed_regime_probs = msar_results.smoothed_regime_probs
        smoothed_curr_and_next_regime_probs = \
                msar_results.smoothed_curr_and_next_regime_probs

        # Preparing input data for Markov switching regression

        # Endogenous data
        markov_regression_endog = self.endog.ravel()

        # Dimension of exogenous vector, which contains `order` lagged
        # observations, 1 intercept and `k_exog` MS-AR exogenous variables
        markov_regression_exog_dim = order + 1
        if self.k_exog is not None:
            markov_regression_exog_dim += self.k_exog

        markov_regression_exog = np.zeros((markov_regression_endog.shape[0],
                markov_regression_exog_dim), dtype=dtype)

        # Filling regression exogenous data by lagged observations
        for i in range(order):
            markov_regression_exog[:i + 1, i] = self.endog_head[-i - 1:]
            markov_regression_exog[i + 1:, i] = markov_regression_endog[:-i - 1]

        # Adding intercept value. It is noted in docstrings, that `self.exog`
        # mustn't contain constant variable.
        markov_regression_exog[:, order] = 1

        # Copying MS-AR exogenous data unchanged.
        if self.exog is not None:
            markov_regression_exog[:, order + 1:] = self.exog

        # EM-iteration for Markov switching regression
        coefs, variances, ar_regime_transition = \
                _em_iteration_for_markov_regression(dtype, k_ar_regimes,
                markov_regression_endog, markov_regression_exog,
                smoothed_regime_probs, smoothed_curr_and_next_regime_probs)

        # Recovering new MS-AR parameters

        new_params = np.zeros((self.parameters.k_params,), dtype=dtype)

        # MS-AR regime transition recovery - just copying transition matrix from
        # regression

        self._set_param_regime_transition(new_params, ar_regime_transition)

        # AR coefficients recovery

        ar_coefs = np.zeros((k_ar_regimes, order), dtype=dtype)
        ar_coefs[:, :] = coefs[:, :order]

        # Approximation - if coefficient is non-switching, set it to the average
        # value of the corresponding switching coefficients
        ar_coefs[:, ~np.array(self.switching_ar)] = \
                ar_coefs[:, ~np.array(self.switching_ar)].mean(axis=0)

        for i in range(k_ar_regimes):
            new_params[self.parameters[i, 'autoregressive']] = ar_coefs[i, :]

        # Variance recovery

        if self.switching_variance:
            new_params[self.parameters['variance']] = variances
        else:
            # Approximation - using average variance, when it doesn't switch
            new_params[self.parameters['variance']] = variances.mean()

        # Recovery of process mean value

        # Switching coefficient, corresponding to constant exog variable
        intercept_terms = coefs[:, order]

        ar_means = np.zeros((k_ar_regimes,), dtype=dtype)

        #ar_means[:] = intercept_terms

        # Following recovery requires approximation.
        # As you can notice, intercepts of different lagged variables in MS-AR
        # definition enter the equation with different indices (which, by the
        # way, causes exponential growth of regimes in the state space form of
        # MS-AR).
        # The approximation treats these intercepts as if all regime indices
        # were the same and equal to current regime index.
        # Than the coefficient of regression is this intercept, multiplied by
        # following thing:
        multiplier = 1 - coefs[:, :order].sum(axis=1)

        # Recovering of switching mean values
        ar_means[multiplier == 0] = 0
        ar_means[multiplier != 0] = intercept_terms / multiplier

        # Again, approximation with averaging in case of non-switching parameter
        if self.switching_mean:
            new_params[self.parameters['mean']] = ar_means
        else:
            new_params[self.parameters['mean']] = ar_means.mean()

        return new_params

    def fit_em(self, start_params=None, transformed=True,
            em_iterations=3):
        """
        Fit EM algorithm one time from given starting parameters

        Parameters
        ----------
        start_params : array_like
            Starting parameters of the model. If `None`, the default is given
            by `start_params` property.
        transformed : bool
            Whether or not `start_params` is already transformed. Default is
            `True`.
        em_iterations : int
            Number of EM-iterations to perform. Default is 5.

        Returns
        -------
        params : array_like
            Result of applying EM-algorithm

        Notes
        -----
        Presented algorithm allows very fast jump from random initial
        parameters to high likelihood values, comparable with optimal.
        However, algorithm is approximate, and doesn't guarantee
        monotonous convergence.
        """

        # If `start_params` are not provided, use default ones
        if start_params is None:
            start_params = self.start_params
            transformed = True

        # Transformation, if needed
        if not transformed:
            start_params = self.transform_params(start_params)

        # EM-iterations
        params = start_params
        for i in range(em_iterations):
            params = self._em_iteration(params)

        return params

    def fit_em_with_random_starts(self, seed=1, em_optimizations=50,
            em_iterations=10, print_info=True, return_loglike=False):
        """
        Fit EM algorithm several times from random starts and choose the best
        parameters vector

        Parameters
        ----------
        seed : int, optional
            Random seed, used for starting parameters generation. Default is 1.
        em_optimizaitons : int
            The number of EM-algorithm sessions to perform. Default is 50.
        em_iterations : int
            The number of iterations during every EM-algorithm session. Default
            is 5.
        print_info : bool
            Whether to print the info about passed optimization sessions.
        return_loglike : bool
            If set, the tuple is returned, where the first element is a best
            guess, and the second is its loglikelihood. Otherwise, only the best
            guess is returned. Default is `False`

        Returns
        -------
        array_like, or tuple
        """

        # Set the seed
        np.random.seed(seed=seed)

        best_loglike = None
        best_params = None

        print_step = 5

        for i in range(em_optimizations):
            # Random start generating
            random_start_params = np.random.normal(
                    size=self.parameters.k_params)

            # Fitting of filtering can throw errors in case of invalid
            # parameters or input data
            try:
                params = self.fit_em(start_params=random_start_params,
                    transformed=False, em_iterations=em_iterations)
                loglike = self.loglike(params)
            except KeyboardInterrupt:
                raise
            except:
                continue

            # Update best results
            if (best_params is None or loglike > best_loglike) and \
                    not np.isnan(loglike):
                best_params = params
                best_loglike = loglike

            if print_info and (i + 1) % print_step == 0:
                print('{0} EM-optimizations passed'.format(i + 1))

        # Return results
        if return_loglike:
            return (best_params, best_loglike)
        else:
            return best_params

    def smooth(self, params, return_extended_probs=True, **kwargs):
        """
        Apply the Kim smoother to Markov switching autoregression

        Parameters
        ----------
        params : array_like
            Model parameters used for smoothing.
        return_extended_probs : bool
            Whether to return smoothed probabilities of underlying state space
            representation regimes or MS-AR regimes. `True` by default.
        **kwargs
            Optional keyword arguments, passed to `SwitchingMLEModel.smooth`
            method.

        Returns
        -------
        MSARSmootherResults

        See Also
        --------
        SwitchingMLEModel.smooth
        statsmodels.tsa.statespace.regime_switching.KimSmoother.smooth
        """

        results = None

        if not return_extended_probs:
            if 'regime_partition' in kwargs and \
                    kwargs['regime_partition'] is not None:
                raise ValueError('`regime_partition` argument can be applied ' \
                        'only for extended probabilities')

            results = MSARSmootherResults(self.ssm, self.k_ar_regimes)
            kwargs['results'] = results

        return super(MarkovAutoregression, self).smooth(params,  **kwargs)


class MSARSmootherResults(KimSmootherResults):
    """
    Smoother results for MS-AR model

    Parameters
    ----------
    model : KimFilter
        Kim filter, used by Markov switching autoregression model
    k_ar_regimes : int
        Number of MS-AR regimes

    Notes
    -----
    This class stores smoothed MS-AR regime probabilities, rather than smoothed
    state-space ones.

    See Also
    --------
    statsmodels.tsa.statespace.regime_switching.kim_smoother.KimSmootherResults
    """

    def __init__(self, model, k_ar_regimes):

        # Save the number of regimes
        self.k_ar_regimes = k_ar_regimes

        super(MSARSmootherResults, self).__init__(model)

    def update_smoother(self, smoother):

        k_ar_regimes = self.k_ar_regimes

        # Switching from logprobs to probs
        smoothed_regime_probs = np.exp(smoother.smoothed_regime_logprobs)

        # Regime, used in state space representation, encodes `order + 1`
        # consecutive MS-AR regimes using positional base-`k_ar_regimes` numeral
        # system:
        # S_{internal;t} = S_t + S_{t-1} * k + ... + S_{t-p} * k^p

        self.smoothed_ar_regime_probs = np.zeros((k_ar_regimes,
                smoothed_regime_probs.shape[1]),
                dtype=smoothed_regime_probs.dtype)

        # Collapsing probabilities
        # State space regimes with one residue are related to one AR regime
        for i in range(k_ar_regimes):
            self.smoothed_ar_regime_probs[i, :] = smoothed_regime_probs[
                    i::k_ar_regimes, :].sum(axis=0)

        # Switching from logprobs to probs
        smoothed_curr_and_next_regime_probs = \
                np.exp(smoother.smoothed_curr_and_next_regime_logprobs)

        self.smoothed_curr_and_next_ar_regime_probs = np.zeros((k_ar_regimes,
                k_ar_regimes, smoothed_curr_and_next_regime_probs.shape[2]),
                dtype=smoothed_curr_and_next_regime_probs.dtype)

        # Collapsing probabilities
        for i in range(k_ar_regimes):
            for j in range(k_ar_regimes):
                self.smoothed_curr_and_next_ar_regime_probs[i, j, :] = \
                        smoothed_curr_and_next_regime_probs[i::k_ar_regimes,
                        j::k_ar_regimes, :].sum(axis=(0, 1))

    @property
    def smoothed_regime_probs(self):
        """
        (array) Probabilities of given regime being active at given moment,
        conditional on all endogenous data.
        """
        return self.smoothed_ar_regime_probs

    @property
    def smoothed_curr_and_next_regime_probs(self):
        """
        (array) Joint probabilities of two given regimes being active at given
        moment and previous one, conditional on all endogenous data.
        """
        return self.smoothed_curr_and_next_ar_regime_probs
