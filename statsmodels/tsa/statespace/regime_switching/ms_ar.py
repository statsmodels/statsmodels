import numpy as np
from .tools import RegimePartition
from statsmodels.tsa.statespace.regime_switching.rs_mlemodel import \
        RegimeSwitchingMLEModel
from statsmodels.tsa.statespace.regime_switching.tools import \
        MarkovSwitchingParams
from statsmodels.tsa.statespace.sarimax import SARIMAX

def _em_iteration_for_markov_regression(dtype, k_regimes, endog, exog,
        smoothed_regime_probs, smoothed_curr_and_next_regime_probs):
    '''
    Kim-Nelson p. 77
    '''

    coefs = np.zeros((k_regimes, exog.shape[1]), dtype=dtype)
    variances = np.zeros((k_regimes,), dtype=dtype)

    for regime in range(k_regimes):
        regression_exog = exog * \
                np.sqrt(smoothed_regime_probs[:, regime].reshape(-1, 1))
        regression_endog = endog * np.sqrt(smoothed_regime_probs[:, regime])

        # may raise an Exception
        coefs[regime, :] = np.linalg.lstsq(regression_exog, regression_endog)[0]

        sqr_residuals = (endog - \
                exog.dot(coefs[regime, :].reshape(-1, 1)).ravel())**2

        marginal_regime_prob_sum = np.sum(smoothed_regime_probs[:, regime])

        if marginal_regime_prob_sum != 0:
            variances[regime] = \
                    np.sum(sqr_residuals * smoothed_regime_probs[:, regime]) / \
                    marginal_regime_prob_sum
        else:
            # any value would be alright?
            variances[regime] = 1

    joint_prob_sum = np.sum(smoothed_curr_and_next_regime_probs, axis=0)

    marginal_prob_sum = np.sum(smoothed_regime_probs[:-1], axis=0)

    ar_regime_transition = np.zeros((k_regimes, k_regimes), dtype=dtype)

    ar_regime_transition[marginal_prob_sum == 0, :] = 1.0/k_regimes

    ar_regime_transition[marginal_prob_sum != 0, :] = \
            joint_prob_sum[marginal_prob_sum != 0] / \
            marginal_prob_sum[marginal_prob_sum != 0].reshape(-1, 1)

    # to make matrix left-stochastic
    ar_regime_transition = ar_regime_transition.transpose()

    return (coefs, variances, ar_regime_transition)


class _NonswitchingAutoregression(SARIMAX):
    '''
    SARIMAX wrapper, simplifying it to AR(p) model
    '''

    def __init__(self, endog, order, exog=None, dtype=np.float64, **kwargs):

        mle_regression = True

        intercept = np.ones((endog.shape[1], 1))

        if exog is not None:
            exog = np.hstack((intercept, exog))
        else:
            exog = intercept

        mle_regression = True

        super(_NonswitchingAutoregression, self).init(endog, exog=exog,
                order=(order, 0, 0), mle_regression=mle_regression,
                enforce_stationarity=False)


class MarkovAutoregression(RegimeSwitchingMLEModel):
    '''
    Markov switching autoregressive model
    '''

    def __init__(self, k_ar_regimes, order, endog, switching_ar=False,
            switching_mean=False, switching_variance=False, exog=None,
            k_exog=None, **kwargs):
        '''
        order - the order of autoregressive lag polynomial
        k_ar_regimes - the number of regimes for AR model (not for a state space
        representation)
        '''

        self.order = order
        self.k_ar_regimes = k_ar_regimes

        if k_exog is not None:
            self.k_exog = k_exog
        elif exog is not None:
            exog = np.asarray(exog)
            if exog.ndim == 1:
                exog = exog.reshape(-1, 1)
            self.k_exog = exog.shape[1]
        else:
            self.k_exog = None

        endog = endog.ravel()

        self.endog_head = endog[:order]

        endog = endog[order:]

        if isinstance(switching_ar, bool):
            self.switching_ar = [switching_ar] * order
        else:
            if len(switching_ar) != order:
                raise ValueError('Invalid iterable passed to `switching_ar`.')
            self.switching_ar = switching_ar

        if k_ar_regimes == 1 or not (switching_mean or switching_variance or \
                any(self.switching_ar)):
            raise ValueError('Consider using SARIMAX model')

        self.switching_mean = switching_mean
        self.switching_variance = switching_variance

        kwargs['k_posdef'] = 1

        super(MarkovAutoregression, self).__init__(k_ar_regimes**(order + 1),
                endog, order, param_k_regimes=k_ar_regimes, exog=exog, **kwargs)

        if self.k_exog is not None:
            self.parameters['exog'] = [False] * self.k_exog
        self.parameters['autoregressive'] = self.switching_ar

        self.parameters['mean'] = [self.switching_mean]
        self.parameters['variance'] = [self.switching_variance]

    def nonswitching_model_type(self):

        return _NonswitchingAutoregression

    def update_params(self, params, nonswitching_params):
        '''
        nonswitching_params = (mean exog_regression_coefs ar_coefs var)
        '''

        dtype = self.ssm.dtype
        order = self.order
        k_ar_regimes = self.k_ar_regimes

        offset = 0

        params[self.parameters['mean']] = nonswitching_params[offset]

        offset += 1

        if self.k_exog is not None:
            params[self.parameters['exog']] = \
                    nonswitching_params[offset:offset + self.k_exog]
            offset += self.k_exog

        # TODO: check if order of coefs is the same
        ar_coefs = nonswitching_params[offset:offset + order]
        for i in range(k_ar_regimes):
            params[self.parameters[i, 'autoregressive']] = ar_coefs
        offset += order

        params[self.parameters['variance']] = nonswitching_params[offset]

        return params

    def get_nonswitching_params(self, params):
        '''
        nonswitching_params = (mean exog_regression_coefs ar_coefs var)
        '''

        dtype = self.ssm.dtype
        order = self.order
        k_ar_regimes = self.k_ar_regimes

        if self.k_exog is not None:
            exog_regression_coefs = \
                    params[self.parameters['exog']].tolist()
        else:
            exog_regression_coefs = []

        ar_coefs = np.zeros((k_ar_regimes, order), dtype=dtype)

        for i in range(k_ar_regimes):
            ar_coefs[i] = params[self.parameters[i, 'autoregressive']]

        ar_coefs = ar_coefs.mean(axis=0).tolist()

        mean = [params[self.parameters['mean']].mean()]
        variance = [params[self.parameters['variance']].mean()]

        return np.array(mean + exog_regression_coefs + ar_coefs + var,
                dtype=dtype)

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

    def normalize_params(self, params, transformed=True):

        dtype = self.ssm.dtype
        k_ar_regimes = self.k_ar_regimes
        order = self.order

        if not transformed:
            params = self.transform_params(params)

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

        regime_permutation = sorted(range(k_ar_regimes),
                key=lambda regime:regime_sort_keys[regime])

        ar_regime_transition = self._get_param_regime_transition(params)
        new_ar_regime_transition = np.zeros((k_ar_regimes, k_ar_regimes),
                dtype=dtype)

        for i in range(k_ar_regimes):
            for j in range(k_ar_regimes):
                new_ar_regime_transition[i, j] = \
                        ar_regime_transition[regime_permutation[i],
                        regime_permutation[j]]

        new_params = np.zeros((self.parameters.k_params,), dtype=dtype)

        self._set_param_regime_transition(new_params, new_ar_regime_transition)

        for i in range(k_ar_regimes):
            new_params[self.parameters[i]] = \
                    params[self.parameters[regime_permutation[i]]]

        if not transformed:
            new_params = self.untransform_params(new_params)

        return new_params

    def get_ar_mean_regimes(self, regime_index):
        '''
        get ar mean regimes
        from extended regime (tuple (S_t, S_{t-1}, ..., S_{t-order})) index
        '''

        order = self.order

        k_ar_regimes = self.k_ar_regimes

        regimes_suffix = regime_index // k_ar_regimes
        for ar_coef_index in range(order):
            yield regimes_suffix % k_ar_regimes
            regimes_suffix /= k_ar_regimes

    def _iterate_regimes(self):

        k_regimes = self.k_regimes
        k_ar_regimes = self.k_ar_regimes
        order = self.order

        for prev_regime_index in range(k_regimes):
            prev_ar_regime = prev_regime_index % k_ar_regimes
            curr_regime_index_without_curr_ar_regime = k_ar_regimes * \
                    (prev_regime_index % (k_ar_regimes**order))
            for curr_ar_regime in range(k_ar_regimes):
                curr_regime_index = \
                        curr_regime_index_without_curr_ar_regime + \
                        curr_ar_regime

                yield (prev_regime_index, curr_regime_index, prev_ar_regime,
                        curr_ar_regime)

    def update(self, params, **kwargs):
        '''
        params = (transition_matrix(not extended) exog_regression_coefs
        ar_coefs means vars)

        Durbin-Koopman, page 46
        '''

        order = self.order
        k_regimes = self.k_regimes
        k_ar_regimes = self.k_ar_regimes
        dtype = self.ssm.dtype

        params = super(MarkovAutoregression, self).update(params, **kwargs)

        ar_regime_transition = self._get_param_regime_transition(params)

        # Extended transition_matrix for state space representation
        # Every regime here is a tuple (S_t, S_{t-1}, ..., S_{t-order})
        # Index of the regime in transition_matrix is calculated as
        # S_t + S_{t-1}*k_ar_regimes + ... + S_{t-order}*k_ar_regimes**order

        regime_transition = np.zeros((k_regimes, k_regimes),
                dtype=dtype)

        for prev_regime_index, curr_regime_index, prev_ar_regime, \
                        curr_ar_regime in self._iterate_regimes():
            regime_transition[curr_regime_index, prev_regime_index] = \
                    ar_regime_transition[curr_ar_regime, prev_ar_regime]

        self['regime_transition'] = regime_transition

        # exog coefs

        if self.k_exog is not None:
            exog_intercept_term = self.exog.dot(
                    params[self.parameters['exog']].reshape(-1, 1)
                    ).reshape((1, -1))

        # Ar coefs

        ar_coefs = np.zeros((k_ar_regimes, order), dtype=dtype)

        for i in range(k_ar_regimes):
            ar_coefs[i] = params[self.parameters[i, 'autoregressive']]

        transition = np.zeros((k_regimes, order, order, 1), dtype=dtype)

        for regime_index in range(k_regimes):
            curr_ar_regime = regime_index % k_ar_regimes
            transition[regime_index, :-1, 1:, 0] = np.identity(order - 1)
            transition[regime_index, :, 0, 0] = ar_coefs[curr_ar_regime, :]

        self['transition'] = transition

        # Switching means

        if self.k_exog is not None:
            obs_intercept = np.zeros((k_regimes,) + \
                    exog_intercept_term.shape, dtype=dtype)
            for regime_index in range(k_regimes):
                obs_intercept[regime_index, :, :] = exog_intercept_term
                curr_ar_regime = regime_index % k_ar_regimes
                obs_intercept[regime_index, :, :] += \
                        params[self.parameters[curr_ar_regime, 'mean']]
        else:
            obs_intercept = np.zeros((k_regimes, 1, 1), dtype=dtype)
            for regime_index in range(k_regimes):
                curr_ar_regime = regime_index % k_ar_regimes
                obs_intercept[regime_index, 0, 0] = \
                        params[self.parameters[curr_ar_regime, 'mean']]

        self['obs_intercept'] = obs_intercept

        # Switching variances

        state_cov = np.zeros((k_regimes, 1, 1, 1), dtype=dtype)
        for i in range(k_regimes):
            state_cov[i, 0, 0, 0] = params[self.parameters[i % k_ar_regimes,
                    'variance']]

        self['state_cov'] = state_cov

        # Other matrices

        selection = np.zeros((order, 1, 1), dtype=dtype)
        selection[0, 0, 0] = 1

        self['selection'] = selection

        design = np.zeros((1, order, 1), dtype=dtype)
        design[0, 0, 0] = 1

        self['design'] = design

        # obs_cov and state_intercept is zero by default

        # initialization

        # zero state is known a priori
        initial_state_cov = np.zeros((order, order), dtype=dtype)

        initial_state = np.zeros((k_regimes, order), dtype=dtype)

        for regime_index in range(k_regimes):

            curr_ar_regime = regime_index % k_ar_regimes

            curr_regime_ar_coefs = ar_coefs[curr_ar_regime, :]
            curr_regime_ar_means = np.zeros((order, ), dtype=dtype)

            for ar_lag_index, ar_regime in zip(range(order),
                    self.get_ar_mean_regimes(regime_index)):
                curr_regime_ar_means[ar_lag_index] = \
                        params[self.parameters[ar_regime, 'mean']]

            initial_state[regime_index, 0] = self.endog_head[-1] - \
                    curr_regime_ar_means[0]

            for i in range(1, order):
                for j in range(i, order):
                    initial_state[regime_index, i] += \
                            curr_regime_ar_coefs[j] * \
                            (self.endog_head[order - 2 + i - j] - \
                            curr_regime_ar_means[j])

        self.initialize_known(initial_state, initial_state_cov)

    def _em_iteration(self, params):

        dtype = self.ssm.dtype
        order = self.order
        k_regimes = self.k_regimes
        k_ar_regimes = self.k_ar_regimes

        # obtaining smoothed probs

        smoothed_regime_probs, smoothed_curr_and_next_regime_probs = \
                self.get_smoothed_regime_probs(params)

        # preparing data for regression em iteration

        markov_regression_endog = self.endog.ravel()

        markov_regression_exog_dim = order + 1

        if self.k_exog is not None:
            markov_regression_exog_dim += self.k_exog

        markov_regression_exog = np.zeros((markov_regression_endog.shape[0],
                markov_regression_exog_dim), dtype=dtype)

        for i in range(order):
            markov_regression_exog[:i + 1, i] = self.endog_head[-i - 1:]
            markov_regression_exog[i + 1:, i] = markov_regression_endog[:-i - 1]

        # Adding intercept exog value. What if self.exog already contains one?
        markov_regression_exog[:, order] = 1

        if self.k_exog is not None:
            markov_regression_exog[:, order + 1:] = self.exog

        # regression em iteration

        coefs, variances, ar_regime_transition = \
                _em_iteration_for_markov_regression(dtype, k_ar_regimes,
                markov_regression_endog, markov_regression_exog,
                smoothed_regime_probs, smoothed_curr_and_next_regime_probs)

        new_params = np.zeros((self.parameters.k_params,), dtype=dtype)

        # ar regime transition recovery

        self._set_param_regime_transition(new_params, ar_regime_transition)

        # ar coefs recovery

        ar_coefs = np.zeros((k_ar_regimes, order), dtype=dtype)
        ar_coefs[:, :] = coefs[:, :order]

        ar_coefs[:, ~np.array(self.switching_ar)] = \
                ar_coefs[:, ~np.array(self.switching_ar)].mean(axis=0)

        for i in range(k_ar_regimes):
            new_params[self.parameters[i, 'autoregressive']] = ar_coefs[i, :]

        # variance recovery

        if self.switching_variance:
            new_params[self.parameters['variance']] = variances
        else:
            new_params[self.parameters['variance']] = variances.mean()

        # mean recovery

        intercept_terms = coefs[:, order]

        ar_means = np.zeros((k_ar_regimes,), dtype=dtype)

        #ar_means[:] = intercept_terms

        multiplier = 1 - coefs[:, :order].sum(axis=1)
        ar_means[multiplier == 0] = 0
        ar_means[multiplier != 0] = intercept_terms / multiplier

        if self.switching_mean:
            new_params[self.parameters['mean']] = ar_means
        else:
            new_params[self.parameters['mean']] = ar_means.mean()

        return new_params

    def fit_em(self, start_params=None, transformed=True,
            em_iterations=3):
        '''
        Fits EM algorithm one time with given start params
        '''

        if start_params is None:
            start_params = self.start_params
            transformed = True

        if not transformed:
            start_params = self.transform_params(start_params)

        params = start_params
        for i in range(em_iterations):
            params = self._em_iteration(params)

        return params

    def fit_em_with_random_starts(self, seed=1, em_optimizations=50,
            em_iterations=10, return_loglike=False):
        '''
        Fits EM algorithm several times with random starts and chooses the
        best params vector
        random_scale - scale parameters of numpy.random.normal
        '''

        np.random.seed(seed=seed)

        best_loglike = None
        best_params = None

        for i in range(em_optimizations):

            random_start_params = np.random.normal(
                    size=self.parameters.k_params)

            # Params can be invalid at some point
            try:
                params = self.fit_em(start_params=random_start_params,
                    transformed=False, em_iterations=em_iterations)
                loglike = self.loglike(params)
            except:
                continue

            if best_params is None or loglike > best_loglike:
                best_params = params
                best_loglike = loglike

        if return_loglike:
            return (best_params, best_loglike)
        else:
            return best_params

    def get_smoothed_regime_probs(self, params, return_extended_probs=False,
            **kwargs):
        '''
        return_extended_probs - returns state space smoothed regime probs.
        this is overridden, because this method returns smoothed AR
        regimes, rather then state space regimes.
        '''

        k_regimes = self.k_regimes
        k_ar_regimes = self.k_ar_regimes
        order = self.order
        nobs = self.nobs
        dtype = self.ssm.dtype

        # these are smoothed state space regimes
        smoothed_regime_probs, smoothed_curr_and_next_regime_probs = \
                super(MarkovAutoregression,
                self).get_smoothed_regime_probs(params, **kwargs)

        if return_extended_probs:
            return smoothed_regime_probs, smoothed_curr_and_next_regime_probs

        if 'regime_partition' in kwargs and kwargs['regime_partition'] is None:
            raise ValueError('regime_partition argument can be applied only ' \
                    'for extended probabilities')

        smoothed_ar_regime_probs = np.zeros((nobs, k_ar_regimes), dtype=dtype)

        for i in range(k_ar_regimes):
            smoothed_ar_regime_probs[:, i] = smoothed_regime_probs[:,
                    i::k_ar_regimes].sum(axis=1)

        smoothed_curr_and_next_ar_regime_probs = np.zeros((nobs - 1,
                k_ar_regimes, k_ar_regimes), dtype=dtype)

        for i in range(k_ar_regimes):
            for j in range(k_ar_regimes):
                smoothed_curr_and_next_ar_regime_probs[:, i, j] = \
                        smoothed_curr_and_next_regime_probs[:,
                        i::k_ar_regimes, j::k_ar_regimes].sum(axis=(1, 2))

        return smoothed_ar_regime_probs, smoothed_curr_and_next_ar_regime_probs
