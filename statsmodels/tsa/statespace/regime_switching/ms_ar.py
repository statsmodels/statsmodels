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

    coefs = np.zeros((k_regimes, exog.shape[0]), dtype=dtype)
    variances = np.zeros((k_regimes,), dtype=dtype)

    for regime in range(k_regimes):
        regression_exog = exog * \
                np.sqrt(smoothed_regime_probs[:, regime].reshape(-1, 1))
        regression_endog = endog * np.sqrt(smoothed_regime_probs[:, regime])

        coefs[regime, :], residuals = \
                np.linalg.lstsq(regression_exog, regression_endog)[:2]

        variances[regime] = \
                np.sum(residuals*smoothed_regime_probs[:, regime]) / \
                np.sum(smoothed_regime_probs[:, regime])

    regime_transition = np.sum(smoothed_curr_and_next_regime_probs, axis=0) / \
            np.sum(smoothed_regime_probs[:-1], axis=0).reshape(-1, 1)

    # to make matrix left-stochastic
    regime_transition = regime_transition.transpose()

    return (coefs, variances, regime_transition)


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
            switching_mean=True, switching_variance=True, exog=None,
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

        self.parameters = MarkovSwitchingParams(k_ar_regimes)
        if self.k_exog is not None:
            self.parameters['exog'] = [False] * self.k_exog
        self.parameters['autoregressive'] = self.switching_ar

        self.parameters['mean'] = [self.switching_mean]
        self.parameters['variance'] = [self.switching_variance]

        kwargs['k_posdef'] = 1

        super(MarkovAutoregression, self).__init__(k_ar_regimes**(order + 1),
                endog, order, **kwargs)

    def nonswitching_model_type(self):

        return _NonswitchingAutoregression

    def get_model_params(self, nonswitching_model_params):
        '''
        nonswitching_model_params = (mean exog_regression_coefs ar_coefs var)
        model_params = (exog_regression_coefs ar_coefs means vars)
        '''

        dtype = self.ssm.dtype
        order = self.order
        k_ar_regimes = self.k_ar_regimes

        model_params = np.zeros((self.parameters.k_params,), dtype=dtype)

        offset = 0

        model_params[self.parameters['mean']] = \
                nonswitching_model_params[offset]

        offset += 1

        if self.k_exog is not None:
            model_params[self.parameters['exog']] = \
                    nonswitching_model_params[offset:offset + self.k_exog]
            offset += self.k_exog

        # TODO: check if order of coefs is the same
        ar_coefs = nonswitching_model_params[offset:offset + order]
        for i in range(k_ar_regimes):
            model_params[self.parameters[i, 'autoregressive']] = ar_coefs
        offset += order

        model_params[self.parameters['variance']] = \
                nonswitching_model_params[offset]

        return model_params

    def get_nonswitching_model_params(self, model_params):
        '''
        nonswitching_model_params = (mean exog_regression_coefs ar_coefs var)
        model_params = (exog_regression_coefs ar_coefs means vars)
        '''

        dtype = self.ssm.dtype
        order = self.order
        k_ar_regimes = self.k_ar_regimes

        if self.k_exog is not None:
            exog_regression_coefs = \
                    model_params[self.parameters['exog']].tolist()
        else:
            exog_regression_coefs = []

        ar_coefs = np.zeros((k_ar_regimes, order), dtype=dtype)

        for i in range(k_ar_regimes):
            ar_coefs[i] = model_params[self.parameters[i, 'autoregressive']]

        ar_coefs = ar_coefs.mean(axis=0).tolist()

        mean = [model_params[self.parameters['mean']].mean()]
        variance = [model_params[self.parameters['variance']].mean()]

        return np.array(mean + exog_regression_coefs + ar_coefs + var,
                dtype=dtype)

    def transform_model_params(self, unconstrained_model_params):
        '''
        model_params = (exog_regression_coefs ar_coefs means vars)
        '''

        k_ar_regimes = self.k_ar_regimes

        constrained_model_params = super(MarkovAutoregression,
                self).transform_model_params(unconstrained_model_params)

        # Keeping variance positive
        s = self.parameters['variance']
        constrained_model_params[s] = constrained_model_params[s]**2

        return constrained_model_params

    def untransform_model_params(self, constrained_model_params):
        '''
        model_params = (exog_regression_coefs ar_coefs means vars)
        '''

        k_ar_regimes = self.k_ar_regimes

        unconstrained_model_params = super(MarkovAutoregression,
                self).untransform_model_params(constrained_model_params)

        # Keeping variance positive
        s = self.parameters['variance']
        unconstrained_model_params[s] = unconstrained_model_params[s]**0.5

        return unconstrained_model_params

    def get_ar_coef_regimes(self, regime_index):
        '''
        get ar coeficient regimes
        from extended regime (tuple (S_t, S_{t-1}, ..., S_{t-order})) index
        '''

        k_ar_regimes = self.k_ar_regimes

        regimes_suffix = regime_index // k_ar_regimes
            for ar_coef_index in range(order):
                yield regimes_suffix % k_ar_regimes
                regimes_suffix /= k_ar_regimes

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

        ar_regime_transition, model_params = self.get_explicit_params(params,
                k_regimes=self.k_ar_regimes)

        # Extended transition_matrix for state space representation
        # Every regime here is a tuple (S_t, S_{t-1}, ..., S_{t-order})
        # Index of the regime in transition_matrix is calculated as
        # S_t + S_{t-1}*k_ar_regimes + ... + S_{t-order}*k_ar_regimes**order

        regime_transition = np.zeros((k_regimes, k_regimes),
                dtype=dtype)

        for prev_regime_index in range(k_regimes):
            prev_ar_regime = prev_regime_index % k_ar_regimes
            curr_regime_index_without_curr_ar_regime = k_ar_regimes * \
                    (prev_regime_index % (k_ar_regimes**order))

            for curr_ar_regime in range(k_ar_regimes):
                curr_regime_index = \
                        curr_regime_index_without_curr_ar_regime + \
                        curr_ar_regime

                regime_transition[curr_regime_index, prev_regime_index] = \
                        ar_regime_transition[curr_ar_regime, prev_ar_regime]

        self['regime_transition'] = regime_transition

        if self.k_exog is not None:
            exog_intercept_term = self.exog.dot(
                    model_params[self.parameters['exog']].reshape(-1, 1)
                    ).reshape((1, -1))

        # Ar coefs

        ar_coefs = np.zeros((k_ar_regimes, order), dtype=dtype)

        for i in range(k_ar_regimes):
            ar_coefs[i] = model_params[self.parameters[i, 'autoregressive']]

        transition = np.zeros((k_regimes, order, order, 1), dtype=dtype)

        for regime_index in range(k_regimes):
            for ar_coef_index, ar_regime in zip(range(order),
                    self.get_ar_coef_regimes(regime_index)):
                transition[regime_index, ar_coef_index, 0, 0] = \
                        ar_coefs[ar_coef_index, ar_regime]

        self['transition'] = transition

        # Switching means

        if self.k_exog is not None:
            obs_intercept = np.zeros((k_regimes,) + \
                    exog_intercept_term.shape, dtype=dtype)
            for regime_index in range(k_regimes):
                obs_intercept[regime_index, :, :] = exog_intercept_term
                curr_ar_regime = regime_index % k_ar_regimes
                obs_intercept[regime_index, :, :] += \
                        model_params[self.parameters[curr_ar_regime, 'mean']]
        else:
            obs_intercept = np.zeros((k_regimes, 1, 1), dtype=dtype)
            for regime_index in range(k_regimes):
                curr_ar_regime = regime_index % k_ar_regimes
                obs_intercept[regime_index, 0, 0] = \
                        model_params[self.parameters[curr_ar_regime, 'mean']]

        self['obs_intercept'] = obs_intercept

        # Switching variances

        state_cov = np.zeros((k_regimes, 1, 1, 1), dtype=dtype)
        state_cov[:, 0, 0, 0] = model_params[self.parameters['variance']]

        self['state_cov'] = state_cov

        # Other matrices

        self['selection'] = np.zeros((order, 1, 1), dtype=dtype)
        self['selection'][0, 0, 0] = 1

        self['design'] = np.zeros((1, order, 1), dtype=dtype)
        self['design'][0, 0, 0] = 1

        # obs_cov and state_intercept is zero by default

    def _em_iteration(self, params, **kwargs):
        '''
        params = (transition_matrix(not extended) exog_regression_coefs
        ar_coefs means vars)

        Kim-Nelson p.77
        '''

        dtype = self.ssm.dtype
        order = self.order
        k_regimes = self.k_regimes
        k_ar_regimes = self.k_ar_regimes

        # obtaining smoothed probs

        self.update(params, **kwargs)

        smoothed_regime_probs, smoothed_curr_and_next_regime_probs = \
                sels.ssm.get_smoothed_regime_probs(**kwargs)

        # preparing data for regression em iteration

        markov_regression_endog = self.endog.ravel()

        markov_regression_exog_dim = order + 1

        if self.k_exog is not None:
            markov_regression_exog_dim += self.k_exog

        markov_regression_exog = np.zeros((markov_regression_endog.shape[0],
                markov_regression_exog_dim), dtype=dtype)

        for i in range(order):
            markov_regression_exog[i:, i] = markov_regression_endog[:i]

        # Adding intercept exog value. What if self.exog already contains one?
        markov_regression_exog[:, order] = 1

        if self.k_exog is not None:
            markov_regression_exog[:, order + 1:] = self.exog

        # regression em iteration

        coefs, variances, regime_transition = \
                _em_iteration_for_markov_regression(dtype, k_regimes,
                markov_regression_endog, markov_regression_exog,
                smoothed_regime_probs, smoothed_curr_and_next_regime_probs)

        model_params = np.zeros((self.parameters.k_params,), dtype=dtype)

        ar_regime_transition = np.zeros((k_ar_regimes, k_ar_regimes),
                dtype=dtype)

        # ar regime transition recovery

        for prev_regime_index in range(k_regimes):
            prev_ar_regime = prev_regime_index % k_ar_regimes
            curr_regime_index_without_curr_ar_regime = \
                    prev_regime_index // k_ar_regimes
            for curr_ar_regime in range(k_ar_regimes):
                curr_regime_index = \
                        curr_regime_index_without_curr_ar_regime + \
                        curr_ar_regime

                # For debug; remove after testing
                if ar_regime_transition[curr_ar_regime, prev_ar_regime] != 0 and \
                        ar_regime_transition[curr_ar_regime, prev_ar_regime] != \
                        regime_transition[curr_regime_index, prev_regime_index]:
                    raise RuntimeError(
                            'EM-algorithm bug: {0} prob is not equal to {1}'.format(
                            ar_regime_transition[curr_ar_regime, prev_ar_regime],
                            regime_transition[curr_regime_index, prev_regime_index]))

                ar_regime_transition[curr_ar_regime, prev_ar_regime] = \
                        regime_transition[curr_regime_index, prev_regime_index]

        # variance recovery

        for regime_index in range(k_regimes):

            ar_regime = regime_index % k_ar_regimes
            s = self.parameters[ar_regime, 'variance']

            # For debug; remove after testing
            if model_params[s] != 0 and \
                    model_params[s] != variances[regime_index]:
                raise RuntimeError(
                        'EM-algorithm bug: {0} variance is not equal to {1}'.format(
                        model_params[s], variances[regime_index]))

            model_params[s] = variances[regime_index]

        # ar coefs recovery

        for regime_index in range(k_regimes):

            ar_coefs = np.zeros((k_ar_regimes, order), dtype=dtype)

            for ar_coef_index, ar_regime in zip(range(order),
                    self.get_ar_coef_regimes(regime_index)):

                # For debug; remove after testing
                if ar_coefs[ar_regime, ar_coef_index] != 0 and \
                        ar_coefs[ar_regime, ar_coef_index] != \
                        coefs[regime_index, ar_coef_index]:
                    raise RuntimeError(
                            'EM-algorithm bug: {0} ar coef is not equal to {1}'.format(
                            ar_coefs[ar_regime, ar_coef_index],
                            coefs[regime_index, ar_coef_index]))

                ar_coefs[ar_regime, ar_coef_index] = coefs[regime_index, ar_coef_index]

        for i in range(k_ar_regimes):
            model_params[i, 'autoregressive'] = ar_coefs[i, :]

        # mean recovery
        # Mean values for k_ar_regimes can be obtained via system of linear
        # equations solving, since intercept is a linear combination of them

        system_coeficients = np.zeros((k_regimes, k_ar_regimes),
                dtype=dtype)

        intercept_terms = coefs[:, order]

        for regime_index in range(k_regimes):

            curr_ar_regime = regime_index % k_ar_regimes

            system_coeficients[regime_index, curr_ar_regime] = 1

            for ar_coef_index, ar_regime in zip(range(order),
                    self.get_ar_coef_regimes(regime_index)):
                system_coeficients[regime_index, ar_regime] -= \
                        ar_coefs[ar_regime, ac_coef_index]

        ar_means, residuals = np.linalg.lstsq(system_coeficients,
                intercept_terms)[:2]

        # For debug; remove after testing
        if any(residuals != 0):
            raise RuntimeError(
                    'EM-algorithm bug: {0}, {1} system has no solution'.format(
                    system_coeficients, intercept_terms))

        if self.switching_mean:
            model_params['mean'] = ar_means
        else:
            # ar means should be all the same
            model_params['mean'] = ar_means.mean()

        return self.get_params_vector(ar_regime_transition, model_params)

    #def fit():

    #    pass
