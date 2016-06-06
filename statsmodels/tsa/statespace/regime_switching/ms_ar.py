from statsmodels.tsa.statespace.regime_switching.rs_mlemodel import \
        RegimeSwitchingMLEModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

        offset = 0

        if self.switching_mean:
            means = [nonswitching_model_params[offset]] * k_ar_regimes
        else:
            means = [nonswitching_model_params[offset]]

        offset += 1

        if self.k_exog is not None:
            exog_regression_coefs = \
                    nonswitching_model_params[offset:offset + \
                    self.k_exog].tolist()
            offset += self.k_exog
        else:
            exog_regression_coefs = []

        ar_coefs = []

        # TODO: check if order of coefs is the same
        for ar_coef_index, is_switching in zip(range(order),
                self.switching_ar):
            if is_switching:
                ar_coefs += \
                        [nonswitching_model_params[offset]] * k_ar_regimes
            else:
                ar_coefs += [nonswitching_model_params[offset]]
            offset += 1

        if self.switching_variance:
            variances = [nonswitching_model_params[offset]] * k_ar_regimes
        else:
            variances = [nonswitching_model_params[offset]]

        return np.array(means + exog_regression_coefs + ar_coefs + \
                variances, dtype=dtype)

    def get_nonswitching_model_params(self, model_params):
        '''
        nonswitching_model_params = (mean exog_regression_coefs ar_coefs var)
        model_params = (exog_regression_coefs ar_coefs means vars)
        '''

        dtype = self.ssm.dtype
        order = self.order
        k_ar_regimes = self.k_ar_regimes

        offset = 0

        if self.k_exog is not None:
            exog_regression_coefs = model_params[offset:offset + \
                    self.k_exog].tolist()
            offset += self.k_exog
        else:
            exog_regression_coefs = []

        ar_coefs = []

        for ar_coef_index, is_switching in zip(range(order),
                self.switching_ar):
            if is_switching:
                ar_coefs += [model_params[offset:offset + k_ar_regimes].mean()]
                offset += k_ar_regimes
            else:
                ar_coefs += [model_params[offset]]
                offset += 1

        if self.switching_mean:
            mean = [model_params[offset:offset + k_ar_regimes].mean()]
            offset += k_ar_regimes
        else:
            mean = [model_params[offset]]
            offset += 1

        if self.switching_variance:
            var = [model_params[offset:offset + k_ar_regimes].mean()]
            # offset += k_ar_regimes
        else:
            var = [model_params[offset]]

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
        if self.switching_variance:
            constrained_model_params[-1] = constrained_model_params[-1]**2
        else:
            constrained_model_params[-k_ar_regimes] = \
                    constrained_model_params[-k_ar_regimes]**2

        return constrained_model_params

    def untransform_model_params(self, constrained_model_params):
        '''
        model_params = (exog_regression_coefs ar_coefs means vars)
        '''

        k_ar_regimes = self.k_ar_regimes

        unconstrained_model_params = super(MarkovAutoregression,
                self).untransform_model_params(constrained_model_params)

        # Keeping variance positive
        if self.switching_variance:
            unconstrained_model_params[-1] = \
                    unconstrained_model_params[-1]**0.5
        else:
            unconstrained_model_params[-k_ar_regimes] = \
                    unconstrained_model_params[-k_ar_regimes]**0.5

        return unconstrained_model_params

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

        offset = 0

        if self.k_exog is not None:
            self['obs_intercept'] = self.exog.dot(
                    model_params[:self.k_exog].reshape(-1, 1)
                    ).reshape((1, 1, -1))
            offset += self.k_exog

        # Ar coefs

        ar_coefs = np.zeros((order, k_ar_regimes), dtype=dtype)

        for ar_coef_index in range(order):
            if self.switching_ar[ar_coef_index]:
                ar_coefs[ar_coef_index, :] = \
                        model_params[offset:offset + k_ar_regimes]
                offset += k_ar_regimes
            else:
                ar_coefs[ar_coef_index, :] = model_params[offset]
                model_param_offset += 1

        transition = np.zeros((k_regimes, order, order, 1), dtype=dtype)

        for regime_index in range(k_regimes):
            regimes_suffix = regime_index // k_ar_regimes
            for ar_coef_index in range(order):
                ar_regime = regimes_suffix % k_ar_regimes
                regimes_suffix /= k_ar_regimes
                transition[regime_index, ar_coef_index, 0, 0] = \
                        ar_coefs[ar_coef_index, ar_regime]

        self['transition'] = transition

        # Switching means

        if self.switching_mean:
            ar_means = model_params[offset:offset + k_ar_regimes]
            offset += k_ar_regimes
        else:
            ar_means = np.zeros(k_ar_regimes, dtype=dtype)
            ar_means[:] = model_params[offset]
            offset += 1

        state_intercept = np.zeros((k_regimes, order, 1), dtype=dtype)

        for regime_index in range(k_regimes):
            curr_ar_regime = regime_index % k_ar_regimes
            regimes_suffix = regime_index // k_ar_regimes
            state_intercept[regime_index, 0, 0] = ar_means[curr_ar_regime]
            for ar_coef_index in range(order):
                ar_regime = regimes_suffix % k_ar_regimes
                regimes_suffix /= k_ar_regimes
                state_intercept[regime_index, 0, 0] -= \
                        ar_coefs[ar_coef_index, ar_regime] * \
                        ar_means[ar_regime]

        self['state_intercept'] = state_intercept

        # Switching variances

        state_cov = np.zeros((k_regimes, 1, 1, 1), dtype=dtype)
        if self.switching_variance:
            state_cov[:, 0, 0, 0] = model_params[offset:]
        else:
            state_cov[:, 0, 0, 0] = model_params[-1]

        self['state_cov'] = state_cov

        # Other matrices

        self['selection'] = np.zeros((order, 1, 1), dtype=dtype)
        self['selection'][0, 0, 0] = 1

        self['design'] = np.zeros((1, order, 1), dtype=dtype)
        self['design'][0, 0, 0] = 1

        # obs_cov is zero by default
