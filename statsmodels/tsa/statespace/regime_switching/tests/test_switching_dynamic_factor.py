import numpy as np
from numpy.testing import assert_allclose
from statsmodels.tsa.statespace.regime_switching.switching_dynamic_factor \
        import DynamicFactorWithFactorIntercept, SwitchingDynamicFactor
from .results import results_kim_yoo1995


class KimYoo1995NonswitchingModel(DynamicFactorWithFactorIntercept):
    '''
    This is dynamic factor model with some restrictions on parameters.
    See http://econ.korea.ac.kr/~cjkim/MARKOV/programs/sw_ms.opt.
    '''

    def __init__(self, endog, k_factors, factor_order, **kwargs):

        super(KimYoo1995NonswitchingModel, self).__init__(endog, k_factors,
                factor_order, **kwargs)

        k_endog = self.k_endog
        error_order = self.error_order

        offset = 0
        self._gamma_idx = np.s_[offset:offset + k_endog + k_factors - 1]
        offset += k_endog + k_factors - 1
        self._phi_idx = np.s_[offset:offset + error_order]
        offset += error_order
        self._psi_idx = np.s_[offset:offset + k_endog * error_order]
        offset += k_endog * error_order
        self._sigma_idx = np.s_[offset:offset + k_endog]
        offset += k_endog
        self._mu_idx = np.s_[offset]
        self._params_without_intercept_idx = np.s_[:offset]
        self._kimyoo_k_params_without_intercept = offset
        offset += 1

        self.kimyoo_k_params = offset

    def _get_dynamic_factor_params(self, params_without_intercept):
        '''
        params_without_intercept - just a prefix of parameters vector, since
        intercept is the last value
        '''

        dtype = self.ssm.dtype
        k_endog = self.k_endog
        k_factors = self.k_factors
        factor_order = self.factor_order
        error_order = self.error_order

        # k_params property is inherited from DynamicFactor class
        dynamic_factor_params = np.zeros((self.k_params,), dtype=dtype)

        # 1. Factor loadings

        factor_loadings_matrix = np.zeros((k_endog, k_factors), dtype=dtype)

        gammas = params_without_intercept[self._gamma_idx]

        factor_loadings_matrix[:, 0] = gammas[:k_endog]
        factor_loadings_matrix[-1, 1:] = gammas[k_endog:]

        dynamic_factor_params[self._params_loadings] = \
                factor_loadings_matrix.ravel()

        # 2. Error covs

        dynamic_factor_params[self._params_error_cov] = \
                params_without_intercept[self._sigma_idx]

        # 3. Factor transition

        phi = params_without_intercept[self._phi_idx]


        # `factor_order` == 1, so this essentially is a square matrix
        factor_transition_params = np.zeros((k_factors,
                k_factors * factor_order), dtype=dtype)

        # TODO: check order of parameters

        factor_transition_params[0, :k_factors - 1] = phi
        factor_transition_params[1:, :-1] = np.identity(k_factors - 1,
                dtype=dtype)

        dynamic_factor_params[self._params_factor_transition] = \
                factor_transition_params.ravel()

        # 4. Error transition

        psi = params_without_intercept[self._psi_idx]

        dynamic_factor_params[self._params_error_transition] = psi

        return dynamic_factor_params

    def _get_params_without_intercept(self, dynamic_factor_params):
        '''
        reverse to previous
        '''

        dtype = self.ssm.dtype
        k_endog = self.k_endog
        k_factors = self.k_factors
        factor_order = self.factor_order
        error_order = self.error_order

        params_without_intercept = np.zeros(( \
                self._kimyoo_k_params_without_intercept,),
                dtype=dtype)

        # 1. Factor loadings

        factor_loadings_matrix = \
                dynamic_factor_params[self._params_loadings].reshape( \
                k_endog, k_factors)

        gammas = np.zeros((k_endog + k_factors - 1,), dtype=dtype)

        gammas[:k_endog] = factor_loadings_matrix[:, 0]
        gammas[k_endog:] = factor_loadings_matrix[-1, 1:]

        params_without_intercept[self._gamma_idx] = gammas

        # 2. Error covs

        params_without_intercept[self._sigma_idx] = dynamic_factor_params[ \
                self._params_error_cov]

        # 3. Factor transition

        # `factor_order` == 1, so this essentially is a square matrix
        factor_transition_params = \
                dynamic_factor_params[self._params_factor_transition \
                ].reshape(k_factors, k_factors * factor_order)

        params_without_intercept[self._phi_idx] = factor_transition_params[0,
                :k_factors - 1]

        # 4. Error transition

        psi = dynamic_factor_params[self._params_error_transition]

        params_without_intercept[self._psi_idx] = psi

        return params_without_intercept

    @property
    def start_params(self):

        dtype = self.ssm.dtype

        base_start_params = super(KimYoo1995NonswitchingModel,
                self).start_params

        params_without_intercept = self._get_params_without_intercept( \
                base_start_params[self._dynamic_factor_params_idx])

        start_params = np.zeros((self.kimyoo_k_params,), dtype=dtype)

        start_params[self._params_without_intercept_idx] = \
                params_without_intercept

        start_params[self._mu_idx] = base_start_params[ \
                self._factor_intercept_idx]

        return start_params

    def transform_params(self, unconstrained):

        dtype = self.ssm.dtype

        constrained = np.array(unconstrained)

        # `k_params` property is inherited from `DynamicFactor` class, so it
        # doesn't encounter factor intercept value in
        # `DynamicFactorWithFactorIntercept` class params
        unconstrained_base_params = np.zeros((self.k_params + 1,), dtype=dtype)

        unconstrained_base_params[self._dynamic_factor_params_idx] = \
                self._get_dynamic_factor_params( \
                unconstrained[self._params_without_intercept_idx])

        unconstrained_base_params[self._factor_intercept_idx] = \
                unconstrained[self._mu_idx]

        constrained_base_params = super(KimYoo1995NonswitchingModel,
                self).transform_params(unconstrained_base_params)

        constrained[self._params_without_intercept_idx] = \
                self._get_params_without_intercept( \
                constrained_base_params[self._dynamic_factor_params_idx])

        return constrained

    def untransform_params(self, constrained):

        dtype = self.ssm.dtype

        unconstrained = np.array(constrained)

        # `k_params` property is inherited from `DynamicFactor` class, so it
        # doesn't encounter factor intercept value in
        # `DynamicFactorWithFactorIntercept` class params
        constrained_base_params = np.zeros((self.k_params + 1,), dtype=dtype)

        constrained_base_params[self._dynamic_factor_params_idx] = \
                self._get_dynamic_factor_params( \
                constrained[self._params_without_intercept_idx])

        constrained_base_params[self._factor_intercept_idx] = \
                constrained[self._mu_idx]

        unconstrained_base_params = super(KimYoo1995NonswitchingModel,
                self).untransform_params(constrained_base_params)

        unconstrained[self._params_without_intercept_idx] = \
                self._get_params_without_intercept( \
                unconstrained_base_params[self._dynamic_factor_params_idx])

        return unconstrained

    def update(self, params, **kwargs):

        dtype = self.ssm.dtype
        k_states = self.k_states

        base_class_params = np.zeros((self.k_params + 1,), dtype=dtype)

        base_class_params[self._dynamic_factor_params_idx] = \
                self._get_dynamic_factor_params( \
                params[self._params_without_intercept_idx])

        base_class_params[self._factor_intercept_idx] = params[self._mu_idx]

        super(KimYoo1995NonswitchingModel, self).update(base_class_params,
                **kwargs)

        # Filter initialization.

        state_intercept = self['state_intercept']

        transition = self['transition']

        raveled_state_cov = self['state_cov'].T.reshape(-1, 1)

        initial_state = np.linalg.inv(np.identity(k_states, dtype=dtype) - \
                transition).dot(state_intercept).ravel()

        transition_outer_sqr = np.zeros((k_states * k_states,
                k_states * k_states), dtype=dtype)

        for i in range(k_states):
            for j in range(k_states):
                transition_outer_sqr[i * k_states:i * k_states + k_states,
                        j * k_states:j * k_states + k_states] = \
                        transition * transition[i, j]

        initial_state_cov = np.linalg.inv(np.identity(k_states * k_states,
                dtype=dtype) - transition_outer_sqr).dot(raveled_state_cov
                ).reshape(k_states, k_states).T

        self.initialize_known(initial_state, initial_state_cov)


class KimYoo1995Model(SwitchingDynamicFactor):
    '''
    This is switching dynamic factor model with some restrictions on
    parameters. See http://econ.korea.ac.kr/~cjkim/MARKOV/programs/sw_ms.opt.
    '''

    def __init__(self, k_regimes, endog, k_factors, factor_order, **kwargs):

        super(KimYoo1995Model, self).__init__(k_regimes, endog, k_factors,
                factor_order, **kwargs)

        # we need this instance because of its useful methods
        # `_get_dynamic_factor_params`, `_get_params_without_intercept` and
        # others
        self._nonswitching_model = KimYoo1995NonswitchingModel(endog,
                k_factors, factor_order, **kwargs)

        regime_transition_params_len = (k_regimes) * (k_regimes - 1)

        # params vector for this model = [transition_matrix_params
        # kimyoo_params_without_intercepts factor_intercepts]
        self._kimyoo_params_without_intercepts_idx = np.s_[ \
                regime_transition_params_len:-k_regimes]

        self._kimyoo_factor_intercepts_idx = np.s_[-k_regimes:]

        # A dirty hack, required, because Kim-Yoo model's specification is a
        # little different from Statsmodels one.
        self['state_cov', :k_factors, :k_factors] = 0
        self['state_cov', 0, 0] = 1

    @property
    def start_params(self):

        dtype = self.ssm.dtype
        k_regimes = self.k_regimes

        regime_transition_params_len = \
                self.parameters['regime_transition'].shape[0]

        base_start_params = super(KimYoo1995Model, self).start_params

        start_params = np.zeros((regime_transition_params_len + \
                self._nonswitching_model.kimyoo_k_params + \
                k_regimes,), dtype=dtype)

        # 1. Regime transition params

        start_params[self.parameters['regime_transition']] = \
                base_start_params[self.parameters['regime_transition']]

        # 2. Kim Yoo params without intercept

        start_params[self._kimyoo_params_without_intercepts_idx] = \
                self._nonswitching_model._get_params_without_intercept( \
                base_start_params[self._dynamic_factor_params_idx])

        # 3. Factor intercept params

        start_params[self._kimyoo_factor_intercepts_idx] = base_start_params[ \
                self._factor_intercepts_idx]

        return start_params

    def get_nonswitching_model(self):

        # don't need to instantiate a new model, since we already have one
        return self._nonswitching_model

    def update_params(self, params, nonswitching_params):

        params = np.array(params)

        params[self._kimyoo_params_without_intercepts_idx] = \
                nonswitching_params[ \
                self._nonswitching_model._params_without_intercept_idx]

        params[self._kimyoo_factor_intercepts_idx] = \
                nonswitching_params[ \
                self._nonswitching_model._factor_intercept_idx]

        return params

    def get_nonswitching_params(self, params):

        dtype = self.ssm.dtype

        nonswitching_params_without_intercept_idx = \
                self._nonswitching_model._params_without_intercept_idx

        nonswitching_factor_intercept_idx = \
                self._nonswitching_model._factor_intercept_idx

        nonswitching_params_length = \
                nonswitching_params_without_intercept_idx.shape[0] + \
                nonswitching_factor_intercept_idx.shape[0]

        nonswitching_params = np.zeros((nonswitching_params_length,),
                dtype=dtype)

        nonswitching_params[nonswitching_params_without_intercept_idx] = \
                params[self._kimyoo_params_without_intercepts_idx]

        nonswitching_params[nonswitching_factor_intercept_idx] = \
                params[self._kimyoo_factor_intercepts_idx].mean()

        return nonswitching_params

    def transform_model_params(self, unconstrained):

        constrained = np.array(unconstrained)

        unconstrained_dynamic_factor_params = \
                self._nonswitching_model._get_dynamic_factor_params( \
                unconstrained[self._kimyoo_params_without_intercepts_idx])

        constrained_dynamic_factor_params = \
                super(DynamicFactorWithFactorIntercept,
                self._nonswitching_model).transform_params( \
                unconstrained_dynamic_factor_params)

        constrained[self._kimyoo_params_without_intercepts_idx] = \
               self._nonswitching_model._get_params_without_intercept( \
               constrained_dynamic_factor_params)

        return constrained

    def untransform_model_params(self, constrained):

        unconstrained = np.array(constrained)

        constrained_dynamic_factor_params = \
                self._nonswitching_model._get_dynamic_factor_params( \
                constrained[self._kimyoo_params_without_intercepts_idx])

        unconstrained_dynamic_factor_params = \
                super(DynamicFactorWithFactorIntercept,
                self._nonswitching_model).untransform_params( \
                constrained_dynamic_factor_params)

        unconstrained[self._kimyoo_params_without_intercepts_idx] = \
               self._nonswitching_model._get_params_without_intercept( \
               unconstrained_dynamic_factor_params)

        return unconstrained

    def update(self, params, transformed=True, **kwargs):

        dtype = self.ssm.dtype
        k_regimes = self.k_regimes
        k_states = self.k_states

        if not transformed:
            params = self.transform_params(params)

        # `k_params` property is inherited from `SwitchingDynamicFactor` class
        base_class_params = np.zeros((self.k_params,), dtype=dtype)

        # 1. Regime transition params

        base_class_params[self.parameters['regime_transition']] = \
                params[self.parameters['regime_transition']]

        # 2. Dynamic factor params

        base_class_params[self._dynamic_factor_params_idx] = \
                self._nonswitching_model._get_dynamic_factor_params( \
                params[self._kimyoo_params_without_intercepts_idx])

        # 3. Factor intercepts params

        base_class_params[self._factor_intercepts_idx] = params[ \
                self._kimyoo_factor_intercepts_idx]

        super(KimYoo1995Model, self).update(base_class_params,
                transformed=True, **kwargs)

        # Filter initialization.

        initial_state = np.zeros((k_regimes, k_states), dtype=dtype)

        state_intercept = self['state_intercept']

        transition = self['transition'][0]

        raveled_state_cov = self['state_cov'][0].T.reshape(-1, 1)

        for i in range(k_regimes):
            initial_state[i] = np.linalg.inv(np.identity(k_states,
                    dtype=dtype) - transition).dot(state_intercept[i]
                    ).ravel()

        transition_outer_sqr = np.zeros((k_states * k_states,
                k_states * k_states), dtype=dtype)

        for i in range(k_states):
            for j in range(k_states):
                transition_outer_sqr[i * k_states:i * k_states + k_states,
                        j * k_states:j * k_states + k_states] = \
                        transition * transition[i, j]

        initial_state_cov = np.linalg.inv(np.identity(k_states * k_states,
                dtype=dtype) - transition_outer_sqr).dot(raveled_state_cov
                ).reshape(k_states, k_states).T

        self.initialize_known(initial_state, initial_state_cov)

class KimYoo1995(object):

    @classmethod
    def setup_class(cls, with_standardizing=True):

        cls.dtype = np.float64
        dtype = cls.dtype

        cls.k_regimes = 2
        cls.k_factors = 3
        cls.factor_order = 1
        cls.error_order = 2

        cls.true = results_kim_yoo1995.sw_ms

        ip = np.array(cls.true['data']['ip'], dtype=dtype)
        gmyxpq = np.array(cls.true['data']['gmyxpq'], dtype=dtype)
        mtq = np.array(cls.true['data']['mtq'], dtype=dtype)
        lpnag = np.array(cls.true['data']['lpnag'], dtype=dtype)
        cls.dcoinc = np.array(cls.true['data']['dcoinc'], dtype=dtype)

        yy = np.zeros((432, 4), dtype=dtype)
        yy[:, 0] = (np.log(ip[1:]) - np.log(ip[:-1])) * 100
        yy[:, 1] = (np.log(gmyxpq[1:]) - np.log(gmyxpq[:-1])) * 100
        yy[:, 2] = (np.log(mtq[1:]) - np.log(mtq[:-1])) * 100
        yy[:, 3] = (np.log(lpnag[1:]) - np.log(lpnag[:-1])) * 100

        yy -= yy.mean(axis=0)

        if with_standardizing:
            yy /= yy.std(axis=0)

        cls.obs = yy

        cls.model = KimYoo1995Model(cls.k_regimes, cls.obs,
                cls.k_factors, cls.factor_order, error_order=cls.error_order,
                loglikelihood_burn=cls.true['start'])


class TestKimYoo1995_Filtering(KimYoo1995):

    @classmethod
    def setup_class(cls):

        super(TestKimYoo1995_Filtering,
                cls).setup_class(with_standardizing=False)

        dtype = cls.dtype

        doc_in = cls.dcoinc[1:]
        doc_din = cls.dcoinc[1:] - cls.dcoinc[:-1]
        doc_mn = doc_din.mean()

        start_params = np.array(
                cls.true['start_parameters2'], dtype=dtype)

        results = cls.model.filter(start_params)

        dlt_ctt = results.filtered_states[:, 0]

        new_ctt = dlt_ctt * doc_din.std() / dlt_ctt.std()

        t = cls.obs.shape[0]

        new_ind = np.zeros((t,), dtype=dtype)
        new_ind[0] = doc_in[0]
        for i in range(1, t):
            new_ind[i] = new_ctt[i] + new_ind[i - 1] + doc_mn

        new_ind = new_ind - new_ind[119] + doc_in[119]

        cls.result = {
                'loglike': results.loglike(),
                'new_ind': new_ind,
                'prtt0': results.filtered_regime_probs[:, 0]
        }

    def test_loglike(self):
        assert_allclose(self.result['loglike'], self.true['loglike2'],
                rtol=3e-3)

    def test_new_ind(self):
        assert_allclose(self.result['new_ind'], self.true['new_ind'], rtol=1e-2)

    def test_probs(self):
        assert_allclose(self.result['prtt0'], self.true['prtt0'], atol=1e-2)
