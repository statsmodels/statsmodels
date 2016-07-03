import numpy as np
from numpy.testing import assert_allclose
from statsmodels.tsa.statespace.regime_switching.tools import \
        MarkovSwitchingParams
from statsmodels.tsa.statespace.regime_switching.switching_dynamic_factor \
        import _DynamicFactorWithFactorIntercept, SwitchingDynamicFactor
from .results import results_kim_yoo1995


class KimYoo1995NonswitchingModel(_DynamicFactorWithFactorIntercept):
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

        # For the sake of clarity
        self._base_class_k_params = self.k_params_with_factor_intercept

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

        dynamic_factor_params = np.zeros((self.dynamic_factor_k_params,),
                dtype=dtype)

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

    def _get_base_class_params(self, params):

        dtype = self.ssm.dtype

        base_class_params = np.zeros((self._base_class_k_params,), dtype=dtype)

        base_class_params[self._dynamic_factor_params_idx] = \
                self._get_dynamic_factor_params(
                params[self._params_without_intercept_idx])

        base_class_params[self._factor_intercept_idx] = params[self._mu_idx]

        return base_class_params

    def _get_params(self, base_class_params):

        dtype = self.ssm.dtype

        params_without_intercept = self._get_params_without_intercept(
                base_class_params[self._dynamic_factor_params_idx])

        params = np.zeros((self.kimyoo_k_params,), dtype=dtype)

        params[self._params_without_intercept_idx] = params_without_intercept

        params[self._mu_idx] = base_class_params[self._factor_intercept_idx]

        return params

    @property
    def start_params(self):

        base_start_params = super(KimYoo1995NonswitchingModel,
                self).start_params

        return self._get_params(base_start_params)

    def transform_params(self, unconstrained):

        unconstr_base_class_params = self._get_base_class_params(unconstrained)

        constr_base_class_params = super(KimYoo1995NonswitchingModel,
                self).transform_params(unconstr_base_class_params)

        constrained = self._get_params(constr_base_class_params)

        return constrained

    def untransform_params(self, constrained):

        constr_base_class_params = self._get_base_class_params(constrained)

        unconstr_base_class_params = super(KimYoo1995NonswitchingModel,
                self).untransform_params(constr_base_class_params)

        unconstrained = self._get_params(unconstr_base_class_params)

        return unconstrained

    def update(self, params, **kwargs):

        dtype = self.ssm.dtype
        k_states = self.k_states

        base_class_params = self._get_base_class_params(params)

        super(KimYoo1995NonswitchingModel, self).update(base_class_params,
                **kwargs)

        # Filter initialization.

        state_intercept = self['state_intercept']

        transition = self['transition']

        raveled_state_cov = (self['selection'].dot(self['state_cov']).dot(
                self['selection'].T)).reshape(-1, 1)

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

        # For the sake of clarity
        self._base_class_parameters = self.parameters

        # params vector for this model differs from params vector in
        # `SwitchingDynamicFactor`.
        self._kimyoo_parameters = MarkovSwitchingParams(k_regimes)

        self._kimyoo_parameters['regime_transition'] = [False] * k_regimes * \
                (k_regimes - 1)

        # Number of nonswitching params is equal to number of parameters in
        # nonswitching model, except of factor intercept (1 value).
        self._kimyoo_parameters['nonswitching_params'] = [False] * \
                (self._nonswitching_model.kimyoo_k_params - 1)

        self._kimyoo_parameters['factor_intercept'] = [True]

        # A dirty hack, required, because Kim-Yoo model's specification is a
        # little different from Statsmodels one.
        self['state_cov', :k_factors, :k_factors] = 0
        self['state_cov', 0, 0] = 1

    def _get_base_class_params(self, params):

        dtype = self.ssm.dtype

        base_class_params = np.zeros((self._base_class_parameters.k_params,),
                dtype=dtype)

        base_class_params[self._base_class_parameters['regime_transition']] = \
                params[self._kimyoo_parameters['regime_transition']]

        params_without_intercept = params[self._kimyoo_parameters[
                'nonswitching_params']]

        base_class_params[self._base_class_parameters['dynamic_factor']] = \
                self._nonswitching_model._get_dynamic_factor_params(
                params_without_intercept)

        base_class_params[self._base_class_parameters['factor_intercept']] = \
                params[self._kimyoo_parameters['factor_intercept']]

        return base_class_params

    def _get_params(self, base_class_params):

        dtype = self.ssm.dtype

        params = np.zeros((self._kimyoo_parameters.k_params,), dtype=dtype)

        params[self._kimyoo_parameters['regime_transition']] = \
                base_class_params[self._base_class_parameters[
                'regime_transition']]

        dynamic_factor_params = base_class_params[self._base_class_parameters[
                'dynamic_factor']]

        params[self._kimyoo_parameters['nonswitching_params']] = \
                self._nonswitching_model._get_params_without_intercept(
                dynamic_factor_params)

        params[self._kimyoo_parameters['factor_intercept']] = \
                base_class_params[self._base_class_parameters[
                'factor_intercept']]

        return params

    @property
    def start_params(self):

        dtype = self.ssm.dtype

        base_start_params = super(KimYoo1995Model, self).start_params

        return self._get_params(base_start_params)

    def get_nonswitching_model(self):

        # don't need to instantiate a new model, since we already have one
        return self._nonswitching_model

    def update_params(self, params, nonswitching_model_params):

        base_class_params = self._get_base_class_params(params)

        nonswitching_base_class_params = \
                self._nonswitching_model._get_base_class_params(
                nonswitching_model_params)

        updated_base_class_params = super(KimYoo1995Model,
                self).update_params(base_class_params,
                nonswitching_base_class_params)

        return self._get_params(updated_base_class_params)

    def transform_model_params(self, unconstrained):

        unconstr_base_class_params = self._get_base_class_params(unconstrained)

        constr_base_class_params = super(KimYoo1995Model,
                self).transform_model_params(unconstr_base_class_params)

        return self._get_params(constr_base_class_params)

    def untransform_model_params(self, constrained):

        constr_base_class_params = self._get_base_class_params(constrained)

        unconstr_base_class_params = super(KimYoo1995Model,
                self).untransform_model_params(constr_base_class_params)

        return self._get_params(unconstr_base_class_params)

    def update(self, params, **kwargs):

        dtype = self.ssm.dtype
        k_regimes = self.k_regimes
        k_states = self.k_states

        base_class_params = self._get_base_class_params(params)

        super(KimYoo1995Model, self).update(base_class_params, **kwargs)

        # Filter initialization.

        initial_state = np.zeros((k_regimes, k_states), dtype=dtype)

        state_intercept = self['state_intercept']

        transition = self['transition'][0]

        raveled_state_cov = self['state_cov'][0].reshape(-1, 1)

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
    def setup_class(cls, with_standardizing):

        cls.dtype = np.float64
        dtype = cls.dtype

        cls.k_regimes = 2
        cls.k_factors = 3
        cls.factor_order = 1
        cls.error_order = 2
        cls.enforce_stationarity = False

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
                loglikelihood_burn=cls.true['start'],
                enforce_stationarity=cls.enforce_stationarity)


class TestKimYoo1995_Filtering(KimYoo1995):

    @classmethod
    def setup_class(cls):

        super(TestKimYoo1995_Filtering, cls).setup_class(False)

        dtype = cls.dtype

        doc_in = cls.dcoinc[1:]
        doc_din = cls.dcoinc[1:] - cls.dcoinc[:-1]
        doc_mn = doc_din.mean()

        start_params = np.array(
                cls.true['start_params_nonstd_data'], dtype=dtype)

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
        assert_allclose(self.result['loglike'], self.true['start_loglike_nonstd_data'],
                rtol=1e-4)

    def test_new_ind(self):
        assert_allclose(self.result['new_ind'], self.true['new_ind'], rtol=1e-2)

    def test_probs(self):
        assert_allclose(self.result['prtt0'], self.true['prtt0'], atol=1e-2)


class TestKimYoo1995_Smoothing(KimYoo1995):

    @classmethod
    def setup_class(cls):

        super(TestKimYoo1995_Smoothing, cls).setup_class(False)

        dtype = cls.dtype

        start_params = np.array(
                cls.true['start_params_nonstd_data'], dtype=dtype)

        results = cls.model.smooth(start_params)

        cls.result = {
                'smooth0': results.smoothed_regime_probs[:, 0]
        }

    def test_probs(self):
        assert_allclose(self.result['smooth0'], self.true['smooth0'], atol=1e-2)


class KimYoo1995_MLE(KimYoo1995):

    @classmethod
    def setup_class(cls, with_standardizing):

        super(KimYoo1995_MLE, cls).setup_class(with_standardizing)

        dtype = cls.dtype

        if with_standardizing:
            start_params_name = 'start_params_std_data'
            cls.optimized_params_name = 'optimized_params_std_data'
            cls.optimized_loglike_name = 'optimized_loglike_std_data'
        else:
            start_params_name = 'start_params_nonstd_data'
            cls.optimized_params_name = 'optimized_params_nonstd_data'
            cls.optimized_loglike_name = 'optimized_loglike_nonstd_data'

        start_params = np.array(
                cls.true[start_params_name], dtype=dtype)

        params = cls.model.fit(start_params=start_params)

        cls.result = {
                'loglike': cls.model.loglike(params),
                'params': params
        }

    def test_loglike(self):
        # It occurs that the result loglike is even better than the value
        # it's tested against in both standardized and non-standardized cases.
        assert_allclose(self.result['loglike'],
                self.true[self.optimized_loglike_name], rtol=1e-2)

    def test_params(self):
        assert_allclose(self.result['params'],
                self.true[self.optimized_params_name], atol=2e-1)

class TestKimYoo1995_MLEWithStandardizing(KimYoo1995_MLE):

    @classmethod
    def setup_class(cls):

        super(TestKimYoo1995_MLEWithStandardizing, cls).setup_class(True)


class TestKimYoo1995_MLEWithoutStandardizing(KimYoo1995_MLE):

    @classmethod
    def setup_class(cls):

        super(TestKimYoo1995_MLEWithoutStandardizing, cls).setup_class(False)


class TestKimYoo1995_MLEFitNonswitchingFirst(KimYoo1995):

    @classmethod
    def setup_class(cls):

        super(TestKimYoo1995_MLEFitNonswitchingFirst, cls).setup_class(False)

        dtype = cls.dtype

        params = cls.model.fit(set_equal_transition_probs=True,
                fit_nonswitching_first=True)

        cls.result = {
                'loglike': cls.model.loglike(params),
                'params': params
        }

    def test_loglike(self):
        assert_allclose(self.result['loglike'],
                self.true['optimized_loglike_nonstd_data'], rtol=1e-2)

    def test_params(self):
        assert_allclose(self.result['params'],
                self.true['optimized_params_nonstd_data'], atol=2e-1)
