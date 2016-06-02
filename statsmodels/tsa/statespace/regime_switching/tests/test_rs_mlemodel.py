import numpy as np
from numpy.testing import assert_allclose
from statsmodels.tsa.statespace.regime_switching.rs_mlemodel import \
        RegimeSwitchingMLEModel
from kim1994 import Kim1994

class Kim1994Model(RegimeSwitchingMLEModel):

    def transform_model_params(self, unconstrained_model_params):

        constrained_model_params = np.array(unconstrained_model_params)
        root1 = unconstrained_model_params[0] / (1 + \
                np.abs(unconstrained_model_params[0]))
        root2 = unconstrained_model_params[1] / (1 + \
                np.abs(unconstrained_model_params[1]))
        constrained_model_params[0] = root1 + root2
        constrained_model_params[1] = -root1 * root2

        return constrained_model_params

    def untransform_model_params(self, constrained_model_params):

        unconstrained_model_params = np.array(constrained_model_params)
        b_coef = constrained_model_params[0]
        c_coef = constrained_model_params[1]
        root1 = (b_coef - np.sqrt(b_coef * b_coef + 4 * c_coef)) / 2.0
        root2 = (b_coef + np.sqrt(b_coef * b_coef + 4 * c_coef)) / 2.0

        unconstrained_model_params[0] = root1 / (1 - np.sign(root1) * root1)
        unconstrained_model_params[1] = root2 / (1 - np.sign(root2) * root2)

        return unconstrained_model_params


    def update(self, params, **kwargs):

        params = super(Kim1994Model, self).update(params, **kwargs)

        self['regime_transition'], self['design'], self['obs_intercept'], \
                self['transition'], self['selection'], self['state_cov'], \
                initial_state_mean, initial_state_cov = \
                Kim1994.get_model_matrices(self.ssm.dtype, params)

        self.initialize_known(initial_state_mean, initial_state_cov)
        self.initialize_stationary_regime_probs()

        self.ssm.filter()

class TestKim1994_MLEModel(Kim1994):

    @classmethod
    def setup_class(cls):

        super(TestKim1994_MLEModel, cls).setup_class()

        cls.model = cls.init_model()
        cls.model.initialize_stationary_regime_probs()
        cls.result = cls.fit_model()

    @classmethod
    def init_model(cls):
        model = Kim1994Model(cls.k_regimes, cls.obs, cls.k_states,
                dtype=cls.dtype, loglikelihood_burn=cls.true['start'],
                k_posdef=cls.k_posdef)

        return model

    @classmethod
    def fit_model(cls):
        constrained_start_params = \
                cls.model.transform_params(
                np.array(cls.true['untransformed_start_parameters'],
                dtype=cls.dtype))

        start_transition, start_model_params = \
                cls.model._get_explicit_params(constrained_start_params)

        params = cls.model.fit(start_transition=start_transition,
                start_model_params=start_model_params,
                fit_nonswitching_first=False,
                return_params=True)

        return {
                'loglike': cls.model.loglike(params, filter_first=False),
                'params': params
        }

    def test_loglike(self):
        assert_allclose(self.result['loglike'], self.true['loglike'], rtol=1e-5)

    def test_params(self):
        assert_allclose(self.result['params'], self.true['parameters'], rtol=1e-2)
