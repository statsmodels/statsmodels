import numpy as np
from numpy.testing import assert_allclose
from statsmodels.tsa.statespace.api import MLEModel
from statsmodels.tsa.statespace.regime_switching.api import \
        SwitchingMLEModel
from kim1994 import Kim1994


def _transform_ar_coefs(unconstrained):
    root1 = unconstrained[0] / (1 + np.abs(unconstrained[0]))
    root2 = unconstrained[1] / (1 + np.abs(unconstrained[1]))
    return (root1 + root2, -root1 * root2)

def _untransform_ar_coefs(constrained):
    b_coef = constrained[0]
    c_coef = constrained[1]
    root1 = (b_coef - np.sqrt(b_coef * b_coef + 4 * c_coef)) / 2.0
    root2 = (b_coef + np.sqrt(b_coef * b_coef + 4 * c_coef)) / 2.0
    unconstrained_roots = (root1 / (1 - np.sign(root1) * root1),
            root2 / (1 - np.sign(root2) * root2))
    return unconstrained_roots


class Linear_Kim1994Model(MLEModel):
    '''
    Linear Model without switching. Is used for starting parameters
    estimation in TestKim1994_MLEModelLinearFitFirst.
    '''

    def transform_params(self, unconstrained):
        constrained = np.array(unconstrained)
        constrained[:2] = _transform_ar_coefs(unconstrained[:2])
        return constrained

    def untransform_params(self, constrained):
        unconstrained = np.array(constrained)
        unconstrained[:2] = _untransform_ar_coefs(constrained[:2])
        return unconstrained

    def update(self, params, *args, **kwargs):

        params = super(Linear_Kim1994Model, self).update(params, *args,
                **kwargs)

        # Some dirty hack
        _, self['design'], obs_intercept, self['transition'], \
                self['selection'], self['state_cov'], initial_state_mean, \
                initial_state_cov = \
                Kim1994.get_model_matrices(self.ssm.dtype,
                np.hstack((0.5, 0.5, params, 0)))

        self['obs_intercept'] = obs_intercept[0]

        self.initialize_known(initial_state_mean, initial_state_cov)


class Kim1994Model(SwitchingMLEModel):
    '''
    Switching model.
    '''

    def __init__(self, *args, **kwargs):
        '''
        need to specify switching params
        '''

        super(Kim1994Model, self).__init__(*args, **kwargs)

        self.parameters['phi'] = [False, False]
        self.parameters['sigma'] = [False]
        self.parameters['delta'] = [True]

    def get_nonswitching_model(self):

        return Linear_Kim1994Model(endog=self.endog, k_states=self.k_states,
                k_posdef=1)

    def update_params(self, params, nonswitching_params):

        params[self.parameters['phi']] = nonswitching_params[:2]
        params[self.parameters['sigma']] = nonswitching_params[2]
        # Without shifting a little we are stuck in local maximum
        params[self.parameters[0, 'delta']] = nonswitching_params[3] - 1e-2
        params[self.parameters[1, 'delta']] = nonswitching_params[3] + 1e-2
        return params

    def get_nonswitching_params(self, params):

        nonswitching_params = np.zeros((4,), dtype=self.ssm.dtype)
        nonswitching_params[:2] = params[self.parameters['phi']]
        nonswitching_params[2] = params[self.parameters['sigma']]
        nonswitching_params[3] = params[self.parameters['delta']].mean()
        return nonswitching_params

    def transform_model_params(self, unconstrained):

        constrained = np.array(unconstrained)
        constrained[self.parameters['phi']] = \
                _transform_ar_coefs(unconstrained[self.parameters['phi']])
        return constrained

    def untransform_model_params(self, constrained):
        unconstrained = np.array(constrained)
        unconstrained[self.parameters['phi']] = \
                _untransform_ar_coefs(constrained[self.parameters['phi']])
        return unconstrained

    def update(self, params, **kwargs):

        params = super(Kim1994Model, self).update(params, **kwargs)

        self['regime_transition'], self['design'], self['obs_intercept'], \
                self['transition'], self['selection'], self['state_cov'], \
                initial_state_mean, initial_state_cov = \
                Kim1994.get_model_matrices(self.ssm.dtype, params)

        self.initialize_known(initial_state_mean, initial_state_cov)
        self.initialize_stationary_regime_probs()

class Kim1994WithMLEModel(Kim1994):
    '''
    Basic class for testing SwitchingMLEModel.
    '''

    @classmethod
    def setup_class(cls):
        super(Kim1994WithMLEModel, cls).setup_class()

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

        raise NotImplementedError


class TestKim1994_MLEModel(Kim1994WithMLEModel):
    '''
    Test for equivalence with kim_je example.
    '''

    @classmethod
    def fit_model(cls):

        params = cls.model.fit(start_params=np.array(
            cls.true['untransformed_start_parameters'], dtype=cls.dtype),
            transformed=False)

        return {
                'loglike': cls.model.loglike(params),
                'params': params
        }

    def test_loglike(self):
        assert_allclose(self.result['loglike'], self.true['loglike'], rtol=1e-5)

    def test_params(self):
        assert_allclose(self.result['params'], self.true['parameters'], rtol=1e-2)


class TestKim1994_MLEModelLinearFitFirst(Kim1994WithMLEModel):
    '''
    Testing feature with starting optimization from the linear fit.
    '''

    @classmethod
    def fit_model(cls):

        params = cls.model.fit(fit_nonswitching_first=True)

        return {
                'loglike': cls.model.loglike(params),
                'params': params
        }

    def test_loglike(self):
        assert_allclose(self.result['loglike'], self.true['loglike'], rtol=1e-2)

    def test_params(self):
        assert_allclose(self.result['params'], self.true['parameters'], rtol=0.25)
