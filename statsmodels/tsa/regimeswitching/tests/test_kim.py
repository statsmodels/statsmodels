import numpy as np
import pandas as pd

from statsmodels.tsa.regimeswitching.kim_filter import KimFilter
from .results import results_kim_filter
from numpy.testing import assert_allclose

class TestKim1994(object):
    
    @classmethod
    def setup_class(cls):
        cls.dtype = np.float64
        dtype = cls.dtype
        
        cls.true = results_kim_filter.kim_je
        cls.true_cycle = np.array(cls.true['cycle'], dtype=dtype)

        data = np.array(cls.true['data'], dtype=dtype)
        data = np.log(data)*100 

        cls.obs = np.array(data[1:152] - data[:151], dtype=dtype)
        
        cls.set_model_matrices()

        cls.kim_filter = cls.init_filter()
        cls.kim_filter.initialize_known(cls.initial_state_mean,
                cls.initial_state_cov)

        cls.result = cls.run_filter() 

    @classmethod
    def set_model_matrices(cls):
        dtype = cls.dtype

        p, q, phi_1, phi_2, sigma, delta_0, delta_1 = cls.true['parameters']

        cls.k_regimes = k_regimes = 2
        cls.k_endog = k_endog = 1
        cls.k_states = k_states = 2

        cls.regime_switch_probs = np.zeros((k_regimes, k_regimes))
        cls.regime_switch_probs[:, :] = [[q, 1 - q], [1 - p, p]] 

        cls.design = np.zeros((k_endog, k_states, 1), dtype=dtype)
        cls.design[0, :, 0] = [1, -1]
        
        cls.obs_intercepts = np.zeros((k_regimes, k_endog, 1), dtype=dtype)
        cls.obs_intercepts[:, 0, 0] = [delta_0, delta_0 + delta_1]

        cls.transition = np.zeros((k_states, k_states, 1), dtype=dtype)
        cls.transition[:, :, 0] = [[phi_1, phi_2], [1, 0]]

        cls.selection = np.zeros((k_states, k_states, 1), dtype=dtype)
        cls.selection[:, :, 0] = [[1, 0], [0, 1]]

        cls.state_cov = np.zeros((k_states, k_states, 1), dtype=dtype)
        cls.state_cov[0, 0, 0] = sigma**2
        
        cls.initial_state_mean = np.zeros((k_states,), dtype=dtype)
        
        transition_outer_sqr = np.zeros((4, 4), dtype=dtype)
        
        for i in range(0, 2):
            for j in range(0, 2):
                transition_outer_sqr[i * 2:i * 2 + 2, j * 2:j * 2 + 2] = \
                        cls.transition[i, j, 0] * cls.transition[:, :, 0]
        
        initial_state_cov_vector = np.linalg.inv(np.eye(4, dtype=dtype) -
                transition_outer_sqr).dot(cls.state_cov.reshape(-1, 1))
        
        cls.initial_state_cov = initial_state_cov_vector.reshape(k_states,
                k_states).T

    @classmethod
    def init_filter(cls):
        return KimFilter(cls.k_endog, cls.k_states, cls.k_regimes,
                dtype=cls.dtype, loglikelihood_burn=cls.true['start'],
                designs=cls.design, obs_intercepts=cls.obs_intercepts,
                transitions=cls.transition, selections=cls.selection,
                state_covs=cls.state_cov,
                regime_switch_probs=cls.regime_switch_probs)
 
    @classmethod
    def run_filter(cls): 
        cls.kim_filter.bind(cls.obs)

        cls.kim_filter.filter()

        return {
            'loglike': cls.kim_filter.loglike(),
            'cycle': cls.kim_filter.filtered_states[cls.true['start']:, 0]
        }

    def test_loglike(self):
        assert_allclose(self.result['loglike'], self.true['loglike'],
                atol=1e-3)

    def test_filtered_state(self):
        assert_allclose(self.result['cycle'], self.true_cycle, atol=1e-2)
