import numpy as np
from .results import results_kim_filter

class Kim1994(object):
    '''
    Base class for Kim filter and switching MLE model test. Handles data
    and transformation params to state space matrices.
    See chapter 5.4.2 of Kim and Nelson book and
    http://econ.korea.ac.kr/~cjkim/MARKOV/programs/kim_je.opt for details.
    '''

    @classmethod
    def setup_class(cls):
        cls.dtype = np.float64
        dtype = cls.dtype

        cls.k_regimes = 2
        cls.k_endog = 1
        cls.k_states = 2
        cls.k_posdef = 1

        cls.true = results_kim_filter.kim_je
        cls.true_cycle = np.array(cls.true['cycle'], dtype=dtype)

        data = np.array(cls.true['data'], dtype=dtype)
        data = np.log(data)*100

        cls.obs = np.array(data[1:152] - data[:151], dtype=dtype)

    @classmethod
    def get_model_matrices(cls, dtype, params):
        '''
        Transforms parameter vector into state space representation
        matrices.
        '''

        k_regimes = 2
        k_endog = 1
        k_states = 2
        k_posdef = 1

        p, q, phi_1, phi_2, sigma, delta_0, delta_1 = params

        regime_transition = np.zeros((k_regimes, k_regimes))
        regime_transition[:, :] = [[q, p], [1 - q, 1 - p]]

        design = np.zeros((k_endog, k_states, 1), dtype=dtype)
        design[0, :, 0] = [1, -1]

        obs_intercept = np.zeros((k_regimes, k_endog, 1), dtype=dtype)
        obs_intercept[:, 0, 0] = [delta_0, delta_1]

        transition = np.zeros((k_states, k_states, 1), dtype=dtype)
        transition[:, :, 0] = [[phi_1, phi_2], [1, 0]]

        selection = np.zeros((k_states, k_posdef, 1), dtype=dtype)
        selection[:, :, 0] = [[1], [0]]

        state_cov = np.zeros((k_posdef, k_posdef, 1), dtype=dtype)
        state_cov[0, 0, 0] = sigma**2

        initial_state_mean = np.zeros((k_states,), dtype=dtype)

        transition_outer_sqr = np.zeros((4, 4), dtype=dtype)

        for i in range(0, 2):
            for j in range(0, 2):
                transition_outer_sqr[i * 2:i * 2 + 2, j * 2:j * 2 + 2] = \
                        transition[i, j, 0] * transition[:, :, 0]

        nonpos_def_state_cov = selection.dot(state_cov).dot(selection.T)

        initial_state_cov_vector = np.linalg.inv(np.eye(4, dtype=dtype) -
                transition_outer_sqr).dot(nonpos_def_state_cov.reshape(-1, 1))

        initial_state_cov = initial_state_cov_vector.reshape(k_states,
                k_states).T

        return (regime_transition,
                design,
                obs_intercept,
                transition,
                selection,
                state_cov,
                initial_state_mean,
                initial_state_cov)
