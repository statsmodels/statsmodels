import numpy as np
from numpy.testing import assert_allclose
from statsmodels.tsa.statespace.regime_switching.kim_filter import KimFilter
from kim1994 import Kim1994

class TestKim1994_KimFilter(Kim1994):
    '''
    Kim filter test based on output of
    http://econ.korea.ac.kr/~cjkim/MARKOV/programs/kim_je.opt.
    See chapter 5.4.2 of Kim and Nelson book and for details.
    '''

    @classmethod
    def setup_class(cls):

        super(TestKim1994_KimFilter, cls).setup_class()

        cls.kim_filter = cls.init_filter()
        cls.result = cls.run_filter()

    @classmethod
    def init_filter(cls):
        regime_transition, design, obs_intercept, transition, selection, \
                state_cov, initial_state_mean, initial_state_cov = \
                cls.get_model_matrices(cls.dtype, cls.true['parameters'])

        kim_filter = KimFilter(cls.k_endog, cls.k_states, cls.k_regimes,
                k_posdef=cls.k_posdef, dtype=cls.dtype,
                loglikelihood_burn=cls.true['start'], design=design,
                obs_intercept=obs_intercept, transition=transition,
                selection=selection, state_cov=state_cov,
                regime_transition=regime_transition)

        kim_filter.initialize_stationary_regime_probs()
        kim_filter.initialize_known(initial_state_mean, initial_state_cov)

        return kim_filter

    @classmethod
    def run_filter(cls):
        cls.kim_filter.bind(cls.obs)

        cls.kim_filter.filter()

        return {
            'loglike': cls.kim_filter.loglike(filter_first=False),
            'cycle': cls.kim_filter.filtered_states[cls.true['start']:, 0]
        }

    def test_loglike(self):
        assert_allclose(self.result['loglike'], self.true['loglike'],
                rtol=1e-6)

    def test_filtered_state(self):
        assert_allclose(self.result['cycle'], self.true_cycle, rtol=2e-3)
