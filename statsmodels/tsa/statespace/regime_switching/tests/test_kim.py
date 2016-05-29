from numpy.testing import assert_allclose
from statsmodels.tsa.statespace.regime_switching.kim_filter import KimFilter
from kim1994 import Kim1994

class TestKim1994_KimFilter(Kim1994):

    @classmethod
    def setup_class(cls):

        super(TestKim1994_KimFilter, cls).setup_class()

        cls.regime_switch_probs, cls.design, cls.obs_intercepts, \
                cls.transition, cls.selection, cls.state_cov, \
                cls.initial_state_mean, cls.initial_state_cov = \
                cls.get_model_matrices(cls.dtype, cls.true['parameters'])

        cls.kim_filter = cls.init_filter()
        cls.kim_filter.initialize_known(cls.initial_state_mean,
                cls.initial_state_cov)

        cls.result = cls.run_filter()

    @classmethod
    def init_filter(cls):
        return KimFilter(cls.k_endog, cls.k_states, cls.k_regimes,
                k_posdef=cls.k_posdef, dtype=cls.dtype,
                loglikelihood_burn=cls.true['start'], designs=cls.design,
                obs_intercepts=cls.obs_intercepts, transitions=cls.transition,
                selections=cls.selection, state_covs=cls.state_cov,
                regime_switch_probs=cls.regime_switch_probs)

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
                rtol=1e-4)

    def test_filtered_state(self):
        assert_allclose(self.result['cycle'], self.true_cycle, atol=1e-2)
