"""
Tests for Kim filter

Author: Valery Likhosherstov
License: Simplified-BSD

References
----------

Kim, Chang-Jin, and Charles R. Nelson. 1999.
"State-Space Models with Regime Switching:
Classical and Gibbs-Sampling Approaches with Applications".
MIT Press Books. The MIT Press.
"""
import numpy as np
from numpy.testing import assert_allclose
from statsmodels.tsa.statespace.regime_switching.kim_filter import KimFilter
from kim1994 import Kim1994
from statsmodels.tsa.statespace.tools import compatibility_mode
from nose.exc import SkipTest

if compatibility_mode:
    raise SkipTest


class TestKim1994_KimFilter(Kim1994):
    """
    Basic test for the loglikelihood and filtered states precision
    """

    @classmethod
    def setup_class(cls):

        super(TestKim1994_KimFilter, cls).setup_class()

        cls.kim_filter = cls.init_filter()
        cls.result = cls.run_filter()

    @classmethod
    def init_filter(cls):
        # Base class usage
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

        results = cls.kim_filter.filter()

        return {
            'loglike': results.loglike(),
            'cycle': results.filtered_state[0, cls.true['start']:]
        }

    def test_loglike(self):
        assert_allclose(self.result['loglike'], self.true['loglike'],
                rtol=1e-6)

    def test_filtered_state(self):
        assert_allclose(self.result['cycle'], self.true_cycle, rtol=2e-3)
