"""
Miscellaneous Tests

- Tests for setting options in KalmanFilter, KalmanSmoother, SimulationSmoother
  (does not test the filtering, smoothing, or simulation smoothing for each
  option)

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from statsmodels.tsa.statespace.kalman_filter import (
    FILTER_CONVENTIONAL,

    INVERT_UNIVARIATE,
    SOLVE_LU,
    INVERT_LU,
    SOLVE_CHOLESKY,
    INVERT_CHOLESKY,
    INVERT_NUMPY,

    STABILITY_FORCE_SYMMETRY,

    MEMORY_STORE_ALL,
    MEMORY_NO_FORECAST,
    MEMORY_NO_PREDICTED,
    MEMORY_NO_FILTERED,
    MEMORY_NO_LIKELIHOOD,
    MEMORY_CONSERVE
)
from statsmodels.tsa.statespace.model import Model
from numpy.testing import assert_equal


class Options(Model):
    def __init__(self, *args, **kwargs):

        # Dummy data
        endog = np.arange(10)
        k_states = 1

        super(Options, self).__init__(endog, k_states, *args, **kwargs)

class TestOptions(Options):
    def test_filter_methods(self):
        # Clear the filter method
        self.filter_method = 0

        # Try setting via boolean
        self.filter_conventional = True
        assert_equal(self.filter_method, FILTER_CONVENTIONAL)
        self.filter_conventional = False
        assert_equal(self.filter_method, 0)

        # Try setting directly via method
        self.set_filter_method(FILTER_CONVENTIONAL)
        assert_equal(self.filter_method, FILTER_CONVENTIONAL)

        # Try setting via boolean via method
        self.set_filter_method(filter_conventional=True)
        assert_equal(self.filter_method, FILTER_CONVENTIONAL)

    def test_inversion_methods(self):
        # Clear the inversion method
        self.inversion_method = 0

        # Try setting via boolean
        self.invert_univariate = True
        assert_equal(self.inversion_method, INVERT_UNIVARIATE)
        self.invert_cholesky = True
        assert_equal(self.inversion_method, INVERT_UNIVARIATE | INVERT_CHOLESKY)
        self.invert_univariate = False
        assert_equal(self.inversion_method, INVERT_CHOLESKY)

        # Try setting directly via method
        self.set_inversion_method(INVERT_LU)
        assert_equal(self.inversion_method, INVERT_LU)

        # Try setting via boolean via method
        self.set_inversion_method(invert_cholesky=True, invert_univariate=True, invert_lu=False)
        assert_equal(self.inversion_method, INVERT_UNIVARIATE | INVERT_CHOLESKY)

        # Try setting and unsetting all
        self.inversion_method = 0
        for name in self.inversion_methods:
            setattr(self, name, True)
        assert_equal(
            self.inversion_method,
            INVERT_UNIVARIATE | SOLVE_LU | INVERT_LU | SOLVE_CHOLESKY |
            INVERT_CHOLESKY | INVERT_NUMPY
        )
        for name in self.inversion_methods:
            setattr(self, name, False)
        assert_equal(self.inversion_method, 0)

    def test_stability_methods(self):
        # Clear the stability method
        self.stability_method = 0

        # Try setting via boolean
        self.stability_force_symmetry = True
        assert_equal(self.stability_method, STABILITY_FORCE_SYMMETRY)
        self.stability_force_symmetry = False
        assert_equal(self.stability_method, 0)

        # Try setting directly via method
        self.stability_method = 0
        self.set_stability_method(STABILITY_FORCE_SYMMETRY)
        assert_equal(self.stability_method, STABILITY_FORCE_SYMMETRY)

        # Try setting via boolean via method
        self.stability_method = 0
        self.set_stability_method(stability_method=True)
        assert_equal(self.stability_method, STABILITY_FORCE_SYMMETRY)


    def test_conserve_memory(self):
        # Clear the filter method
        self.conserve_memory = MEMORY_STORE_ALL

        # Try setting via boolean
        self.memory_no_forecast = True
        assert_equal(self.conserve_memory, MEMORY_NO_FORECAST)
        self.memory_no_filtered = True
        assert_equal(self.conserve_memory, MEMORY_NO_FORECAST | MEMORY_NO_FILTERED)
        self.memory_no_forecast = False
        assert_equal(self.conserve_memory, MEMORY_NO_FILTERED)

        # Try setting directly via method
        self.set_conserve_memory(MEMORY_NO_PREDICTED)
        assert_equal(self.conserve_memory, MEMORY_NO_PREDICTED)

        # Try setting via boolean via method
        self.set_conserve_memory(memory_no_filtered=True, memory_no_predicted=False)
        assert_equal(self.conserve_memory, MEMORY_NO_FILTERED)

        # Try setting and unsetting all
        self.conserve_memory = 0
        for name in self.memory_options:
            if name == 'memory_conserve':
                continue
            setattr(self, name, True)
        assert_equal(
            self.conserve_memory,
            MEMORY_NO_FORECAST | MEMORY_NO_PREDICTED | MEMORY_NO_FILTERED |
            MEMORY_NO_LIKELIHOOD
        )
        assert_equal(self.conserve_memory, MEMORY_CONSERVE)
        for name in self.memory_options:
            if name == 'memory_conserve':
                continue
            setattr(self, name, False)
        assert_equal(self.conserve_memory, 0)
