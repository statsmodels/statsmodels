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

        self.model = Model(endog, k_states, *args, **kwargs)

class TestOptions(Options):
    def test_filter_methods(self):
        model = self.model

        # TODO test FilterResults for accurante boolean versions of options
        
        # Clear the filter method
        model.filter_method = 0

        # Try setting via boolean
        model.filter_conventional = True
        assert_equal(model.filter_method, FILTER_CONVENTIONAL)
        model.filter_conventional = False
        assert_equal(model.filter_method, 0)

        # Try setting directly via method
        model.set_filter_method(FILTER_CONVENTIONAL)
        assert_equal(model.filter_method, FILTER_CONVENTIONAL)

        # Try setting via boolean via method
        model.set_filter_method(filter_conventional=True)
        assert_equal(model.filter_method, FILTER_CONVENTIONAL)

    def test_inversion_methods(self):
        model = self.model

        # Clear the inversion method
        model.inversion_method = 0

        # Try setting via boolean
        model.invert_univariate = True
        assert_equal(model.inversion_method, INVERT_UNIVARIATE)
        model.invert_cholesky = True
        assert_equal(model.inversion_method, INVERT_UNIVARIATE | INVERT_CHOLESKY)
        model.invert_univariate = False
        assert_equal(model.inversion_method, INVERT_CHOLESKY)

        # Try setting directly via method
        model.set_inversion_method(INVERT_LU)
        assert_equal(model.inversion_method, INVERT_LU)

        # Try setting via boolean via method
        model.set_inversion_method(invert_cholesky=True, invert_univariate=True, invert_lu=False)
        assert_equal(model.inversion_method, INVERT_UNIVARIATE | INVERT_CHOLESKY)

        # Try setting and unsetting all
        model.inversion_method = 0
        for name in model.inversion_methods:
            setattr(model, name, True)
        assert_equal(
            model.inversion_method,
            INVERT_UNIVARIATE | SOLVE_LU | INVERT_LU | SOLVE_CHOLESKY |
            INVERT_CHOLESKY | INVERT_NUMPY
        )
        for name in model.inversion_methods:
            setattr(model, name, False)
        assert_equal(model.inversion_method, 0)

    def test_stability_methods(self):
        model = self.model

        # Clear the stability method
        model.stability_method = 0

        # Try setting via boolean
        model.stability_force_symmetry = True
        assert_equal(model.stability_method, STABILITY_FORCE_SYMMETRY)
        model.stability_force_symmetry = False
        assert_equal(model.stability_method, 0)

        # Try setting directly via method
        model.stability_method = 0
        model.set_stability_method(STABILITY_FORCE_SYMMETRY)
        assert_equal(model.stability_method, STABILITY_FORCE_SYMMETRY)

        # Try setting via boolean via method
        model.stability_method = 0
        model.set_stability_method(stability_method=True)
        assert_equal(model.stability_method, STABILITY_FORCE_SYMMETRY)


    def test_conserve_memory(self):
        model = self.model

        # Clear the filter method
        model.conserve_memory = MEMORY_STORE_ALL

        # Try setting via boolean
        model.memory_no_forecast = True
        assert_equal(model.conserve_memory, MEMORY_NO_FORECAST)
        model.memory_no_filtered = True
        assert_equal(model.conserve_memory, MEMORY_NO_FORECAST | MEMORY_NO_FILTERED)
        model.memory_no_forecast = False
        assert_equal(model.conserve_memory, MEMORY_NO_FILTERED)

        # Try setting directly via method
        model.set_conserve_memory(MEMORY_NO_PREDICTED)
        assert_equal(model.conserve_memory, MEMORY_NO_PREDICTED)

        # Try setting via boolean via method
        model.set_conserve_memory(memory_no_filtered=True, memory_no_predicted=False)
        assert_equal(model.conserve_memory, MEMORY_NO_FILTERED)

        # Try setting and unsetting all
        model.conserve_memory = 0
        for name in model.memory_options:
            if name == 'memory_conserve':
                continue
            setattr(model, name, True)
        assert_equal(
            model.conserve_memory,
            MEMORY_NO_FORECAST | MEMORY_NO_PREDICTED | MEMORY_NO_FILTERED |
            MEMORY_NO_LIKELIHOOD
        )
        assert_equal(model.conserve_memory, MEMORY_CONSERVE)
        for name in model.memory_options:
            if name == 'memory_conserve':
                continue
            setattr(model, name, False)
        assert_equal(model.conserve_memory, 0)
