"""
"Cholesky Factor Algorithm" (CFA) simulation smoothing for state space models

Author: Chad Fulton
License: BSD-3
"""

from . import tools


class CFASimulationSmoother(object):
    r"""
    "Cholesky Factor Algorithm" (CFA) simulation smoothing

    Parameters
    ----------
    model : Representation
        The state space model.

    References
    ----------
    .. [*] McCausland, William J., Shirley Miller, and Denis Pelletier.
           "Simulation smoothing for state-space models: A computational
           efficiency analysis."
           Computational Statistics & Data Analysis 55, no. 1 (2011): 199-212.
    .. [*] Chan, Joshua CC, and Ivan Jeliazkov.
           "Efficient simulation and integrated likelihood estimation in
           state space models."
           International Journal of Mathematical Modelling and Numerical
           Optimisation 1, no. 1-2 (2009): 101-120.
    """

    def __init__(self, model, cfa_simulation_smoother_classes=None):
        self.model = model

        # Get the simulation smoother classes
        self.prefix_simulation_smoother_map = (
            cfa_simulation_smoother_classes
            if cfa_simulation_smoother_classes is not None
            else tools.prefix_cfa_simulation_smoother_map.copy())

        self._simulation_smoothers = {}

        self._simulated_state = None

    @property
    def _simulation_smoother(self):
        prefix = self.model.prefix
        if prefix in self._simulation_smoothers:
            return self._simulation_smoothers[prefix]
        return None

    def simulate(self, variates=None, update_posterior=True):
        r"""
        Perform simulation smoothing (via Cholesky factor algorithm)

        Does not return anything, but populates the object's `simulated_state`
        attribute.

        Parameters
        ----------
        variates : array_like, optional
            Random variates, distributed standard Normal. Usually only
            specified if results are to be replicated (e.g. to enforce a seed)
            or for testing. If not specified, random variates are drawn. Must
            be shaped (nobs, k_states).
        """
        # (Re) initialize the _statespace representation
        prefix, dtype, create = self.model._initialize_representation()

        # Validate variates and get in required datatype
        if variates is not None:
            tools.validate_matrix_shape('variates', variates.shape,
                                        self.model.nobs,
                                        self.model.k_states, 1)
            variates = variates.ravel().astype(dtype)

        # (Re) initialize the state
        self.model._initialize_state(prefix=prefix)

        # Construct the Cython simulation smoother instance, if necessary
        if create or prefix not in self._simulation_smoothers:
            cls = self.prefix_simulation_smoother_map[prefix]
            self._simulation_smoothers[prefix] = cls(
                self.model._statespaces[prefix])
        sim = self._simulation_smoothers[prefix]

        # Update posterior moments, if requested
        if update_posterior:
            sim.update_sparse_posterior_moments()

        # Perform simulation smoothing
        self.simulated_state = sim.simulate(variates=variates)
