"""
Modified Efficient Importance Sampling (MEIS) for statsmodels

Complete implementation following Koopman, Lit, and Nguyen (2019) with
proper multivariate support.

This module implements MEIS for partially non-Gaussian state space models
following the paper "Modified efficient importance sampling for partially
non-Gaussian state space models", Statistica Neerlandica, 73(1), 44-62.

Key Features:
- Full multivariate observation support
- Vector-valued signals θ_t ∈ ℝ^q
- Efficient Kalman filter-based implementation
- Bias-corrected likelihood (Equations 17-18)
- Integration with statsmodels MLEModel

To integrate into statsmodels:
    Place in: statsmodels/tsa/statespace/meis.py
"""

import numpy as np
from scipy import optimize
from statsmodels.tsa.statespace.representation import Representation
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother
from statsmodels.tsa.statespace.mlemodel import MLEModel, MLEResults


# ============================================================================
# Simulation Smoother (Durbin & Koopman 2002)
# ============================================================================

class SimulationSmoother:
    """
    Simulation smoother for state space models.

    Implements the simulation smoothing algorithm of Durbin and Koopman (2002)
    for drawing from the conditional state density p(α|y).

    Parameters
    ----------
    model : Representation
        State space model from which to simulate

    References
    ----------
    Durbin, J., and Koopman, S.J. (2002). A simple and efficient simulation
    smoother for state space time series analysis. Biometrika, 89(3), 603-615.
    """

    def __init__(self, model):
        self.model = model
        self.smoother = KalmanSmoother(model)

    def simulate(self, seed=None):
        """
        Simulate state vector from p(α|y).

        Algorithm:
        1. Smooth actual data → E[α|y]
        2. Simulate unconditional α⁺, y⁺
        3. Smooth simulated data → E[α⁺|y⁺]
        4. Return: E[α|y] + α⁺ - E[α⁺|y⁺]

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        alpha_sim : ndarray (k_states, nobs)
            Simulated state vector
        """
        if seed is not None:
            np.random.seed(seed)

        # Step 1: Smooth actual data
        smoother_results = self.smoother.smooth()

        # Step 2: Simulate from unconditional model
        nobs = self.model.nobs
        k_states = self.model.k_states
        k_posdef = self.model.k_posdef
        k_endog = self.model.k_endog

        alpha_plus = np.zeros((k_states, nobs + 1))
        y_plus = np.zeros((k_endog, nobs))

        # Initial state
        init_mean = self.model.initialization.initial_state.ravel()
        init_cov = self.model.initialization.initial_state_cov
        alpha_plus[:, 0] = np.random.multivariate_normal(init_mean, init_cov)

        # Forward simulation
        for t in range(nobs):
            # Get system matrices at time t
            design = self.model['design', :, :, t]
            obs_cov = self.model['obs_cov', :, :, t]
            transition = self.model['transition', :, :, t]
            selection = self.model['selection', :, :, t]
            state_cov = self.model['state_cov', :, :, t]

            # Simulate observation
            obs_mean = design @ alpha_plus[:, t]
            y_plus[:, t] = np.random.multivariate_normal(obs_mean, obs_cov)

            # Simulate state
            state_mean = transition @ alpha_plus[:, t]
            eta = np.random.multivariate_normal(np.zeros(k_posdef), state_cov)
            alpha_plus[:, t + 1] = state_mean + selection @ eta

        # Step 3: Create model with simulated data and smooth
        model_plus = Representation(
            endog=y_plus.T,
            k_states=k_states,
            k_posdef=k_posdef
        )

        # Copy state space structure
        model_plus['design'] = self.model['design']
        model_plus['obs_cov'] = self.model['obs_cov']
        model_plus['transition'] = self.model['transition']
        model_plus['selection'] = self.model['selection']
        model_plus['state_cov'] = self.model['state_cov']
        model_plus.initialize(self.model.initialization)

        # Smooth simulated data
        smoother_plus = KalmanSmoother(model_plus)
        smooth_plus = smoother_plus.smooth()

        # Step 4: Combine (Durbin & Koopman equation 7)
        alpha_sim = (smoother_results.smoothed_state +
                     alpha_plus[:, :-1] -
                     smooth_plus.smoothed_state)

        return alpha_sim


# ============================================================================
# MEIS Importance Density
# ============================================================================

class MEISImportanceDensity:
    """
    MEIS importance density for partially non-Gaussian state space models.

    Constructs a Gaussian importance density g(y|α;ψ) that approximates
    the non-Gaussian observation density p(y|α;ψ) following Section 3 of
    Koopman, Lit, and Nguyen (2019).

    The importance density is parameterized by b_t and c_t for each time t
    and observation dimension, defined by the Gaussian kernel:

        g(y_t|α_t) ∝ exp(a_t + b_t'θ_t - 0.5 θ_t'C_t θ_t)     (Eq. 10)

    where θ_t = Z_t(α_t) is the signal and C_t = diag(c_t).

    Parameters
    ----------
    model : object with methods
        - transform_states_to_signal(alpha) : Returns (q_signal, nobs) array
        - loglikelihood_obs(t, theta_t) : Returns scalar log p(y_t|θ_t)
        - ssm : State space representation
    M : int, optional
        Number of importance sampling draws. Default is 100.
    max_iter : int, optional
        Maximum MEIS iterations. Default is 50.
    tol : float, optional
        Convergence tolerance. Default is 1e-3.

    Attributes
    ----------
    b_t : ndarray (nobs, q_signal)
        Importance density parameter b_t
    c_t : ndarray (nobs, q_signal)
        Importance density parameter c_t (must be positive)
    converged : bool
        Whether MEIS algorithm converged

    References
    ----------
    Koopman, S.J., Lit, R., and Nguyen, T.M. (2019). Modified efficient
    importance sampling for partially non-Gaussian state space models.
    Statistica Neerlandica, 73(1), 44-62.
    """

    def __init__(self, model, M=100, max_iter=50, tol=1e-3):
        self.model = model
        self.M = M
        self.max_iter = max_iter
        self.tol = tol

        # Determine signal dimension q from model
        test_alpha = np.zeros((model.k_states, 1))
        test_signal = model.transform_states_to_signal(test_alpha)

        if test_signal.ndim == 1:
            self.q_signal = 1
        else:
            self.q_signal = test_signal.shape[0]

        # Initialize parameters
        self.b_t = None  # Shape: (nobs, q_signal)
        self.c_t = None  # Shape: (nobs, q_signal)
        self.converged = False
        self.theta_draws = None

    def construct_approximation_model(self, b_t, c_t):
        """
        Construct linear Gaussian approximation model (Section 3.1, Eq. 11).

        Creates artificial observations x_t = b_t / c_t with observation
        equation: x_t = θ_t + u_t, where u_t ~ N(0, diag(1/c_t)).

        Parameters
        ----------
        b_t : ndarray (nobs, q_signal)
            b parameters for each time and signal dimension
        c_t : ndarray (nobs, q_signal)
            c parameters (precision) for each time and signal dimension

        Returns
        -------
        approx_model : Representation
            Linear Gaussian state space model for importance density
        """
        nobs = self.model.nobs
        q = self.q_signal
        k_states = self.model.k_states
        k_posdef = self.model.k_posdef

        # Artificial observations: x_t = b_t / c_t (Eq. 11)
        x_t = b_t / np.maximum(c_t, 1e-10)  # Avoid division by zero

        # Create representation with q artificial observations
        approx_model = Representation(
            endog=x_t,  # Shape: (nobs, q_signal)
            k_states=k_states,
            k_posdef=k_posdef
        )

        # Build observation equation for artificial data
        # x_t = Z_design @ α_t + u_t,  u_t ~ N(0, diag(1/c_t))
        for t in range(nobs):
            # Design matrix: maps states to signal
            # For linear signal θ_t = Z @ α_t, use model's design
            # For nonlinear signal, user must provide signal_design
            if hasattr(self.model, 'signal_design'):
                Z_t = self.model.signal_design[:, :, t]
            else:
                # Assume signal uses same design as observations
                Z_t = self.model.ssm['design', :, :, t]
                if Z_t.shape[0] != q:
                    # If dimensions don't match, extract first q rows
                    Z_t = Z_t[:q, :]

            approx_model['design', :, :, t] = Z_t

            # Observation covariance: diag(1/c_t) (Eq. 11)
            obs_cov_t = np.diag(1.0 / np.maximum(c_t[t], 1e-8))
            approx_model['obs_cov', :, :, t] = obs_cov_t

            # State equation (same as true model)
            approx_model['transition', :, :, t] = self.model.ssm['transition', :, :, t]
            approx_model['selection', :, :, t] = self.model.ssm['selection', :, :, t]
            approx_model['state_cov', :, :, t] = self.model.ssm['state_cov', :, :, t]

        # Initialize state (same as true model)
        approx_model.initialize(self.model.ssm.initialization)

        return approx_model

    def simulate_signal(self, b_t, c_t, seed=None):
        """
        Simulate signal draws θ_t from importance density g(α|y).

        Uses simulation smoothing (Durbin & Koopman 2002) to draw state
        trajectories from the Gaussian approximation, then transforms to
        signal space.

        Parameters
        ----------
        b_t : ndarray (nobs, q_signal)
        c_t : ndarray (nobs, q_signal)
        seed : int, optional
            Random seed

        Returns
        -------
        theta_draws : ndarray (M, q_signal, nobs)
            Simulated signals from importance density
        alpha_draws : ndarray (M, k_states, nobs)
            Simulated states
        """
        if seed is not None:
            np.random.seed(seed)

        # Build Gaussian approximation
        approx_model = self.construct_approximation_model(b_t, c_t)

        # Create simulation smoother
        sim_smoother = SimulationSmoother(approx_model)

        # Generate M draws
        nobs = self.model.nobs
        k_states = self.model.k_states
        q = self.q_signal

        theta_draws = np.zeros((self.M, q, nobs))
        alpha_draws = np.zeros((self.M, k_states, nobs))

        for i in range(self.M):
            # Simulate state trajectory
            alpha_sim = sim_smoother.simulate(seed=seed + i if seed else None)
            alpha_draws[i] = alpha_sim

            # Transform to signal: θ_t = Z_t(α_t)
            theta_sim = self.model.transform_states_to_signal(alpha_sim)

            # Handle scalar vs vector signals
            if theta_sim.ndim == 1:
                theta_draws[i, 0, :] = theta_sim
            else:
                theta_draws[i, :, :] = theta_sim

        return theta_draws, alpha_draws

    def weighted_least_squares(self, theta_draws, b_t_old, c_t_old):
        """
        Update b_t and c_t via weighted least squares (Eq. 16).

        For each time t and signal dimension k, performs WLS regression:

            min_{b_{kt}, c_{kt}} Σ_i w_i [log p(y_t|θ_t^i) - (a_t + b_{kt}θ_{kt}^i - 0.5 c_{kt}(θ_{kt}^i)²)]²

        where w_i are importance weights and θ_t^i is draw i at time t.

        Parameters
        ----------
        theta_draws : ndarray (M, q_signal, nobs)
            Current signal draws
        b_t_old : ndarray (nobs, q_signal)
            Previous b parameters
        c_t_old : ndarray (nobs, q_signal)
            Previous c parameters

        Returns
        -------
        b_t : ndarray (nobs, q_signal)
            Updated b parameters
        c_t : ndarray (nobs, q_signal)
            Updated c parameters
        """
        nobs = self.model.nobs
        q = self.q_signal

        b_t = np.zeros((nobs, q))
        c_t = np.zeros((nobs, q))

        for t in range(nobs):
            for k in range(q):
                # Extract signal dimension k at time t across all draws
                theta_kt = theta_draws[:, k, t]  # Shape: (M,)

                # Design matrix for quadratic regression: [1, θ, -0.5θ²]
                X = np.column_stack([
                    np.ones(self.M),
                    theta_kt,
                    -0.5 * theta_kt ** 2
                ])

                # Response: log p(y_t | θ_t) for each draw
                y_reg = np.array([
                    self.model.loglikelihood_obs(t, theta_draws[i, :, t])
                    for i in range(self.M)
                ])

                # Importance weights w(y_t, θ_t) = p(y_t|θ_t) / g(y_t|θ_t)
                weights = self._compute_weights(
                    t, theta_draws[:, :, t], b_t_old[t], c_t_old[t]
                )

                # Stabilize weights (avoid numerical issues)
                weights = np.clip(weights, 1e-10, 1e10)
                weights = weights / (np.sum(weights) + 1e-10) * self.M

                # Weighted least squares: β = (X'WX)^{-1} X'Wy
                W = np.diag(weights)
                try:
                    beta_kt = np.linalg.solve(X.T @ W @ X, X.T @ W @ y_reg)
                    # β = [a_t, b_t, c_t]
                    b_t[t, k] = beta_kt[1]
                    c_t[t, k] = max(beta_kt[2], 1e-6)  # Ensure c_t > 0
                except np.linalg.LinAlgError:
                    # Singular matrix - use previous values
                    b_t[t, k] = b_t_old[t, k]
                    c_t[t, k] = c_t_old[t, k]

        return b_t, c_t

    def _compute_weights(self, t, theta_values, b_t, c_t):
        """
        Compute importance weights (Eq. 12).

        w(y_t, θ_t) = p(y_t|θ_t) / g(y_t|θ_t)

        Parameters
        ----------
        t : int
            Time index
        theta_values : ndarray (M, q_signal)
            Signal values for all draws at time t
        b_t : ndarray (q_signal,)
            b parameters at time t
        c_t : ndarray (q_signal,)
            c parameters at time t

        Returns
        -------
        weights : ndarray (M,)
            Importance weights for each draw
        """
        M = theta_values.shape[0]
        q = self.q_signal

        # True observation log-density p(y_t|θ_t)
        log_p = np.array([
            self.model.loglikelihood_obs(t, theta_values[i, :])
            for i in range(M)
        ])

        # Gaussian approximation log-density g(y_t|θ_t)
        # From Eq. 10: log g ∝ b_t'θ_t - 0.5 θ_t'C_t θ_t
        # Full density: -0.5 log(2π) + 0.5 log(c_t) + ...
        log_g = np.zeros(M)
        for j in range(q):
            theta_j = theta_values[:, j]
            log_g += (b_t[j] * theta_j - 0.5 * c_t[j] * theta_j ** 2 -
                      0.5 * np.log(2 * np.pi) + 0.5 * np.log(max(c_t[j], 1e-10)))

        # Weights: w = p / g
        weights = np.exp(log_p - log_g)

        return weights

    def fit(self, seed=None, verbose=False):
        """
        Run MEIS algorithm to find optimal b_t and c_t (Section 3.3).

        Algorithm:
        1. Initialize b_t, c_t
        2. Simulate θ_t from g(α|y) using current b_t, c_t
        3. Update b_t, c_t via weighted least squares
        4. Check convergence; repeat from step 2 if needed

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
        verbose : bool, optional
            Print iteration information

        Returns
        -------
        self : MEISImportanceDensity
            Fitted importance density
        """
        nobs = self.model.nobs
        q = self.q_signal

        # Step 1: Initialize parameters
        if self.b_t is None:
            self.b_t = np.zeros((nobs, q))
            self.c_t = np.ones((nobs, q))

        b_t_old = self.b_t.copy()
        c_t_old = self.c_t.copy()

        # Iterative optimization
        for iteration in range(self.max_iter):
            # Step 2: Simulate from current importance density
            theta_draws, alpha_draws = self.simulate_signal(
                b_t_old, c_t_old, seed=seed
            )

            # Step 3: Update parameters via WLS
            b_t_new, c_t_new = self.weighted_least_squares(
                theta_draws, b_t_old, c_t_old
            )

            # Step 4: Check convergence
            rel_change_b = np.max(np.abs(
                (b_t_new - b_t_old) / (np.abs(b_t_old) + 1e-10)
            ))
            rel_change_c = np.max(np.abs(
                (c_t_new - c_t_old) / (c_t_old + 1e-10)
            ))
            max_change = max(rel_change_b, rel_change_c)

            if verbose:
                print(f"  MEIS iter {iteration + 1}: max change = {max_change:.6f}")

            if max_change < self.tol:
                self.converged = True
                self.b_t = b_t_new
                self.c_t = c_t_new
                self.theta_draws = theta_draws
                if verbose:
                    print(f"  MEIS converged in {iteration + 1} iterations")
                break

            b_t_old = b_t_new
            c_t_old = c_t_new

        if not self.converged:
            self.b_t = b_t_new
            self.c_t = c_t_new
            self.theta_draws = theta_draws
            if verbose:
                print(f"  MEIS did not converge after {self.max_iter} iterations")

        return self


# ============================================================================
# MEIS Likelihood Computation
# ============================================================================

class MEISLikelihood:
    """
    Likelihood evaluation via MEIS importance sampling.

    Computes bias-corrected log-likelihood estimate using the MEIS
    importance density (Section 4.1, Equations 17-18).

    Parameters
    ----------
    model : object
        Non-Gaussian state space model
    importance_density : MEISImportanceDensity
        Fitted MEIS importance density
    """

    def __init__(self, model, importance_density):
        self.model = model
        self.importance_density = importance_density

    def compute_loglikelihood(self, M=None, seed=None):
        """
        Compute bias-corrected log-likelihood (Eq. 17).

        L̂(y;ψ) = g(y;ψ) · M^{-1} Σ_i Π_t w(y_t, θ_t^i)

        with bias correction:
        log L̂ = log g + log w̄ + (1/2M) · s²_u / w̄²

        Parameters
        ----------
        M : int, optional
            Number of draws. If None, uses importance_density.M
        seed : int, optional
            Random seed

        Returns
        -------
        loglik : float
            Bias-corrected log-likelihood estimate
        u_bar : float
            Mean of normalized importance weights
        s2_u : float
            Variance of normalized importance weights
        """
        if M is None:
            M = self.importance_density.M

        # Simulate from importance density
        theta_draws, _ = self.importance_density.simulate_signal(
            self.importance_density.b_t,
            self.importance_density.c_t,
            seed=seed
        )

        # Compute log g(y;ψ) via Kalman filter
        approx_model = self.importance_density.construct_approximation_model(
            self.importance_density.b_t,
            self.importance_density.c_t
        )
        kf = KalmanFilter(approx_model)
        kf_results = kf.filter()
        log_g = kf_results.llf

        # Compute log importance weights (Eq. 18)
        log_weights = np.zeros(M)
        q = self.importance_density.q_signal

        for i in range(M):
            log_w_i = 0
            for t in range(self.model.nobs):
                # True observation log-likelihood
                theta_t = theta_draws[i, :, t]  # (q_signal,) vector
                log_p = self.model.loglikelihood_obs(t, theta_t)

                # Gaussian approximation log-likelihood
                log_g_t = 0
                for k in range(q):
                    b_tk = self.importance_density.b_t[t, k]
                    c_tk = self.importance_density.c_t[t, k]
                    theta_tk = theta_t[k]

                    log_g_t += (b_tk * theta_tk - 0.5 * c_tk * theta_tk ** 2 -
                                0.5 * np.log(2 * np.pi) + 0.5 * np.log(c_tk))

                log_w_i += log_p - log_g_t

            log_weights[i] = log_w_i

        # Bias correction (Eq. 18)
        a_bar = np.mean(log_weights)
        u = np.exp(log_weights - a_bar)
        u_bar = np.mean(u)
        s2_u = np.var(u, ddof=1)

        # Bias-corrected log-likelihood (Eq. 17)
        loglik = log_g + a_bar + np.log(u_bar) + (s2_u / (2 * M * u_bar ** 2))

        return loglik, u_bar, s2_u


# ============================================================================
# MEIS Mixin for MLEModel
# ============================================================================

class MEISMixin:
    """
    Mixin class adding MEIS functionality to state space models.

    Use with multiple inheritance:
        class MyModel(MEISMixin, MLEModel):
            ...

    The model must implement:
    - loglikelihood_obs(t, theta_t): Return log p(y_t|θ_t)
    - transform_states_to_signal(alpha): Return θ = Z(α)

    Example
    -------
    >>> class StochasticVolatility(MEISMixin, MLEModel):
    ...     def loglikelihood_obs(self, t, theta_t):
    ...         # Student's t density
    ...         return stats.t.logpdf(self.endog[t], df=self.nu,
    ...                               loc=self.mu, scale=np.exp(0.5*theta_t[0]))
    ...
    ...     def transform_states_to_signal(self, alpha):
    ...         return np.sum(alpha, axis=0).reshape(1, -1)
    >>>
    >>> model = StochasticVolatility(data)
    >>> results = model.fit_meis(M=100)
    """

    def _initialize_meis(self, M=500, max_iter=50, tol=1e-3):
        """Initialize MEIS importance density."""
        return MEISImportanceDensity(self, M=M, max_iter=max_iter, tol=tol)

    def fit_meis(self, start_params=None, M=500, meis_iter=50,
                 transformed=True, includes_fixed=False, method='lbfgs',
                 maxiter=50, full_output=1, disp=True, callback=None,
                 return_params=False, optim_score=None, optim_complex_step=None,
                 optim_hessian=None, **kwargs):
        """
        Fit model using MEIS for likelihood evaluation.

        Extends MLEModel.fit() to use MEIS importance sampling for
        likelihood evaluation in non-Gaussian models.

        Parameters
        ----------
        start_params : array_like, optional
            Initial parameter guess
        M : int, optional
            Number of importance samples. Default is 500.
        meis_iter : int, optional
            Maximum MEIS iterations per likelihood evaluation. Default is 10.
        method : str, optional
            Optimization method. Default is 'lbfgs'.
        maxiter : int, optional
            Maximum optimization iterations
        disp : bool, optional
            Print convergence messages
        **kwargs
            Additional arguments for optimizer

        Returns
        -------
        results : MEISResults
            Results object with parameter estimates
        """
        # Store MEIS settings
        self._meis_M = M
        self._meis_iter = meis_iter
        self._meis_cache = {}

        # Override loglike for MEIS
        original_loglike = self.loglike

        def meis_loglike(params, *args, **kwargs):
            """Compute log-likelihood using MEIS."""
            # Update model
            self.update(params, transformed=transformed,
                        includes_fixed=includes_fixed)

            # Check cache
            param_key = tuple(params)
            if param_key in self._meis_cache:
                return self._meis_cache[param_key]

            # Fit MEIS importance density
            meis = self._initialize_meis(M=M, max_iter=meis_iter, tol=1e-3)
            meis.fit(verbose=False)

            # Compute likelihood
            likelihood = MEISLikelihood(self, meis)
            loglik, u_bar, s2_u = likelihood.compute_loglikelihood()

            # Cache result
            self._meis_cache[param_key] = loglik

            if disp and len(self._meis_cache) % 5 == 0:
                print(f"Eval {len(self._meis_cache)}: LogLik={loglik:.4f}, "
                      f"s²_u={s2_u:.4f}")

            return loglik

        # Replace loglike temporarily
        self.loglike = meis_loglike

        try:
            # Call parent fit
            results = super().fit(
                start_params=start_params,
                transformed=transformed,
                includes_fixed=includes_fixed,
                method=method,
                maxiter=maxiter,
                full_output=full_output,
                disp=disp,
                callback=callback,
                return_params=return_params,
                optim_score=optim_score,
                optim_complex_step=optim_complex_step,
                optim_hessian=optim_hessian,
                **kwargs
            )

            # Wrap in MEIS results
            if not return_params:
                results = MEISResults(self, results.params, results)

        finally:
            # Restore original loglike
            self.loglike = original_loglike

        return results

    def smooth_signal_meis(self, params=None, M=100, meis_iter=10):
        """
        Extract smoothed signal θ_t using MEIS importance sampling.

        Parameters
        ----------
        params : array_like, optional
            Model parameters. If None, uses current parameters.
        M : int, optional
            Number of importance samples
        meis_iter : int, optional
            Maximum MEIS iterations

        Returns
        -------
        theta_smooth : ndarray (q_signal, nobs)
            Smoothed signal values
        theta_draws : ndarray (M, q_signal, nobs)
            Individual signal draws
        weights : ndarray (M,)
            Normalized importance weights
        """
        if params is not None:
            self.update(params)

        # Fit MEIS importance density
        meis = self._initialize_meis(M=M, max_iter=meis_iter)
        meis.fit(verbose=False)

        # Extract signal
        return extract_signal_meis(self, meis, M=M)


# ============================================================================
# MEIS Results Class
# ============================================================================

class MEISResults(MLEResults):
    """
    Results class for models estimated with MEIS.

    Extends MLEResults with MEIS-specific methods.
    """

    def __init__(self, model, params, base_results):
        # Copy base results
        self.__dict__.update(base_results.__dict__)
        self.model = model
        self.params = params

    def smooth_signal(self, M=100, meis_iter=10):
        """
        Extract smoothed signal using MEIS.

        Parameters
        ----------
        M : int, optional
            Number of importance samples
        meis_iter : int, optional
            Maximum MEIS iterations

        Returns
        -------
        theta_smooth : ndarray
            Smoothed signal estimates
        theta_draws : ndarray
            Individual draws
        weights : ndarray
            Importance weights
        """
        return self.model.smooth_signal_meis(
            params=self.params, M=M, meis_iter=meis_iter
        )


# ============================================================================
# Utility Functions
# ============================================================================

def extract_signal_meis(model, meis, M=None, seed=None):
    """
    Extract smoothed signal θ_t using MEIS importance sampling.

    Computes weighted average:
        θ̃_t = Σ_i w_i θ_t^i / Σ_i w_i

    where w_i are importance weights.

    Parameters
    ----------
    model : object
        State space model
    meis : MEISImportanceDensity
        Fitted MEIS importance density
    M : int, optional
        Number of draws
    seed : int, optional
        Random seed

    Returns
    -------
    theta_smooth : ndarray (q_signal, nobs)
        Weighted average of signal draws
    theta_draws : ndarray (M, q_signal, nobs)
        Individual signal draws
    weights : ndarray (M,)
        Normalized importance weights
    """
    if M is None:
        M = meis.M

    # Simulate from importance density
    theta_draws, _ = meis.simulate_signal(meis.b_t, meis.c_t, seed=seed)

    # Compute importance weights
    log_weights = np.zeros(M)
    q = meis.q_signal

    for i in range(M):
        log_w_i = 0
        for t in range(model.nobs):
            # True observation log-likelihood
            theta_t = theta_draws[i, :, t]
            log_p = model.loglikelihood_obs(t, theta_t)

            # Gaussian approximation log-likelihood
            log_g_t = 0
            for k in range(q):
                b_tk = meis.b_t[t, k]
                c_tk = meis.c_t[t, k]
                theta_tk = theta_t[k]

                log_g_t += (b_tk * theta_tk - 0.5 * c_tk * theta_tk ** 2 -
                            0.5 * np.log(2 * np.pi) + 0.5 * np.log(c_tk))

            log_w_i += log_p - log_g_t

        log_weights[i] = log_w_i

    # Normalize weights
    log_weights = log_weights - np.max(log_weights)
    weights = np.exp(log_weights)
    weights = weights / np.sum(weights)

    # Weighted average: θ̃_t = Σ w_i θ_t^i
    theta_smooth = np.sum(
        theta_draws * weights[:, np.newaxis, np.newaxis],
        axis=0
    )

    return theta_smooth, theta_draws, weights


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'SimulationSmoother',
    'MEISImportanceDensity',
    'MEISLikelihood',
    'MEISMixin',
    'MEISResults',
    'extract_signal_meis',
]