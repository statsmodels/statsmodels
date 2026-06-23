"""
Modified Efficient Importance Sampling (MEIS) for state space models.

Implements the MEIS algorithm of Koopman, Lit & Nguyen (2018) for
maximum likelihood estimation of partially non-Gaussian state space
models. The key idea is to construct a Gaussian importance density
by iteratively matching the curvature of the true observation density
via regression, then evaluate the likelihood using importance sampling
with bias correction.

Classes
-------
DurbinKoopmanSimulator
    Simulation smoother following Durbin & Koopman (2002).
MEISImportanceDensity
    Iterative construction of the Gaussian importance density.
MEISLikelihood
    MEIS-based log-likelihood evaluation with bias correction.
MEISMixin
    Mixin class adding ``fit_meis`` to ``MLEModel`` subclasses.
MEISResults
    Results container for MEIS estimation.

Functions
---------
extract_signal_meis
    Importance-weighted signal extraction.

References
----------
.. [1] Koopman, S. J., Lit, R., & Nguyen, T. M. (2018).
   Modified efficient importance sampling for partially non-Gaussian
   state space models. *Statistica Neerlandica*, 73(1), 44--62.
.. [2] Durbin, J., & Koopman, S. J. (2002). A simple and efficient
   simulation smoother for state space time series analysis.
   *Biometrika*, 89(3), 603--615.
"""
import warnings
import numpy as np
from scipy.linalg import cholesky
from statsmodels.tsa.statespace.representation import Representation
from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother
from statsmodels.tsa.statespace.mlemodel import MLEResults

# Numerical constants
_EPS = 1e-12
_MIN_C = 1e-6  # Minimum variance parameter
_MAX_C = 1e4   # Maximum c_t to prevent numerical explosion
_MAX_B = 1e6   # Maximum abs(b_t) to prevent numerical explosion
_THETA_CLIP = 30.0
_LOG_EXP_CLIP = 700.0
_RIDGE = 1e-10


# We use numpy.random.Generator (not the legacy np.random.seed / np.random.randn)
# because:
#   1. Generator is a *local* random state — it does not mutate the global
#      numpy RNG, so calling MEIS won't silently reset a user's random stream.
#   2. It is thread-safe: two concurrent MEIS calls each get their own state.
#   3. np.random.seed has been discouraged by NumPy since v1.17 (2019); new
#      code in statsmodels should prefer Generator (see NEP 19).
def _check_rng(rng):
    """Normalise *rng* to a ``numpy.random.Generator`` instance.

    Parameters
    ----------
    rng : {None, int, numpy.random.Generator}
        * ``None`` — return a new default Generator (unseeded).
        * ``int``  — return ``default_rng(seed)`` for reproducibility.
        * ``Generator`` — return as-is.

    Returns
    -------
    numpy.random.Generator
    """
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


# =============================================================================
# Durbin-Koopman Simulation Smoother (Durbin & Koopman, 2002)
# =============================================================================

class DurbinKoopmanSimulator:
    """
    Durbin-Koopman (2002) simulation smoother.

    Reference: Durbin, J., & Koopman, S. J. (2002). A simple and efficient
    simulation smoother for state space time series analysis. Biometrika,
    89(3), 603-615.
    """

    def __init__(self, model):
        self.model = model
        self.k_endog = int(model.k_endog)
        self.k_states = int(model.k_states)
        self.k_posdef = int(getattr(model, "k_posdef", model.k_states))
        self.nobs = int(model.nobs)

        # store a placeholder for the last simulated state draw
        self.simulated_state = None

    def simulate(self, rng=None):
        """
        Draw one sample from the smoothing distribution.

        Parameters
        ----------
        rng : {None, int, numpy.random.Generator}
            Random number generator instance or seed. If ``None``, the
            global NumPy default is used.
        """
        rng = _check_rng(rng)

        # Step 1-2: Sample disturbances and generate y⁺
        eps_plus, eta_plus, y_plus, alpha_plus = self._generate_plus(rng)

        # Step 3: Smooth y⁺
        _, _, alpha_bar_plus = self._smooth(y_plus)

        # Step 4: Smooth y (the real observations)
        _, _, alpha_bar = self._smooth(self.model.endog)

        # Step 5: Durbin-Koopman correction: α̂ = ᾱ - ᾱ⁺ + α⁺
        # alpha_bar and alpha_bar_plus are shaped (k_states, nobs)
        # alpha_plus returned (k_states, nobs)
        self.simulated_state = alpha_bar - alpha_bar_plus + alpha_plus

    def _generate_plus(self, rng):
        """Generate y⁺ from sampled disturbances using model matrices."""
        # Get matrices as time-varying 3D arrays with third axis == nobs
        Z = self._get_matrix("design")  # shape (k_endog, k_states, nobs)
        H = self._get_matrix("obs_cov")  # shape (k_endog, k_endog, nobs)
        T = self._get_matrix("transition")  # shape (k_states, k_states, nobs)
        R = self._get_matrix("selection")  # shape (k_states, k_posdef, nobs)
        Q = self._get_matrix("state_cov")  # shape (k_posdef, k_posdef, nobs)

        eps_plus = np.zeros((self.k_endog, self.nobs))
        eta_plus = np.zeros((self.k_posdef, self.nobs))
        # alpha_plus has length nobs+1 to handle initial state and subsequent transitions
        alpha_plus = np.zeros((self.k_states, self.nobs + 1))
        y_plus = np.zeros((self.k_endog, self.nobs))

        # Initial state
        a_0, P_0 = self._get_initial_state()
        try:
            L = cholesky(P_0 + _RIDGE * np.eye(self.k_states), lower=True)
            alpha_plus[:, 0] = a_0 + L @ rng.standard_normal(self.k_states)
        except Exception:
            alpha_plus[:, 0] = a_0

        # Forward pass
        for t in range(self.nobs):
            Z_t = Z[:, :, t]
            H_t = H[:, :, t]
            T_t = T[:, :, t]
            R_t = R[:, :, t]
            Q_t = Q[:, :, t]

            # Sample ε⁺ ~ N(0, H_t)
            try:
                L_H = cholesky(H_t + _RIDGE * np.eye(self.k_endog), lower=True)
                eps_plus[:, t] = L_H @ rng.standard_normal(self.k_endog)
            except Exception:
                eps_plus[:, t] = np.zeros(self.k_endog)

            # Sample η⁺ ~ N(0, Q_t)
            try:
                L_Q = cholesky(Q_t + _RIDGE * np.eye(self.k_posdef), lower=True)
                eta_plus[:, t] = L_Q @ rng.standard_normal(self.k_posdef)
            except Exception:
                eta_plus[:, t] = np.zeros(self.k_posdef)

            # Generate y⁺ and α⁺
            y_plus[:, t] = Z_t @ alpha_plus[:, t] + eps_plus[:, t]
            alpha_plus[:, t + 1] = T_t @ alpha_plus[:, t] + R_t @ eta_plus[:, t]

        # Return alpha_plus aligned to times 0..nobs-1 (exclude final state at nobs)
        return eps_plus, eta_plus, y_plus, alpha_plus[:, : self.nobs]

    def _smooth(self, endog):
        """Run Kalman smoother on given data and return smoothed states."""
        # Use KalmanSmoother (not the high-level model) to compute smoothed states
        ks = KalmanSmoother(self.k_endog, self.k_states, self.k_posdef)

        # Important: set ks.nobs before assigning time-varying matrices so validate_matrix_shape works
        try:
            ks.nobs = int(self.nobs)
        except Exception:
            # best-effort: if ks has a different attribute name, try setting via __dict__
            ks.__dict__["nobs"] = int(self.nobs)

        # Copy matrices from model to smoother (if present)
        for attr in ["design", "obs_cov", "transition", "selection", "state_cov"]:
            if hasattr(self.model, attr):
                setattr(ks, attr, getattr(self.model, attr))

        # Ensure endog is ndarray in the expected shape (nobs, k_endog)
        endog_arr = np.asarray(endog)
        # KalmanSmoother.bind expects (nobs, k_endog)
        if endog_arr.ndim == 1:
            endog_arr = endog_arr.reshape(-1, 1)
        ks.bind(endog_arr)

        # Initialize smoother
        if hasattr(self.model, "initialization"):
            ks.initialization = self.model.initialization
        else:
            a_0, P_0 = self._get_initial_state()
            try:
                ks.initialize_known(a_0, P_0)
            except Exception:
                # fallback: try approximate diffuse
                try:
                    ks.initialize_approximate_diffuse(1e6)
                except Exception:
                    pass

        # Run smoother; use ks.smooth() which returns a results-like object
        res = ks.smooth()
        # res.smoothed_state expected shape (k_states, nobs)
        alpha_bar = getattr(res, "smoothed_state", None)
        if alpha_bar is None:
            # Some interfaces may return slightly different attribute names; try to extract
            if hasattr(res, "state"):
                alpha_bar = np.asarray(res.state)
            else:
                raise RuntimeError("KalmanSmoother returned no smoothed_state in _smooth()")
        # eps_bar and eta_bar are not needed for MEIS but return placeholders
        eps_bar = np.zeros((self.k_endog, self.nobs))
        eta_bar = np.zeros((self.k_posdef, self.nobs))
        return eps_bar, eta_bar, alpha_bar

    def _get_initial_state(self):
        """Get initial state mean and covariance from model initialization if present."""
        # Check model.ssm.initialization first (for MLEModel subclasses)
        if hasattr(self.model, 'ssm') and hasattr(self.model.ssm, 'initialization'):
            init = self.model.ssm.initialization
            if hasattr(init, "constant") and hasattr(init, "stationary_cov"):
                return init.constant.copy(), init.stationary_cov.copy()
        # Then check model.initialization (for Representation-like objects)
        if hasattr(self.model, "initialization"):
            init = self.model.initialization
            if hasattr(init, "constant") and hasattr(init, "stationary_cov"):
                return init.constant.copy(), init.stationary_cov.copy()
        # default: moderate diffuse variance (1e4 instead of 1e6 for stability)
        return np.zeros(self.k_states), np.eye(self.k_states) * 1e4

    def _get_matrix(self, attr):
        """
        Get matrix for attr and coerce to a 3D array with shape (r, c, nobs),
        where r/c depend on attr and nobs == self.nobs.

        This function tolerantly handles inputs that are:
         - 2D arrays (r, c) and will be repeated over time
         - 3D arrays (r, c, 1) which will be repeated to length nobs
         - 3D arrays (r, c, nobs) returned unchanged

        It raises a ValueError if a provided 3D array has an incompatible time axis.
        """
        # Determine expected shape (rows, cols) for the attribute
        if attr == "design":
            r, c = self.k_endog, self.k_states
        elif attr == "obs_cov":
            r, c = self.k_endog, self.k_endog
        elif attr == "transition":
            r, c = self.k_states, self.k_states
        elif attr == "selection":
            r, c = self.k_states, self.k_posdef
        elif attr == "state_cov":
            r, c = self.k_posdef, self.k_posdef
        else:
            # generic fallback
            r = c = self.k_states

        M = getattr(self.model, attr, None)
        if M is None:
            # return identity-like default, repeated in time
            base = np.eye(r, c)
            return np.repeat(base[:, :, None], self.nobs, axis=2)

        M = np.asarray(M)
        if M.ndim == 2:
            if M.shape != (r, c):
                # allow some flexibility: try to transpose if dimensions swapped
                if M.shape == (c, r):
                    M = M.T
                else:
                    raise ValueError(
                        f"Provided {attr} has shape {M.shape}; expected {(r, c)} for attribute {attr}"
                    )
            return np.repeat(M[:, :, None], self.nobs, axis=2)

        if M.ndim == 3:
            if M.shape[0] != r or M.shape[1] != c:
                # try permutations to see if user supplied different axis ordering
                for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
                    Mp = np.transpose(M, perm)
                    if Mp.shape[0] == r and Mp.shape[1] == c:
                        M = Mp
                        break
                else:
                    raise ValueError(f"{attr} has incompatible leading dims {M.shape[:2]}; expected {(r, c)}")

            # now M has shape (r, c, k) for some k
            if M.shape[2] == self.nobs:
                return M.copy()
            if M.shape[2] == 1:
                return np.repeat(M, self.nobs, axis=2)
            # If 3rd axis is not 1 or nobs, it's a mismatch: raise to avoid silent errors
            raise ValueError(
                f"Invalid time axis length for {attr}: expected 1 or {self.nobs}, got {M.shape[2]}"
            )

        # Unexpected ndim
        raise ValueError(f"{attr} has unsupported ndim={M.ndim}")


# =============================================================================
# MEIS Implementation (Koopman et al., 2018)
# =============================================================================

class MEISImportanceDensity:
    r"""
    MEIS importance density for partially non-Gaussian state space models.

    Constructs a Gaussian approximation to the observation density by
    iteratively matching curvature via OLS regression on simulated
    signals.  The importance density has the form (equation 10 of [1]_):

    .. math::

        g(y_t | \alpha_t; \psi)
        = \exp\!\bigl(a_t + b_t \theta_t - \tfrac12 c_t \theta_t^2\bigr)

    where :math:`\theta_t = Z_t \alpha_t` is the signal.

    Parameters
    ----------
    model : MLEModel
        State space model that provides ``loglikelihood_obs(t, theta_t)``.
    M : int
        Number of importance samples per iteration.
    max_iter : int
        Maximum number of MEIS fitting iterations.
    tol : float
        Convergence tolerance on max change in *b* and *c*.

    Attributes
    ----------
    b_t : ndarray, shape (nobs, q_signal)
        Linear coefficients of the importance density.
    c_t : ndarray, shape (nobs, q_signal)
        Quadratic (precision) coefficients of the importance density.

    References
    ----------
    .. [1] Koopman, Lit & Nguyen (2018), Statistica Neerlandica.
    """

    def __init__(self, model, M=500, max_iter=50, tol=1e-3):
        self.model = model
        self.M = int(M)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        self.nobs = int(model.nobs)
        self.q_signal = int(getattr(model, 'q_signal', model.k_endog))

        # Initialize b_t and c_t
        # CRITICAL FIX: Initialize b_t using the data y_t, not zeros!
        # For a Gaussian model, the optimal b_t = y_t / obs_var.
        # Starting with b_t = y_t (assuming obs_var ~ 1) gives a much better
        # initial approximation than b_t = 0, leading to faster convergence.
        endog = model.endog if hasattr(model, 'endog') else model.ssm.endog
        endog_arr = np.asarray(endog)
        if endog_arr.ndim == 1:
            endog_arr = endog_arr.reshape(-1, 1)

        self.b_t = np.zeros((self.nobs, self.q_signal))
        # Initialize b_t with the observed data for each signal component
        for k in range(min(self.q_signal, endog_arr.shape[1])):
            self.b_t[:, k] = endog_arr[:, k]

        self.c_t = np.ones((self.nobs, self.q_signal))  # Start at 1.0 (standard Gaussian)

        # cache for last built approx (so simulate/logg/weights use same approx)
        self._last_approx = None

        self._fitted = False

    def fit(self, verbose=False, seed=None):
        """
        Fit importance density parameters iteratively.

        Uses the algorithm from Section 3.3 of Koopman et al. (2018):
        simulate signals from the current Gaussian approximation, then
        update ``b_t`` and ``c_t`` via OLS regression of
        ``log p(y_t | theta_t)`` on ``(1, theta_t, -0.5*theta_t**2)``.

        Parameters
        ----------
        verbose : bool
            If True, print convergence diagnostics each iteration.
        seed : {None, int, numpy.random.Generator}
            Seed or Generator for reproducibility.

        Returns
        -------
        self
        """
        rng = _check_rng(seed)

        for iteration in range(self.max_iter):
            b_old = self.b_t.copy()
            c_old = self.c_t.copy()

            # Step 2: Simulate θ from current approximation (equation 11)
            theta_draws = self.simulate_signal(self.b_t, self.c_t, seed=rng)

            # Step 3: Update β_t using OLS (equation 16)
            self._update_parameters(theta_draws)

            # Step 4: Check convergence
            b_change = np.max(np.abs(self.b_t - b_old))
            c_change = np.max(np.abs(self.c_t - c_old))

            # Print diagnostics for each iteration
            if verbose:
                print(f"[MEIS Iter {iteration+1}] "
                      f"max|db|={b_change:.6g}, max|dc|={c_change:.6g}")

            if b_change < self.tol and c_change < self.tol:
                if verbose:
                    print(f"[MEIS] Converged at iteration {iteration + 1}")
                break

        self._fitted = True
        return self

    def simulate_signal(self, b_t, c_t, M=None, seed=None):
        """
        Simulate signal draws from the Gaussian approximation.

        Builds a pseudo-observation model from ``b_t`` and ``c_t``, then
        draws ``M`` state vectors using the Durbin--Koopman simulation
        smoother (Section 3.3, Step 2).

        Parameters
        ----------
        b_t : ndarray, shape (nobs, q_signal)
            Current linear parameters.
        c_t : ndarray, shape (nobs, q_signal)
            Current precision parameters.
        M : int, optional
            Number of draws (defaults to ``self.M``).
        seed : {None, int, numpy.random.Generator}
            Seed or Generator for reproducibility.

        Returns
        -------
        theta_draws : ndarray, shape (M, q_signal, nobs)
        """
        if M is None:
            M = self.M
        rng = _check_rng(seed)

        # Build Gaussian approximation model (equation 11)
        approx = self._build_approximation(b_t, c_t)

        # Use Durbin-Koopman simulation smoother
        sim = DurbinKoopmanSimulator(approx)

        # Draw M samples
        theta_draws = np.zeros((M, self.q_signal, self.nobs))

        for i in range(M):
            sim.simulate(rng=rng)
            # Extract signal θ_t = Z_t(α_t) from first q_signal states
            theta_draws[i, :, :] = sim.simulated_state[:self.q_signal, :]

        return theta_draws

    def _build_approximation(self, b_t, c_t):
        """
        Build Gaussian approximation model for the MEIS importance density.

        Produces:
          - approx.endog = pseudo-data z (nobs, k_endog)
          - approx['obs_cov'] = H_approx (k_endog, k_endog, nobs)

        Also caches approx on self._last_approx for consistency.
        """
        endog = self.model.endog if hasattr(self.model, 'endog') else self.model.ssm.endog
        endog_arr = np.asarray(endog)
        if endog_arr.ndim == 1:
            endog_arr = endog_arr.reshape(-1, 1)

        approx = Representation(endog_arr, k_states=self.model.k_states)

        # copy state space matrices
        try:
            approx.nobs = int(self.model.ssm.nobs)
        except Exception:
            # best-effort: if ks has a different attribute name, try setting via __dict__
            approx.__dict__["nobs"] = int(self.model.ssm.nobs)

        for attr in ['design', 'transition', 'selection', 'state_cov']:
            if hasattr(self.model, attr):
                setattr(approx, attr, getattr(self.model, attr))
            elif hasattr(self.model, 'ssm'):
                setattr(approx, attr, getattr(self.model.ssm, attr))

        k_endog = int(self.model.k_endog)
        z = np.zeros((k_endog, self.nobs), dtype=float)
        H_approx = np.zeros((k_endog, k_endog, self.nobs), dtype=float)

        for t in range(self.nobs):
            # FIX 2: Build pseudo-observations x_t = b_t / c_t (equation 11)
            # No spurious a_t computation!
            for k in range(self.q_signal):
                b_tk = float(b_t[t, k])
                c_tk = max(float(c_t[t, k]), _MIN_C)  # ensure positive

                # x_t = b_t / c_t
                z[k, t] = b_tk / c_tk
                # Var(x_t) = 1 / c_t (diagonal H matrix)
                H_approx[k, k, t] = 1.0 / c_tk

            # if model has extra observation dims, anchor them to observed y with small variance
            if k_endog > self.q_signal:
                for kk in range(self.q_signal, k_endog):
                    z[kk, t] = endog_arr[t, kk] if endog_arr is not None else 0.0
                    H_approx[kk, kk, t] = 1e-6

        approx["obs_cov"] = H_approx
        approx.bind(z.T.copy())

        # Copy initialization from original model instead of always using diffuse
        # This is critical for convergence - using diffuse when the model has
        # known initialization causes the simulation smoother to produce draws
        # with incorrect variance, leading to poor OLS fits.
        if hasattr(self.model, 'ssm') and hasattr(self.model.ssm, 'initialization'):
            init = self.model.ssm.initialization
            if hasattr(init, 'constant') and hasattr(init, 'stationary_cov'):
                try:
                    approx.initialize_known(init.constant, init.stationary_cov)
                except Exception:
                    approx.initialize_approximate_diffuse(1e4)
            else:
                approx.initialize_approximate_diffuse(1e4)
        elif hasattr(self.model, 'initialization'):
            init = self.model.initialization
            if hasattr(init, 'constant') and hasattr(init, 'stationary_cov'):
                try:
                    approx.initialize_known(init.constant, init.stationary_cov)
                except Exception:
                    approx.initialize_approximate_diffuse(1e4)
            else:
                approx.initialize_approximate_diffuse(1e4)
        else:
            # Fallback to moderate diffuse (not 1e6 which causes numerical issues)
            approx.initialize_approximate_diffuse(1e4)

        # cache approx for consistent reuse
        self._last_approx = approx

        return approx

    def _update_parameters(self, theta_draws):
        """
        Update b_t and c_t via OLS regression.

        For each time *t* and signal component *k*, regresses
        ``log p(y_t | theta_t)`` on ``(1, theta_tk, -0.5*theta_tk**2)``
        and extracts updated *b* and *c* from the regression
        coefficients.
        """
        M = int(theta_draws.shape[0])

        for t in range(self.nobs):
            # Compute p_it (log p(y_t | theta_t^(i))) once per draw
            p_it = np.zeros(M)
            for i in range(M):
                theta_t = theta_draws[i, :, t]
                p_it[i] = self.model.loglikelihood_obs(t, theta_t)

            # Update each signal component k using unweighted OLS
            for k in range(self.q_signal):
                # Build design matrix V (M x 3)
                v_it = np.empty((M, 3), dtype=float)
                # Column 0: intercept
                v_it[:, 0] = 1.0
                # Column 1: theta_tk
                v_it[:, 1] = theta_draws[:, k, t]
                # Column 2: -0.5 * (theta_tk)^2
                v_it[:, 2] = -0.5 * (theta_draws[:, k, t] ** 2)

                # Normal equations: V'V beta = V'p
                try:
                    VtV = v_it.T @ v_it  # 3x3
                    Vtp = v_it.T @ p_it  # 3
                    # ridge regularization scaled to trace of VtV to be robust across scales
                    lam = _RIDGE * max(1.0, np.trace(VtV) / max(1, VtV.shape[0]))
                    beta = np.linalg.solve(VtV + lam * np.eye(3), Vtp)

                    # Extract b_t and c_t: beta = (a*, b_t, c_t)
                    # Clamp both to reasonable bounds to prevent explosion
                    self.b_t[t, k] = np.clip(beta[1], -_MAX_B, _MAX_B)
                    # FIX 1: c_t must be positive (it's a precision), use abs() and clamp
                    self.c_t[t, k] = np.clip(abs(beta[2]), _MIN_C, _MAX_C)
                except np.linalg.LinAlgError:
                    # fallback: least squares solution (more robust but slightly slower)
                    try:
                        beta = np.linalg.lstsq(VtV + _RIDGE * np.eye(3), Vtp, rcond=None)[0]
                        self.b_t[t, k] = np.clip(beta[1], -_MAX_B, _MAX_B)
                        self.c_t[t, k] = np.clip(abs(beta[2]), _MIN_C, _MAX_C)
                    except Exception:
                        # If even fallback fails, keep previous values (no update)
                        continue


class MEISLikelihood:
    """
    MEIS log-likelihood evaluator with bias correction.

    Implements equations (12), (13), and (17) of Koopman et al. (2018):
    simulate importance-weighted draws, evaluate ``log g(y; psi)`` via
    the Kalman filter on the Gaussian approximation, and apply the
    second-order bias correction of Section 4.1.

    Parameters
    ----------
    model : MLEModel
        State space model with ``loglikelihood_obs``.
    importance_density : MEISImportanceDensity
        Fitted importance density.
    """

    def __init__(self, model, importance_density):
        self.model = model
        self.importance_density = importance_density

    def compute_loglikelihood(self, M=None, seed=None):
        """
        Compute the MEIS log-likelihood estimate.

        Parameters
        ----------
        M : int, optional
            Number of importance samples (defaults to the value used
            when fitting the importance density).
        seed : {None, int, numpy.random.Generator}
            Seed or Generator for reproducibility.

        Returns
        -------
        loglik : float
            Estimated log-likelihood.
        u_bar : float
            Mean of centred importance weights (should be near 1).
        s2_u : float
            Variance of centred importance weights.
        """
        if M is None:
            M = self.importance_density.M

        # Simulate from importance density
        theta_draws = self.importance_density.simulate_signal(
            self.importance_density.b_t,
            self.importance_density.c_t,
            M=M,
            seed=seed
        )

        # Compute log g(y; ψ)
        log_g = self._compute_log_g()

        # Compute importance weights
        log_weights = self._compute_weights(theta_draws)

        # Compute likelihood with bias correction (equation 17, Section 4.1)
        a_bar = np.mean(log_weights)
        u = np.exp(np.clip(log_weights - a_bar, -_LOG_EXP_CLIP, _LOG_EXP_CLIP))
        u_bar = np.mean(u)
        s2_u = np.var(u, ddof=1) if M > 1 else 0.0

        # ℓ̂(y; ψ) = log g(y; ψ) + log w̄ + (1/2M) w̄^(-2) s²_w
        loglik = log_g + a_bar + np.log(u_bar + _EPS) + (s2_u / (2.0 * M * (u_bar + _EPS) ** 2))

        return float(loglik), float(u_bar), float(s2_u)

    def _compute_log_g(self):
        """Compute log g(y; psi) via the Kalman filter on the approximation."""
        try:
            # Prefer cached approx from importance_density if present
            imp = self.importance_density
            if hasattr(imp, "_last_approx") and getattr(imp, "_last_approx", None) is not None:
                approx = imp._last_approx
            else:
                approx = imp._build_approximation(imp.b_t, imp.c_t)

            ks = KalmanSmoother(approx.k_endog, approx.k_states, approx.k_posdef)
            try:
                ks.nobs = int(approx.nobs)  # preferred
            except Exception:
                ks.__dict__['nobs'] = int(approx.nobs)

            for attr in ['design', 'obs_cov', 'transition', 'selection', 'state_cov']:
                if hasattr(approx, attr):
                    setattr(ks, attr, getattr(approx, attr))

            ks.bind(approx.endog)
            if hasattr(approx, "initialization"):
                ks.initialization = approx.initialization

            res = ks.smooth()
            kalman_llf = float(res.llf) if hasattr(res, 'llf') else 0.0

            # Return Kalman filter likelihood of the approximating model
            return float(kalman_llf)
        except Exception as e:
            warnings.warn(f"Could not compute log g: {e}")
            return 0.0

    def _compute_weights(self, theta_draws):
        """Compute log importance weights log[p/g] (equation 12)."""
        M, q, nobs = theta_draws.shape
        log_weights = np.zeros(M)

        for i in range(M):
            log_w_i = 0.0
            for t in range(nobs):
                theta_t = np.clip(theta_draws[i, :, t], -_THETA_CLIP, _THETA_CLIP)

                # w(y_t, α_t) = p(y_t|α_t) / g(y_t|α_t)
                log_p = self.model.loglikelihood_obs(t, theta_t)

                # log g_t(θ) = a_t + sum_k (b_tk * θ_k - 0.5 * c_tk * θ_k^2)
                # where a_t = -(1/2) log(2π/c_t) - (1/2) b_t²/c_t (equation 10)
                log_g_t = 0.0

                for k in range(q):
                    b_tk = self.importance_density.b_t[t, k]
                    c_tk = max(self.importance_density.c_t[t, k], _MIN_C)
                    theta_tk = theta_t[k]

                    # Correct a_t formula (normalizing constant)
                    a_tk = -0.5 * np.log(2.0 * np.pi / c_tk) - 0.5 * (b_tk ** 2) / c_tk

                    log_g_t += a_tk + b_tk * theta_tk - 0.5 * c_tk * (theta_tk ** 2)

                log_w_i += (log_p - log_g_t)

            log_weights[i] = log_w_i

        return log_weights


class MEISMixin:
    """
    Mixin class that adds MEIS estimation to ``MLEModel`` subclasses.

    The user model must implement:

    * ``loglikelihood_obs(t, theta_t)`` -- return the scalar
      log-density ``log p(y_t | theta_t)``.
    * Standard ``MLEModel`` interface: ``update``, ``param_names``,
      ``start_params``, ``transform_params``, ``untransform_params``.

    Optionally:

    * ``q_signal`` attribute (int) -- dimension of the signal
      (defaults to ``k_endog``).
    * ``transform_states_to_signal(alpha)`` -- map state vector to
      signal (defaults to identity).
    """

    def _initialize_meis(self, M=500, max_iter=50, tol=1e-3):
        return MEISImportanceDensity(self, M=M, max_iter=max_iter, tol=tol)

    def fit_meis(self, start_params=None, M=500, meis_iter=50,
                 transformed=False, method='nm', maxiter=50,
                 disp=True, seed=None, **kwargs):
        """
        Estimate parameters by maximising the MEIS log-likelihood.

        Replaces the Kalman-filter-based ``loglike`` with a MEIS-based
        version, delegates to ``MLEModel.fit``, then restores the
        original likelihood.

        Parameters
        ----------
        start_params : array_like, optional
            Starting parameter values (constrained space). Automatically
            converted to unconstrained space when ``transformed=False``.
        M : int
            Number of importance samples.
        meis_iter : int
            Maximum MEIS fitting iterations per likelihood evaluation.
        transformed : bool
            If False (default), the optimiser works in unconstrained
            space, which prevents negative variances.
        method : str
            Optimisation method passed to ``MLEModel.fit``
            (e.g. ``'nm'``, ``'powell'``, ``'lbfgs'``).
        maxiter : int
            Maximum optimiser iterations.
        disp : bool
            Whether to display convergence information.
        seed : {None, int, numpy.random.Generator}
            Seed or Generator for reproducibility.
        **kwargs
            Additional keyword arguments passed to ``MLEModel.fit``.

        Returns
        -------
        MEISResults
        """
        self._meis_cache = {}
        original_loglike = getattr(self, 'loglike', None)

        # If start_params are constrained (default) and transformed=False,
        # convert to unconstrained space for the optimizer
        if start_params is not None and not transformed:
            start_params = np.asarray(start_params, dtype=float)
            # If start_params look constrained (all positive, reasonable),
            # convert to unconstrained
            try:
                start_params = self.untransform_params(start_params)
            except Exception:
                pass  # If untransform fails, use as-is

        def meis_loglike(params, *args, **kw):
            try:
                self.update(params, transformed=transformed)
            except TypeError:
                self.update(params)

            param_key = tuple(np.asarray(params).ravel().tolist())
            if param_key in self._meis_cache:
                return self._meis_cache[param_key]

            meis = self._initialize_meis(M=M, max_iter=meis_iter)
            meis.fit(seed=seed, verbose=False)

            likelihood = MEISLikelihood(self, meis)
            loglik, u_bar, s2_u = likelihood.compute_loglikelihood(seed=seed)

            self._meis_cache[param_key] = loglik

            if disp and (len(self._meis_cache) % 5 == 0):
                print(f"Eval {len(self._meis_cache)}: LogLik={loglik:.4f}")

            return loglik

        self.loglike = meis_loglike

        try:
            results = super().fit(
                start_params=start_params,
                transformed=transformed,
                method=method,
                maxiter=maxiter,
                disp=disp,
                **kwargs
            )
            results = MEISResults(self, results.params, results)
        finally:
            if original_loglike is None:
                delattr(self, 'loglike')
            else:
                self.loglike = original_loglike

        return results

    def smooth_signal_meis(self, params=None, M=100, meis_iter=10):
        """
        Compute the importance-weighted smoothed signal (Section 4.2).

        Parameters
        ----------
        params : array_like, optional
            Parameter values; if given, ``update(params)`` is called first.
        M : int
            Number of importance samples.
        meis_iter : int
            Maximum MEIS fitting iterations.

        Returns
        -------
        theta_smooth : ndarray, shape (q_signal, nobs)
            Weighted mean of signal draws.
        theta_draws : ndarray, shape (M, q_signal, nobs)
            Raw signal draws.
        weights : ndarray, shape (M,)
            Normalised importance weights.
        """
        if params is not None:
            self.update(params)

        meis = self._initialize_meis(M=M, max_iter=meis_iter)
        meis.fit(verbose=False, seed=42)

        return extract_signal_meis(self, meis, M=M)


class MEISResults(MLEResults):
    """
    Results class for MEIS estimation.

    Wraps a standard ``MLEResults`` object with MEIS-specific
    post-estimation methods.

    Parameters
    ----------
    model : MLEModel
        The estimated model.
    params : ndarray
        Estimated parameter vector.
    base_results : MLEResults
        The underlying results from ``MLEModel.fit``.
    """

    def __init__(self, model, params, base_results):
        self.__dict__.update(base_results.__dict__)
        self.model = model
        self.params = np.asarray(params)

    def smooth_signal(self, M=100, meis_iter=10):
        """
        Compute importance-weighted smoothed signal at estimated params.

        Parameters
        ----------
        M : int
            Number of importance samples.
        meis_iter : int
            Maximum MEIS fitting iterations.

        Returns
        -------
        theta_smooth : ndarray, shape (q_signal, nobs)
        theta_draws : ndarray, shape (M, q_signal, nobs)
        weights : ndarray, shape (M,)
        """
        return self.model.smooth_signal_meis(params=self.params, M=M,
                                             meis_iter=meis_iter)


def extract_signal_meis(model, meis, M=None, seed=None):
    """
    Extract the smoothed signal using importance-weighted averaging.

    Draws ``M`` signal trajectories from the fitted importance density,
    computes importance weights ``p / g``, and returns the weighted
    mean signal together with the raw draws and normalised weights.

    Parameters
    ----------
    model : MLEModel
        Model with ``loglikelihood_obs``.
    meis : MEISImportanceDensity
        Fitted importance density.
    M : int, optional
        Number of draws (defaults to ``meis.M``).
    seed : {None, int, numpy.random.Generator}
        Seed or Generator for reproducibility.

    Returns
    -------
    theta_smooth : ndarray, shape (q_signal, nobs)
        Weighted mean signal.
    theta_draws : ndarray, shape (M, q_signal, nobs)
        Individual signal draws.
    weights : ndarray, shape (M,)
        Normalised importance weights.
    """
    if M is None:
        M = meis.M

    theta_draws = meis.simulate_signal(meis.b_t, meis.c_t, M=M, seed=seed)

    M, q, nobs = theta_draws.shape
    log_weights = np.zeros(M)

    # Compute weights
    for i in range(M):
        log_w_i = 0.0
        for t in range(nobs):
            theta_t = np.clip(theta_draws[i, :, t], -_THETA_CLIP, _THETA_CLIP)
            log_p = model.loglikelihood_obs(t, theta_t)

            # Compute log g_t(θ) with correct a_t formula
            log_g_t = 0.0
            for k in range(q):
                b_tk = meis.b_t[t, k]
                c_tk = max(meis.c_t[t, k], _MIN_C)
                theta_tk = theta_t[k]

                # Correct a_tk formula (equation 10)
                a_tk = -0.5 * np.log(2.0 * np.pi / c_tk) - 0.5 * (b_tk ** 2) / c_tk

                log_g_t += a_tk + b_tk * theta_tk - 0.5 * c_tk * (theta_tk ** 2)

            log_w_i += (log_p - log_g_t)

        log_weights[i] = log_w_i

    # Normalize weights
    log_weights = log_weights - np.max(log_weights)
    weights = np.exp(np.clip(log_weights, -_LOG_EXP_CLIP, _LOG_EXP_CLIP))
    weights = weights / (np.sum(weights) + _EPS)

    # Weighted average
    theta_smooth = np.sum(theta_draws * weights[:, np.newaxis, np.newaxis], axis=0)

    return theta_smooth, theta_draws, weights


__all__ = [
    'DurbinKoopmanSimulator',
    'MEISImportanceDensity',
    'MEISLikelihood',
    'MEISMixin',
    'MEISResults',
    'extract_signal_meis',
]
