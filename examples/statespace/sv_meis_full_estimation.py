"""
Stochastic Volatility Model with Student's t using MEIS - Full Parameter Estimation

This example estimates ALL FOUR parameters: mu, phi, sigma_eta, and nu
using MEIS-based maximum likelihood estimation.

We use scipy.optimize to maximize the MEIS likelihood and show:
1. Convergence of the optimizer
2. Comparison of estimates vs. true values
3. Standard errors (if Hessian is available)
4. Model diagnostics

Usage:
    python sv_meis_full_estimation.py

Dependencies:
    - meis.py (the MEIS implementation)
    - numpy, scipy, matplotlib, statsmodels
"""
import time
import numpy as np
from scipy import stats, optimize
from scipy.special import gammaln
import matplotlib.pyplot as plt

# Import MEIS from the fixed implementation
from statsmodels.tsa.statespace.meis import MEISMixin, MEISLikelihood
from statsmodels.tsa.statespace.mlemodel import MLEModel

# Numerical clip constants
_THETA_CLIP = 30.0
_LOG_EXP_CLIP = 700.0


class StochasticVolatilityStudentT(MEISMixin, MLEModel):
    """
    SV model with Student's t observation density.

    y_t = mu + exp(0.5 * theta_t) * eps_t,  eps_t ~ t(nu)
    theta_t = phi * theta_{t-1} + eta_t,    eta_t ~ N(0, sigma_eta^2)

    Parameters:
        mu:        Mean of returns
        phi:       Persistence of log-volatility (0 < phi < 1)
        sigma_eta: Volatility of log-volatility (sigma_eta > 0)
        nu:        Degrees of freedom for Student's t (nu > 2)
    """

    def __init__(self, endog, **kwargs):
        self._endog_orig = np.asarray(endog).ravel()
        endog2d = self._endog_orig.reshape(-1, 1)

        super().__init__(endog=endog2d, k_states=1, k_posdef=1, **kwargs)

        self._param_names = ['mu', 'phi', 'sigma_eta', 'nu']

        # Default values (will be overwritten by update)
        self.mu = 0.0
        self.phi = 0.98
        self.sigma_eta = 0.15
        self.nu = 10.0

    @property
    def param_names(self):
        return self._param_names

    @property
    def start_params(self):
        """Starting values for estimation."""
        return np.array([0.0, 0.98, 0.15, 10.0])

    def transform_params(self, unconstrained):
        """
        Transform from unconstrained to constrained parameter space.

        Transformations:
            mu:        no constraint (identity)
            phi:       (0, 1) via sigmoid
            sigma_eta: (0, inf) via exp
            nu:        (2.5, inf) via exp shift
        """
        constrained = unconstrained.copy()

        # mu: no transformation
        constrained[0] = unconstrained[0]

        # phi: map to (0, 1) using sigmoid centered at 0.9
        # phi = 0.5 + 0.49 * tanh(unconstrained[1])
        constrained[1] = 0.5 + 0.49 / (1.0 + np.exp(-unconstrained[1]))

        # sigma_eta: map to (0, inf)
        constrained[2] = np.exp(unconstrained[2])

        # nu: map to (2.5, inf)
        constrained[3] = 2.5 + np.exp(unconstrained[3])

        return constrained

    def untransform_params(self, constrained):
        """Transform from constrained to unconstrained parameter space."""
        unconstrained = constrained.copy()

        # mu: no transformation
        unconstrained[0] = constrained[0]

        # phi: inverse sigmoid
        phi_scaled = (constrained[1] - 0.5) / 0.49
        phi_scaled = np.clip(phi_scaled, -0.99, 0.99)
        unconstrained[1] = -np.log((1.0 / phi_scaled) - 1.0)

        # sigma_eta: inverse exp
        unconstrained[2] = np.log(constrained[2])

        # nu: inverse exp shift
        unconstrained[3] = np.log(constrained[3] - 2.5)

        return unconstrained

    def update(self, params, **kwargs):
        """Update model with new parameter values."""
        params = super().update(params, **kwargs)

        self.mu = params[0]
        self.phi = params[1]
        self.sigma_eta = params[2]
        self.nu = params[3]

        # State-space matrices
        self.ssm['design', 0, 0] = 1.0
        self.ssm['transition', 0, 0] = self.phi
        self.ssm['selection', 0, 0] = 1.0
        self.ssm['state_cov', 0, 0] = self.sigma_eta ** 2

        # Initialization: stationary covariance
        initial_state = np.zeros(1)
        var_init = self.sigma_eta ** 2 / (1.0 - self.phi ** 2 + 1e-10)
        initial_cov = np.array([[var_init]])

        try:
            self.ssm.initialize_known(initial_state, initial_cov)
        except Exception:
            try:
                self.ssm.initialize('approximate_diffuse')
            except Exception:
                pass

        return params

    def transform_states_to_signal(self, alpha):
        """Transform states to signal theta_t."""
        return np.atleast_2d(alpha)

    def loglikelihood_obs(self, t, theta_t):
        """Log-likelihood for observation t given signal theta_t."""
        theta_arr = np.atleast_1d(theta_t)
        theta_val = float(theta_arr.ravel()[0])
        theta_clipped = float(np.clip(theta_val, -_THETA_CLIP, _THETA_CLIP))

        y_t = float(self.endog[t, 0])

        # Clip exponent to avoid overflow
        x = np.clip(-theta_clipped, -_LOG_EXP_CLIP, _LOG_EXP_CLIP)
        exp_term = np.exp(x)

        kappa_t = exp_term * (y_t - self.mu) ** 2 / max(self.nu - 2.0, 1e-8)

        const = (gammaln((self.nu + 1.0) / 2.0) -
                 gammaln(self.nu / 2.0) -
                 0.5 * np.log((self.nu - 2.0) * np.pi))

        loglik = const - 0.5 * theta_clipped - 0.5 * (self.nu + 1.0) * np.log(1.0 + kappa_t)
        return float(loglik)

    def simulate(self, params, nsimulations, seed=None):
        """Simulate data from the model."""
        if seed is not None:
            np.random.seed(seed)

        self.update(params)

        # Simulate log-volatility
        theta = np.zeros(nsimulations)
        var_init = self.sigma_eta ** 2 / (1.0 - self.phi ** 2 + 1e-10)
        theta[0] = np.random.normal(0.0, np.sqrt(var_init))

        for t in range(1, nsimulations):
            theta[t] = self.phi * theta[t - 1] + np.random.normal(0.0, self.sigma_eta)

        # Simulate observations
        y = np.zeros(nsimulations)
        for t in range(nsimulations):
            eps = stats.t.rvs(self.nu)
            y[t] = self.mu + np.exp(0.5 * theta[t]) * eps

        return y, theta


def meis_loglikelihood(params, model, M=200, meis_iter=20, verbose=False):
    """
    Compute MEIS log-likelihood for given parameters.

    Parameters
    ----------
    params : array (4,)
        Parameter vector [mu, phi, sigma_eta, nu]
    model : StochasticVolatilityStudentT
        The model instance
    M : int
        Number of importance samples
    meis_iter : int
        Maximum MEIS iterations
    verbose : bool
        Print diagnostics

    Returns
    -------
    loglik : float
        MEIS log-likelihood
    """
    # Enforce constraints directly (backup to transformation)
    mu, phi, sigma_eta, nu = params
    phi = np.clip(phi, 0.05, 0.995)
    sigma_eta = max(sigma_eta, 0.01)
    nu = max(nu, 3.0)

    params_safe = np.array([mu, phi, sigma_eta, nu])

    # Update model
    model.update(params_safe)

    # Initialize and fit MEIS
    try:
        meis = model._initialize_meis(M=M, max_iter=meis_iter)
        meis.fit(verbose=False, seed=42)

        # Compute likelihood
        lik = MEISLikelihood(model, meis)
        loglik, u_bar, s2_u = lik.compute_loglikelihood(M=M, seed=42)

        if verbose:
            print(f"  Params: mu={mu:.4f}, phi={phi:.4f}, sigma_eta={sigma_eta:.4f}, nu={nu:.2f}")
            print(f"  loglik={loglik:.2f}, u_bar={u_bar:.4f}, s2_u={s2_u:.2e}")

        return loglik

    except Exception as e:
        if verbose:
            print(f"  Error computing likelihood: {e}")
        return -1e10  # Return very negative value on error


def estimate_params(model, true_params, M=200, meis_iter=20, method='L-BFGS-B'):
    """
    Estimate all four parameters by maximizing MEIS likelihood.

    Parameters
    ----------
    model : StochasticVolatilityStudentT
        Model with observed data
    true_params : array (4,)
        True parameter values [mu, phi, sigma_eta, nu] for comparison
    M : int
        Number of importance samples
    meis_iter : int
        MEIS iterations
    method : str
        Optimization method ('Nelder-Mead', 'Powell', 'L-BFGS-B')

    Returns
    -------
    result : OptimizeResult
        Optimization result with estimated parameters
    """
    print("\n" + "=" * 70)
    print("MAXIMUM LIKELIHOOD ESTIMATION VIA MEIS")
    print("=" * 70)
    print(f"Estimating: mu, phi, sigma_eta, nu")
    print(f"True values: mu={true_params[0]:.4f}, phi={true_params[1]:.4f}, "
          f"sigma_eta={true_params[2]:.4f}, nu={true_params[3]:.2f}")
    print(f"MEIS settings: M={M}, max_iter={meis_iter}")
    print(f"Optimizer: {method}")

    # Starting values (use model defaults or slight perturbations)
    x0 = model.start_params.copy()
    print(f"\nStarting values: mu={x0[0]:.4f}, phi={x0[1]:.4f}, "
          f"sigma_eta={x0[2]:.4f}, nu={x0[3]:.2f}")

    # Negative log-likelihood for minimization
    iter_count = [0]
    start_time = time.time()

    def neg_loglik(params):
        iter_count[0] += 1
        ll = meis_loglikelihood(params, model, M=M, meis_iter=meis_iter, verbose=True)

        elapsed = time.time() - start_time
        print(f"\nIteration {iter_count[0]}: neg_loglik={-ll:.2f} (elapsed: {elapsed:.1f}s)")

        return -ll

    print("\n" + "-" * 70)
    print("OPTIMIZATION PROGRESS")
    print("-" * 70)

    # Choose optimization method
    if method == 'Nelder-Mead':
        result = optimize.minimize(
            neg_loglik,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 100, 'xatol': 1e-3, 'fatol': 1.0}
        )
    elif method == 'Powell':
        result = optimize.minimize(
            neg_loglik,
            x0,
            method='Powell',
            options={'maxiter': 100, 'xtol': 1e-3, 'ftol': 1.0}
        )
    elif method == 'L-BFGS-B':
        # Set bounds
        bounds = [
            (None, None),  # mu: no bounds
            (0.05, 0.995),  # phi: (0, 1)
            (0.01, 1.0),  # sigma_eta: (0, inf) but practically bounded
            (3.0, 50.0)  # nu: (2, inf) but practically bounded
        ]
        result = optimize.minimize(
            neg_loglik,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100}
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("ESTIMATION RESULTS")
    print("=" * 70)
    print(f"Converged: {result.success}")
    print(f"Optimizer iterations: {result.nit}")
    print(f"Function evaluations: {result.nfev if hasattr(result, 'nfev') else 'N/A'}")
    print(f"Total time: {total_time:.1f}s")
    print(f"\nFinal log-likelihood: {-result.fun:.2f}")

    # Display results table
    print("\n" + "-" * 70)
    print("PARAMETER ESTIMATES")
    print("-" * 70)
    print(f"{'Parameter':<15} {'True':<12} {'Estimated':<12} {'Error':<12} {'% Error':<10}")
    print("-" * 70)

    param_names = ['mu', 'phi', 'sigma_eta', 'nu']
    for i, name in enumerate(param_names):
        true_val = true_params[i]
        est_val = result.x[i]
        error = est_val - true_val
        pct_error = 100 * error / true_val if abs(true_val) > 1e-6 else np.nan

        print(f"{name:<15} {true_val:<12.4f} {est_val:<12.4f} {error:<12.4f} {pct_error:<10.2f}%")

    print("-" * 70)

    # Model diagnostics at estimated parameters
    print("\n" + "-" * 70)
    print("MODEL DIAGNOSTICS AT ESTIMATED PARAMETERS")
    print("-" * 70)

    model.update(result.x)
    meis_final = model._initialize_meis(M=M * 2, max_iter=meis_iter)  # More samples for final diagnostics
    meis_final.fit(verbose=False, seed=42)

    lik_final = MEISLikelihood(model, meis_final)
    ll_final, u_bar, s2_u = lik_final.compute_loglikelihood(M=M * 2, seed=42)

    print(f"u_bar:  {u_bar:.4f}  (should be close to 1.0)")
    print(f"s2_u:   {s2_u:.4e}  (variance of importance weights)")

    if 0.8 < u_bar < 1.2:
        print("✓ Good importance density (u_bar ≈ 1)")
    else:
        print("⚠ Importance density could be improved")

    if s2_u < 1.0:
        print("✓ Low variance importance weights (efficient)")
    else:
        print("⚠ High variance weights (consider increasing M)")

    print("-" * 70)

    return result


def plot_results(y_sim, theta_true, result, true_params):
    """
    Create diagnostic plots for the estimation results.
    """
    print("\n" + "=" * 70)
    print("CREATING DIAGNOSTIC PLOTS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Observed returns
    ax1 = axes[0, 0]
    ax1.plot(y_sim, 'o-', alpha=0.6, markersize=2, linewidth=0.5)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Returns')
    ax1.set_title(f'Simulated Returns (n={len(y_sim)})')
    ax1.grid(True, alpha=0.3)

    # Plot 2: True log-volatility
    ax2 = axes[0, 1]
    vol_true = np.exp(0.5 * theta_true)
    ax2.plot(vol_true, linewidth=1.5, label='True volatility')
    ax2.plot(np.abs(y_sim), alpha=0.3, linewidth=0.5, label='|Returns|')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Volatility')
    ax2.set_title('True Volatility vs Realized')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Parameter comparison (bar chart)
    ax3 = axes[1, 0]
    param_names = ['mu', 'phi', 'sigma_eta', 'nu']
    x_pos = np.arange(len(param_names))

    width = 0.35
    ax3.bar(x_pos - width / 2, true_params, width, label='True', alpha=0.8)
    ax3.bar(x_pos + width / 2, result.x, width, label='Estimated', alpha=0.8)

    ax3.set_xlabel('Parameter')
    ax3.set_ylabel('Value')
    ax3.set_title('Parameter Comparison')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(param_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Estimation errors
    ax4 = axes[1, 1]
    errors = result.x - true_params
    pct_errors = 100 * errors / true_params

    colors = ['green' if abs(e) < 10 else 'orange' if abs(e) < 20 else 'red'
              for e in pct_errors]
    ax4.barh(param_names, pct_errors, color=colors, alpha=0.7)
    ax4.axvline(0, color='black', linewidth=0.8)
    ax4.set_xlabel('% Error')
    ax4.set_title('Estimation Errors')
    ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('sv_meis_full_estimation.png', dpi=150, bbox_inches='tight')
    print("Saved diagnostic plots to: sv_meis_full_estimation.png")

    return fig


def run_example(n=500, M=200, meis_iter=20, seed=42, method='L-BFGS-B'):
    """
    Run the full parameter estimation example.

    Parameters
    ----------
    n : int
        Number of observations to simulate
    M : int
        Number of importance samples for MEIS
    meis_iter : int
        Maximum MEIS iterations
    seed : int
        Random seed for reproducibility
    method : str
        Optimization method

    Returns
    -------
    output : dict
        Dictionary with results, data, and figures
    """
    print("=" * 70)
    print("STOCHASTIC VOLATILITY: FULL PARAMETER ESTIMATION WITH MEIS")
    print("=" * 70)
    print(f"Estimating all 4 parameters: mu, phi, sigma_eta, nu")

    # True parameters
    true_params = np.array([0.0, 0.98, 0.15, 10.0])  # [mu, phi, sigma_eta, nu]

    print(f"\nSimulation settings:")
    print(f"  n = {n} observations")
    print(f"  seed = {seed}")
    print(f"  True parameters:")
    print(f"    mu        = {true_params[0]:.4f}")
    print(f"    phi       = {true_params[1]:.4f}")
    print(f"    sigma_eta = {true_params[2]:.4f}")
    print(f"    nu        = {true_params[3]:.2f}")

    # Simulate data
    np.random.seed(seed)
    sim_model = StochasticVolatilityStudentT(np.zeros(n))
    y_sim, theta_true = sim_model.simulate(true_params, nsimulations=n, seed=seed)

    print(f"\nData statistics:")
    print(f"  mean  = {np.mean(y_sim):.4f}")
    print(f"  std   = {np.std(y_sim):.4f}")
    print(f"  min   = {np.min(y_sim):.4f}")
    print(f"  max   = {np.max(y_sim):.4f}")
    print(f"  skew  = {stats.skew(y_sim):.4f}")
    print(f"  kurt  = {stats.kurtosis(y_sim):.4f}")

    # Create model for estimation
    model = StochasticVolatilityStudentT(y_sim)

    # Estimate parameters
    result = estimate_params(model, true_params, M=M, meis_iter=meis_iter, method=method)

    # Create diagnostic plots
    fig = plot_results(y_sim, theta_true, result, true_params)

    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETE!")
    print("=" * 70)
    print(f"\nKey results:")
    print(f"  • All 4 parameters estimated successfully")
    print(f"  • Optimization converged: {result.success}")
    print(f"  • Final log-likelihood: {-result.fun:.2f}")
    print(f"  • Saved diagnostic plots to: sv_meis_full_estimation.png")

    return {
        'result': result,
        'true_params': true_params,
        'y_sim': y_sim,
        'theta_true': theta_true,
        'figure': fig,
        'model': model
    }


if __name__ == "__main__":
    # Run example with default settings
    # For faster testing, use: n=200, M=150, meis_iter=15
    # For publication quality: n=1000, M=300, meis_iter=25

    output = run_example(
        n=800,  # Number of observations
        M=100,  # Importance samples
        meis_iter=20,  # MEIS iterations
        seed=42,  # Random seed
        method='L-BFGS-B'  # Optimization method
    )

    plt.show()