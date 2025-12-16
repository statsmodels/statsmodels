"""
Stochastic Volatility Model with Student's t using MEIS

Complete example implementation demonstrating MEIS integration with statsmodels.
Follows Koopman, Lit, Nguyen (2019) Section 2.2 and 6.

Model:
    y_t = μ + exp(0.5 * θ_t) * ε_t,  ε_t ~ t(ν)
    θ_t = Σ_{j=1}^p α_{j,t}
    α_{j,t+1} = φ_j * α_{j,t} + η_{j,t},  η_{j,t} ~ N(0, σ²_{η,j})
"""

import numpy as np
from scipy import stats
from scipy.special import gammaln
import matplotlib.pyplot as plt
import pandas as pd

# Import MEIS components
from meis import MEISMixin, MEISResults
from statsmodels.tsa.statespace.mlemodel import MLEModel

class StochasticVolatilityStudentT(MEISMixin, MLEModel):
    """
    Stochastic Volatility model with Student's t distribution.

    Follows Section 2.2 of Koopman, Lit, Nguyen (2019).

    Parameters
    ----------
    endog : array_like
        Observed time series (returns)
    p : int, optional
        Number of AR components in log-volatility. Default is 1.

    Examples
    --------
    >>> returns = np.random.randn(1000) * np.exp(np.random.randn(1000) * 0.5)
    >>> model = StochasticVolatilityStudentT(returns, p=1)
    >>> results = model.fit_meis(M=100, meis_iter=10)
    >>> theta_smooth, _, _ = results.smooth_signal()
    """

    def __init__(self, endog, p=1, **kwargs):
        # Configuration
        self.p = p
        self._endog_orig = np.asarray(endog).ravel()

        # Initialize state space
        super().__init__(
            endog=self._endog_orig,
            k_states=p,
            k_posdef=p,
            **kwargs
        )

        # Parameter names
        self._param_names = ['mu']
        for j in range(p):
            self._param_names.append(f'phi_{j + 1}')
        for j in range(p):
            self._param_names.append(f'sigma_eta_{j + 1}')
        self._param_names.append('nu')

        # Storage
        self.mu = 0.0
        self.phi = np.zeros(p)
        self.sigma_eta = np.zeros(p)
        self.nu = 10.0

    @property
    def param_names(self):
        """Parameter names for display."""
        return self._param_names

    @property
    def start_params(self):
        """Starting values (Table 1 style)."""
        mu = 0.0
        phi = 0.98 * np.ones(self.p)
        if self.p > 1:
            phi[1] = 0.90
        if self.p > 2:
            phi[2] = 0.80
        sigma_eta = 0.15 * np.ones(self.p)
        nu = 10.0

        return np.concatenate([[mu], phi, sigma_eta, [nu]])

    def transform_params(self, unconstrained):
        """
        Transform unconstrained to constrained parameters.

        Constraints:
        - 0.5 < φ_j < 1
        - σ_η,j > 0
        - ν > 2
        """
        constrained = unconstrained.copy()

        # φ_j: logistic to (0.5, 1)
        for j in range(1, 1 + self.p):
            constrained[j] = 0.5 + 0.5 / (1 + np.exp(-unconstrained[j]))

        # σ_η,j: exponential
        for j in range(1 + self.p, 1 + 2 * self.p):
            constrained[j] = np.exp(unconstrained[j])

        # ν: exponential + 2
        constrained[-1] = 2.5 + np.exp(unconstrained[-1])

        return constrained

    def untransform_params(self, constrained):
        """Transform constrained to unconstrained."""
        unconstrained = constrained.copy()

        # Reverse φ_j
        for j in range(1, 1 + self.p):
            phi_scaled = (constrained[j] - 0.5) / 0.5
            unconstrained[j] = -np.log(1.0 / (phi_scaled + 1e-10) - 1.0)

        # Reverse σ_η,j
        for j in range(1 + self.p, 1 + 2 * self.p):
            unconstrained[j] = np.log(constrained[j])

        # Reverse ν
        unconstrained[-1] = np.log(constrained[-1] - 2.5)

        return unconstrained

    def update(self, params, **kwargs):
        """
        Update model with new parameters.

        Parameters: [μ, φ_1, ..., φ_p, σ_η,1, ..., σ_η,p, ν]
        """
        params = super().update(params, **kwargs)

        # Extract
        self.mu = params[0]
        self.phi = params[1:1 + self.p]
        self.sigma_eta = params[1 + self.p:1 + 2 * self.p]
        self.nu = params[-1]

        # Build state space matrices
        # Design: Z = [1, 1, ..., 1] to sum AR components
        self.ssm['design'] = np.ones((1, self.k_states, 1))

        # Transition: diagonal with φ_j
        self.ssm['transition'] = np.diag(self.phi)[:, :, np.newaxis]

        # Selection: identity
        self.ssm['selection'] = np.eye(self.k_states)[:, :, np.newaxis]

        # State covariance: diagonal with σ²_η,j
        self.ssm['state_cov'] = np.diag(self.sigma_eta ** 2)[:, :, np.newaxis]

        # Initialize at stationary distribution
        initial_state = np.zeros(self.k_states)
        initial_cov = np.diag(self.sigma_eta ** 2 / (1 - self.phi ** 2 + 1e-10))

        self.ssm.initialize_known(initial_state, initial_cov)

        return params

    def transform_states_to_signal(self, alpha):
        """
        Transform states α to signal θ.

        For SV: θ_t = Σ α_{j,t} (sum of AR components)

        Parameters
        ----------
        alpha : ndarray (k_states, nobs)

        Returns
        -------
        theta : ndarray (1, nobs) or (nobs,)
        """
        theta = np.sum(alpha, axis=0)
        return theta.reshape(1, -1) if theta.ndim == 1 else theta

    def loglikelihood_obs(self, t, theta_t):
        """
        Observation log-likelihood for Student's t (Eq. 21).

        Parameters
        ----------
        t : int
            Time index
        theta_t : ndarray
            Log-volatility (scalar or (1,) array)

        Returns
        -------
        loglik : float
        """
        y_t = self.endog[t, 0]

        # Handle scalar or array input
        if np.isscalar(theta_t):
            theta_val = theta_t
        else:
            theta_val = theta_t[0] if len(theta_t) > 0 else theta_t

        # κ_t = exp(-θ_t) * (y_t - μ)² / (ν - 2)
        kappa_t = np.exp(-theta_val) * (y_t - self.mu) ** 2 / (self.nu - 2)

        # Constant term
        const = (gammaln((self.nu + 1) / 2) -
                 gammaln(self.nu / 2) -
                 0.5 * np.log((self.nu - 2) * np.pi))

        # Log-likelihood (Eq. 21)
        loglik = const - 0.5 * theta_val - 0.5 * (self.nu + 1) * np.log(1 + kappa_t)

        return loglik

    def simulate(self, params, nsimulations, seed=None):
        """
        Simulate data from SV model.

        Parameters
        ----------
        params : array_like
            Model parameters
        nsimulations : int
            Number of observations
        seed : int, optional
            Random seed

        Returns
        -------
        y : ndarray
            Simulated returns
        theta : ndarray
            True log-volatility
        """
        if seed is not None:
            np.random.seed(seed)

        # Update parameters
        self.update(params)

        # Simulate states
        alpha = np.zeros((self.k_states, nsimulations))

        # Initialize
        for j in range(self.k_states):
            var_j = self.sigma_eta[j] ** 2 / (1 - self.phi[j] ** 2 + 1e-10)
            alpha[j, 0] = np.random.normal(0, np.sqrt(var_j))

        # Forward simulation
        for t in range(1, nsimulations):
            for j in range(self.k_states):
                alpha[j, t] = (self.phi[j] * alpha[j, t - 1] +
                               np.random.normal(0, self.sigma_eta[j]))

        # Compute log-volatility
        theta = np.sum(alpha, axis=0)

        # Simulate observations
        y = np.zeros(nsimulations)
        for t in range(nsimulations):
            epsilon_t = stats.t.rvs(self.nu)
            y[t] = self.mu + np.exp(0.5 * theta[t]) * epsilon_t

        return y, theta


# ============================================================================
# Example Usage and Demonstration
# ============================================================================

def run_example(n=1000, p=1, M=100, meis_iter=10, maxiter=30, seed=42):
    """
    Run complete SV-MEIS example.

    Parameters
    ----------
    n : int
        Number of observations
    p : int
        Number of volatility components
    M : int
        Number of importance samples
    meis_iter : int
        MEIS iterations per likelihood evaluation
    maxiter : int
        Maximum optimization iterations
    seed : int
        Random seed
    """
    print("=" * 80)
    print("MEIS Stochastic Volatility Example")
    print("Following Koopman, Lit, Nguyen (2019)")
    print("=" * 80)

    # 1. Simulate data
    print(f"\n1. Simulating {n} observations...")
    np.random.seed(seed)

    true_params = [0.0, 0.98, 0.15, 10.0]  # [μ, φ, σ_η, ν]
    model_sim = StochasticVolatilityStudentT(np.zeros(n), p=p)
    y_sim, theta_true = model_sim.simulate(true_params, nsimulations=n, seed=seed)

    print(f"   True parameters: μ={true_params[0]:.2f}, φ={true_params[1]:.2f}, "
          f"σ_η={true_params[2]:.2f}, ν={true_params[3]:.1f}")

    # 2. Fit model
    print(f"\n2. Fitting SV model with MEIS (M={M}, meis_iter={meis_iter})...")
    model = StochasticVolatilityStudentT(y_sim, p=p)

    results = model.fit_meis(
        method='powell',
        maxiter=maxiter,
        M=M,
        meis_iter=meis_iter,
        disp=True
    )

    # 3. Results
    print("\n3. Estimation Results:")
    print("-" * 80)
    print(f"{'Parameter':<15} {'True':<12} {'Estimated':<12} {'Diff':<12}")
    print("-" * 80)
    print(f"{'μ':<15} {true_params[0]:<12.4f} {results.params[0]:<12.4f} "
          f"{abs(results.params[0] - true_params[0]):<12.4f}")
    print(f"{'φ₁':<15} {true_params[1]:<12.4f} {results.params[1]:<12.4f} "
          f"{abs(results.params[1] - true_params[1]):<12.4f}")
    print(f"{'σ_η,1':<15} {true_params[2]:<12.4f} {results.params[2]:<12.4f} "
          f"{abs(results.params[2] - true_params[2]):<12.4f}")
    print(f"{'ν':<15} {true_params[3]:<12.4f} {results.params[3]:<12.4f} "
          f"{abs(results.params[3] - true_params[3]):<12.4f}")
    print("-" * 80)
    print(f"Log-likelihood: {results.llf:.4f}")

    # 4. Extract volatility
    print("\n4. Extracting smoothed log-volatility...")
    theta_smooth, theta_draws, weights = results.smooth_signal(M=M)
    theta_smooth = theta_smooth[0, :]  # Extract from (1, nobs) to (nobs,)

    # Compute RMSE
    rmse_theta = np.sqrt(np.mean((theta_smooth - theta_true) ** 2))
    corr_theta = np.corrcoef(theta_smooth, theta_true)[0, 1]

    print(f"   Log-volatility RMSE: {rmse_theta:.4f}")
    print(f"   Log-volatility Correlation: {corr_theta:.4f}")

    # 5. Plot results
    print("\n5. Generating plots...")
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

    # Returns
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(y_sim, linewidth=0.5, alpha=0.7, color='black')
    ax1.set_title('Simulated Returns', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Return')
    ax1.grid(True, alpha=0.3)

    # Log-volatility
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(theta_true, label='True θ_t', linewidth=1.5, alpha=0.8, color='blue')
    ax2.plot(theta_smooth, label='Estimated θ_t (MEIS)',
             linewidth=1.5, alpha=0.8, color='red', linestyle='--')
    ax2.set_title('Log-Volatility Extraction', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Log-Volatility')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Volatility
    ax3 = fig.add_subplot(gs[2, :])
    vol_true = np.exp(0.5 * theta_true)
    vol_est = np.exp(0.5 * theta_smooth)
    ax3.plot(vol_true, label='True Volatility', linewidth=1.5, alpha=0.8, color='blue')
    ax3.plot(vol_est, label='Estimated Volatility (MEIS)',
             linewidth=1.5, alpha=0.8, color='red', linestyle='--')
    ax3.set_title('Volatility (Standard Deviation)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Volatility')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    # Standardized residuals
    ax4 = fig.add_subplot(gs[3, 0])
    residuals = y_sim / vol_est
    ax4.plot(residuals, linewidth=0.5, alpha=0.7, color='black')
    ax4.axhline(y=2, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax4.axhline(y=-2, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    ax4.set_title('Standardized Residuals', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Standardized Return')
    ax4.set_xlabel('Time')
    ax4.grid(True, alpha=0.3)

    # Residual histogram
    ax5 = fig.add_subplot(gs[3, 1])
    ax5.hist(residuals, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    ax5.plot(x_range, stats.norm.pdf(x_range, 0, 1), 'r-', linewidth=2,
             label='N(0,1)', alpha=0.8)
    ax5.plot(x_range, stats.t.pdf(x_range, df=results.params[-1]), 'g--',
             linewidth=2, label=f't(ν={results.params[-1]:.1f})', alpha=0.8)
    ax5.set_title('Residual Distribution', fontsize=13, fontweight='bold')
    ax5.set_xlabel('Standardized Residual')
    ax5.set_ylabel('Density')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)

    plt.savefig('sv_meis_complete.png', dpi=150, bbox_inches='tight')
    print("   Saved plot to 'sv_meis_complete.png'")

    # 6. Summary statistics
    print("\n6. Model Diagnostics:")
    print("-" * 80)
    print(f"Mean absolute residual:     {np.mean(np.abs(residuals)):.4f}")
    print(f"Std of residuals:           {np.std(residuals):.4f}")
    print(f"Skewness of residuals:      {stats.skew(residuals):.4f}")
    print(f"Kurtosis of residuals:      {stats.kurtosis(residuals):.4f}")
    print(f"Number of |resid| > 2:      {np.sum(np.abs(residuals) > 2)}")
    print("-" * 80)

    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)

    return results, theta_smooth, theta_true


if __name__ == "__main__":
    # Run example
    results, theta_smooth, theta_true = run_example(
        n=1000,
        p=1,
        M=100,
        meis_iter=10,
        maxiter=30,
        seed=42
    )

    print("\n" + "=" * 80)
    print("Key Features Demonstrated:")
    print("=" * 80)
    print("  ✓ MEISMixin integration with MLEModel")
    print("  ✓ fit_meis() for maximum likelihood estimation")
    print("  ✓ smooth_signal() for volatility extraction")
    print("  ✓ Student's t observation density (Eq. 21)")
    print("  ✓ Bias-corrected likelihood (Eq. 17-18)")
    print("  ✓ Parameter transformation and constraints")
    print("  ✓ Complete visualization and diagnostics")
    print("\nFor multivariate extensions:")
    print("  - Vector signals: return theta.shape = (q, nobs) in transform_states_to_signal()")
    print("  - Multiple observations: k_endog > 1 in model initialization")
    print("  - Factor models: theta_t includes multiple latent factors")