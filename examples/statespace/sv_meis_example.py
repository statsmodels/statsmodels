"""
Stochastic Volatility Model with Student's t using MEIS

This variant modifies the example so that only sigma_eta is estimated while
the other parameters (mu, phi, nu) are fixed to their true values. This
makes it easy to (a) plot the MEIS-based log-likelihood as a function of
sigma_eta and (b) debug / inspect behavior.

Usage:
    python sv_meis_example.py        # runs full example (may be slow)
    python sv_meis_example.py --smoke  # quick smoke test (small n, small M)

Notes:
- This script depends on your local `meis` implementation exposing
  MEISMixin, MEISResults and MEISLikelihood (the MEIS importance + likelihood).
- We perform a grid evaluation of sigma_eta and pick the maximizing value.
  Grid evaluation is deterministic and simple to inspect; you can later
  substitute an optimizer that varies only sigma if desired.
"""
import argparse
import time

import numpy as np
from scipy import stats
from scipy.special import gammaln
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.meis import MEISMixin, MEISLikelihood
from statsmodels.tsa.statespace.mlemodel import MLEModel

# Numerical clip constants reused from meis module (tune if needed)
_THETA_CLIP = 30.0
_LOG_EXP_CLIP = 700.0


class StochasticVolatilityStudentT(MEISMixin, MLEModel):
    """
    SV model with Student's t observation density.

    y_t = mu + exp(0.5 * theta_t) * eps_t,  eps_t ~ t(nu)
    theta_t = sum_j alpha_{j,t}
    alpha_{j, t} = phi * alpha_{j, t - 1} + eta_t, eta_t ~ N(0, sigma_eta^2)
    """

    def __init__(self, endog, p=1, **kwargs):
        self.p = int(p)
        self._endog_orig = np.asarray(endog).ravel()
        endog2d = self._endog_orig.reshape(-1, 1)

        super().__init__(endog=endog2d, k_states=self.p, k_posdef=self.p, **kwargs)

        self._param_names = ['mu'] + [f'phi_{j+1}' for j in range(self.p)] + \
                            [f'sigma_eta_{j+1}' for j in range(self.p)] + ['nu']

        # default values (will be overwritten by update when fitting)
        self.mu = 0.0
        self.phi = 0.98 * np.ones(self.p)
        self.sigma_eta = 0.15 * np.ones(self.p)
        self.nu = 10.0

    @property
    def param_names(self):
        return self._param_names

    @property
    def start_params(self):
        mu = 0.0
        phi = 0.98 * np.ones(self.p)
        sigma_eta = 0.15 * np.ones(self.p)
        nu = 10.0
        return np.concatenate([[mu], phi, sigma_eta, [nu]])

    def transform_params(self, unconstrained):
        constrained = unconstrained.copy()
        for j in range(self.p):
            constrained[1 + j] = 0.5 + 0.5 / (1.0 + np.exp(-unconstrained[1 + j]))
        for j in range(self.p):
            constrained[1 + self.p + j] = np.exp(unconstrained[1 + self.p + j])
        constrained[-1] = 2.5 + np.exp(unconstrained[-1])
        return constrained

    def untransform_params(self, constrained):
        unconstrained = constrained.copy()
        for j in range(self.p):
            phi_scaled = (constrained[1 + j] - 0.5) / 0.5
            phi_scaled = np.clip(phi_scaled, 1e-8, 1 - 1e-8)
            unconstrained[1 + j] = -np.log(1.0 / phi_scaled - 1.0)
        for j in range(self.p):
            unconstrained[1 + self.p + j] = np.log(constrained[1 + self.p + j])
        unconstrained[-1] = np.log(constrained[-1] - 2.5)
        return unconstrained

    def update(self, params, **kwargs):
        # Keep the same interface as statsmodels MLEModel.update:
        params = super().update(params, **kwargs)

        self.mu = params[0]
        self.phi = params[1:1 + self.p]
        self.sigma_eta = params[1 + self.p:1 + 2 * self.p]
        self.nu = params[-1]

        # state-space matrices
        self.ssm['design'] = np.ones((1, self.k_states))
        self.ssm['transition'] = np.diag(self.phi)
        self.ssm['selection'] = np.eye(self.k_states)
        self.ssm['state_cov'] = np.diag(self.sigma_eta ** 2)

        # initialization: known stationary covariance approximation
        initial_state = np.zeros(self.k_states)
        initial_cov = np.diag(self.sigma_eta ** 2 / (1.0 - self.phi ** 2 + 1e-10))
        try:
            self.ssm.initialize_known(initial_state, initial_cov)
        except Exception:
            try:
                self.ssm.initialize('approximate_diffuse')
            except Exception:
                pass

        return params

    def transform_states_to_signal(self, alpha):
        alpha = np.atleast_2d(alpha)
        theta = np.sum(alpha, axis=0)
        return theta.reshape(1, -1)

    def loglikelihood_obs(self, t, theta_t):
        # Ensure scalar theta and clip to avoid overflow
        theta_arr = np.atleast_1d(theta_t)
        theta_val = float(theta_arr.ravel()[0])
        theta_clipped = float(np.clip(theta_val, -_THETA_CLIP, _THETA_CLIP))

        y_t = float(self.endog[t, 0])

        # Clip exponent argument to avoid overflow
        x = np.clip(-theta_clipped, -_LOG_EXP_CLIP, _LOG_EXP_CLIP)
        exp_term = np.exp(x)

        kappa_t = exp_term * (y_t - self.mu) ** 2 / max(self.nu - 2.0, 1e-8)

        const = (gammaln((self.nu + 1.0) / 2.0) -
                 gammaln(self.nu / 2.0) -
                 0.5 * np.log((self.nu - 2.0) * np.pi))

        loglik = const - 0.5 * theta_clipped - 0.5 * (self.nu + 1.0) * np.log(1.0 + kappa_t)
        return float(loglik)

    def simulate(self, params, nsimulations, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.update(params)

        alpha = np.zeros((self.k_states, nsimulations))
        for j in range(self.k_states):
            var_j = self.sigma_eta[j] ** 2 / (1.0 - self.phi[j] ** 2 + 1e-10)
            alpha[j, 0] = np.random.normal(0.0, np.sqrt(var_j))

        for t in range(1, nsimulations):
            for j in range(self.k_states):
                alpha[j, t] = self.phi[j] * alpha[j, t - 1] + np.random.normal(0.0, self.sigma_eta[j])

        theta = np.sum(alpha, axis=0)
        y = np.zeros(nsimulations)
        for t in range(nsimulations):
            eps = stats.t.rvs(self.nu)
            y[t] = self.mu + np.exp(0.5 * theta[t]) * eps

        return y, theta


def evaluate_sigma_grid(model, sigma_grid, M=200, meis_iter=20, seed=None, verbose=False):
    """
    Evaluate MEIS log-likelihood as a function of a single sigma_eta (scalar).
    Other parameters must already be fixed on the model (mu, phi, nu).
    Returns arrays (sigma_grid, loglik_vals, u_bar_vals, s2_u_vals).
    """
    loglik_vals = np.zeros_like(sigma_grid, dtype=float)
    u_bar_vals = np.zeros_like(sigma_grid, dtype=float)
    s2_u_vals = np.zeros_like(sigma_grid, dtype=float)

    for i, sigma in enumerate(sigma_grid):
        # Build full parameter vector expected by model.update:
        # [mu, phi_1, ..., sigma_eta_1, ..., nu]
        mu = model.mu
        phi = model.phi
        nu = model.nu
        # p may be >1 but here we assume p=1 for simplicity; extend as needed
        full_params = np.zeros(1 + model.p + model.p + 1)
        full_params[0] = mu
        full_params[1:1 + model.p] = phi
        full_params[1 + model.p:1 + 2 * model.p] = np.ones(model.p) * sigma
        full_params[-1] = nu

        # update model in-place
        model.update(full_params)

        # initialize MEIS importance density for current parameters
        imp = model._initialize_meis(M=M, max_iter=meis_iter)
        # fit importance density (iterative OLS inside)
        try:
            imp.fit(seed=seed, verbose=False)
        except Exception:
            # tolerate failures by attempting a fit without seed/verbose
            imp.fit()

        # compute log-likelihood estimate via MEISLikelihood
        lik = MEISLikelihood(model, imp)
        loglik, u_bar, s2_u = lik.compute_loglikelihood(M=M, seed=seed)

        loglik_vals[i] = loglik
        u_bar_vals[i] = u_bar
        s2_u_vals[i] = s2_u

    return loglik_vals, u_bar_vals, s2_u_vals


def run_example(n=1000, p=1, M=100, meis_iter=10, seed=42, quick=False):
    """
    Run SV-MEIS example. When quick=True, uses small sizes for fast testing.
    """
    if quick:
        n = min(n, 200)
        M = min(M, 50)
        meis_iter = min(meis_iter, 5)

    print("=" * 80)
    print("MEIS Stochastic Volatility Example (sigma_eta-only grid)")
    print("=" * 80)

    np.random.seed(seed)
    # true params for p=1 ordering: mu, phi, sigma_eta, nu
    true_params = [0.0, 0.98, 0.15, 10.0]

    # simulate data from the generative model using an auxiliary instance
    sim_model = StochasticVolatilityStudentT(np.zeros(n), p=p)
    y_sim, theta_true = sim_model.simulate(true_params, nsimulations=n, seed=seed)

    print(f" True parameters: μ={true_params[0]:.2f}, φ={true_params[1]:.3f}, "
          f"σ_η={true_params[2]:.3f}, ν={true_params[3]:.1f}")

    # Build model instance for estimation and fix mu, phi, nu to true values
    model = StochasticVolatilityStudentT(y_sim, p=p)

    # Set fixed params (deterministic) to true values
    mu_true = true_params[0]
    phi_true = np.array([true_params[1]] * p)
    nu_true = true_params[3]
    model.mu = mu_true
    model.phi = phi_true
    model.nu = nu_true
    # update ssm matrices to reflect fixed phi, mu, nu (sigma_eta not set yet)
    # initialize with the true sigma for sensible starting state cov
    model.sigma_eta = np.ones(p) * true_params[2]
    model.update(np.concatenate([[model.mu], model.phi, model.sigma_eta, [model.nu]]))

    # Grid for sigma_eta to evaluate
    if quick:
        sigma_grid = np.linspace(0.05, 0.4, 12)
    else:
        sigma_grid = np.linspace(0.02, 0.35, 20)

    print("Evaluating MEIS log-likelihood on sigma grid (this may take time)...")
    start = time.time()
    loglik_vals, u_bar_vals, s2_u_vals = evaluate_sigma_grid(
        model, sigma_grid, M=M, meis_iter=meis_iter, seed=seed, verbose=True
    )
    elapsed = time.time() - start
    print(f"Grid evaluation completed in {elapsed:.1f}s")

    # Find maximizing sigma
    best_idx = np.nanargmax(loglik_vals)
    sigma_hat = sigma_grid[best_idx]
    print(f"Grid maximum sigma_eta = {sigma_hat:.5f} (loglik={loglik_vals[best_idx]:.6f})")

    # Plot log-likelihood vs sigma
    try:
        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax[0].plot(sigma_grid, loglik_vals, '-o', label='MEIS loglik')
        ax[0].axvline(true_params[2], color='C1', linestyle='--', label='true sigma')
        ax[0].axvline(sigma_hat, color='C2', linestyle=':', label='grid argmax')
        ax[0].set_ylabel('MEIS log-likelihood')
        ax[0].legend()

        ax[1].plot(sigma_grid, u_bar_vals, '-o', label='u_bar')
        ax[1].plot(sigma_grid, s2_u_vals, '-x', label='s2_u')
        ax[1].set_xlabel('sigma_eta')
        ax[1].set_ylabel('Diagnostics')
        ax[1].legend()

        fig.tight_layout()
        plt.savefig('sv_meis_sigma_grid.png', dpi=150, bbox_inches='tight')
        print("Saved sigma-grid plot to sv_meis_sigma_grid.png")
    except Exception as e:
        print(f"Plotting failed: {e}")

    return {
        "sigma_grid": sigma_grid,
        "loglik_vals": loglik_vals,
        "u_bar_vals": u_bar_vals,
        "s2_u_vals": s2_u_vals,
        "sigma_hat": sigma_hat,
        "theta_true": theta_true,
        "y_sim": y_sim,
    }


def run_smoke_test():
    print("\nRunning smoke test (quick run)...")
    out = run_example(n=120, p=1, M=40, meis_iter=4, seed=123, quick=True)
    print("\nSmoke test complete.")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test")
    args = parser.parse_args()
    if args.smoke:
        run_smoke_test()
    else:
        run_example(n=800, p=1, M=500, meis_iter=20, seed=42, quick=False)
