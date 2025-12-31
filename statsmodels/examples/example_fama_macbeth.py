"""
This example demonstrates the Fama-MacBeth two-stage regression procedure
for estimating factor risk premiums in asset pricing models.

Author: Soham Mukherjee (@gaixen)
"""
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.fama_macbeth import (
    FamaMacBethResults,
    FamaMacBeth
)

print(__doc__)

np.random.seed(42)

logging.info("Simulation of panel data")
n_stocks = 25  # Number of stocks/assets
n_months = 120  # Number of time periods (10 years of monthly data)
n_factors = 3  # Number of factors (e.g., Market, SMB, HML)

print(f"\nPanel dimensions:")
print(f"  - Stocks: {n_stocks}")
print(f"  - Time periods: {n_months}")
print(f"  - Factors: {n_factors}")

# Create panel structure
stocks = np.repeat(np.arange(n_stocks), n_months)
months = np.tile(np.arange(n_months), n_stocks)

# Generate factor returns (same for all stocks in each period)
# Simulate market factor, size factor (SMB), and value factor (HML)
np.random.seed(42)
factor_returns = np.zeros((n_months, n_factors))
factor_returns[:, 0] = np.random.randn(n_months) * 0.045 + 0.008  # Market: mean=0.8%, std=4.5%
factor_returns[:, 1] = np.random.randn(n_months) * 0.032  # SMB: mean=0%, std=3.2%
factor_returns[:, 2] = np.random.randn(n_months) * 0.034  # HML: mean=0%, std=3.4%

# Generate stock-specific factor loadings (betas)
true_betas = np.zeros((n_stocks, n_factors))
true_betas[:, 0] = np.random.uniform(0.7, 1.3, n_stocks)  # Market betas
true_betas[:, 1] = np.random.uniform(-0.5, 0.5, n_stocks)  # SMB betas
true_betas[:, 2] = np.random.uniform(-0.5, 0.5, n_stocks)  # HML betas

print(f"\nTrue beta ranges:")
print(f"  - Market: [{true_betas[:, 0].min():.2f}, {true_betas[:, 0].max():.2f}]")
print(f"  - SMB: [{true_betas[:, 1].min():.2f}, {true_betas[:, 1].max():.2f}]")
print(f"  - HML: [{true_betas[:, 2].min():.2f}, {true_betas[:, 2].max():.2f}]")

# Generate stock returns
stock_returns = np.zeros(n_stocks * n_months)
factors = np.zeros((n_stocks * n_months, n_factors))

for i in range(n_stocks):
    idx = stocks == i
    # Returns = beta * factors + idiosyncratic error
    stock_returns[idx] = (
        factor_returns @ true_betas[i] +
        np.random.randn(n_months) * 0.025  # Idiosyncratic volatility
    )
    factors[idx] = factor_returns

# Add constant term
exog = sm.add_constant(factors)

# ============================================================================
# 2. Estimate Fama-MacBeth Model
# ============================================================================
print("\n" + "="*70)
print("FAMA-MACBETH ESTIMATION")
print("="*70)

# Initialize model
model = FamaMacBeth(
    endog=stock_returns,
    exog=exog,
    entity=stocks,
    time=months,
)

# Estimate with robust standard errors
print("\nEstimating with robust standard errors...")
results_robust = model.fit(cov_type="robust")

print("\n" + "-"*70)
print("RESULTS: Robust Standard Errors")
print("-"*70)
print(results_robust.summary())

# Estimate with HAC standard errors
print("\n" + "="*70)
print("\nEstimating with HAC (Newey-West) standard errors...")
results_hac = model.fit(cov_type="HAC", bandwidth=6)

print("\n" + "-"*70)
print("RESULTS: HAC Standard Errors")
print("-"*70)
print(results_hac.summary())

# ============================================================================
# 3. Compare Standard Errors
# ============================================================================
print("\n" + "="*70)
print("COMPARISON OF STANDARD ERRORS")
print("="*70)

comparison = pd.DataFrame({
    "Parameter": ["Constant", "Factor 1 (Mkt)", "Factor 2 (SMB)", "Factor 3 (HML)"],
    "Estimate": results_robust.params,
    "SE (Robust)": results_robust.bse,
    "SE (HAC)": results_hac.bse,
    "Difference": results_hac.bse - results_robust.bse,
    "Rel. Diff. (%)": 100 * (results_hac.bse - results_robust.bse) / results_robust.bse,
})

print("\n", comparison.to_string(index=False))

# ============================================================================
# 4. Examine Time-Series of Risk Premia
# ============================================================================
print("\n" + "="*70)
print("TIME-SERIES PROPERTIES OF RISK PREMIA")
print("="*70)

lambdas_df = pd.DataFrame(
    results_robust.lambdas,
    columns=["Constant", "Mkt", "SMB", "HML"],
)

print("\nDescriptive Statistics:")
print(lambdas_df.describe())

print("\nAutocorrelations (first 3 lags):")
for col in lambdas_df.columns:
    acf_values = [lambdas_df[col].autocorr(lag=i) for i in range(1, 4)]
    print(f"  {col}: {acf_values[0]:.3f}, {acf_values[1]:.3f}, {acf_values[2]:.3f}")

# ============================================================================
# 5. Visualizations
# ============================================================================
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Fama-MacBeth Regression Results", fontsize=14, fontweight="bold")

# Plot 1: Time-series of risk premia
ax = axes[0, 0]
ax.plot(lambdas_df.index, lambdas_df["Mkt"], label="Market", alpha=0.7)
ax.axhline(results_robust.params[1], color="C0", linestyle="--", label="Average")
ax.set_xlabel("Time Period")
ax.set_ylabel("Risk Premium")
ax.set_title("Market Risk Premium Over Time")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Distribution of risk premia
ax = axes[0, 1]
ax.hist(lambdas_df["Mkt"], bins=20, alpha=0.7, edgecolor="black")
ax.axvline(results_robust.params[1], color="red", linestyle="--", 
           linewidth=2, label=f"Mean: {results_robust.params[1]:.4f}")
ax.set_xlabel("Risk Premium")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Market Risk Premium")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Comparison of factor risk premia
ax = axes[1, 0]
factor_names = ["Mkt", "SMB", "HML"]
estimates = results_robust.params[1:]  # Exclude constant
std_errors = results_robust.bse[1:]
x_pos = np.arange(len(factor_names))
ax.bar(x_pos, estimates, yerr=1.96*std_errors, alpha=0.7, 
       capsize=5, edgecolor="black")
ax.set_xticks(x_pos)
ax.set_xticklabels(factor_names)
ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
ax.set_ylabel("Risk Premium")
ax.set_title("Factor Risk Premia with 95% CI")
ax.grid(True, alpha=0.3, axis="y")

# Plot 4: Robust vs HAC standard errors
ax = axes[1, 1]
x_pos = np.arange(len(results_robust.params))
width = 0.35
ax.bar(x_pos - width/2, results_robust.bse, width, label="Robust", alpha=0.7)
ax.bar(x_pos + width/2, results_hac.bse, width, label="HAC", alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(["Const", "Mkt", "SMB", "HML"])
ax.set_ylabel("Standard Error")
ax.set_title("Comparison: Robust vs HAC Standard Errors")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("fama_macbeth_results.png", dpi=300, bbox_inches="tight")
print("\nFigure saved as 'fama_macbeth_results.png'")

# ============================================================================
# 6. Inference
# ============================================================================
print("\n" + "="*70)
print("STATISTICAL INFERENCE")
print("="*70)

print("\nHypothesis Tests (two-sided):")
for i, name in enumerate(["Constant", "Market", "SMB", "HML"]):
    t_stat = results_robust.tvalues[i]
    p_value = results_robust.pvalues[i]
    significant = "***" if p_value < 0.01 else ("**" if p_value < 0.05 else ("*" if p_value < 0.10 else ""))
    print(f"  {name:12s}: t = {t_stat:7.3f}, p-value = {p_value:.4f} {significant}")

print("\n  Significance levels: *** p<0.01, ** p<0.05, * p<0.10")

# 95% Confidence intervals
ci = results_robust.conf_int(alpha=0.05)
print("\n95% Confidence Intervals:")
for i, name in enumerate(["Constant", "Market", "SMB", "HML"]):
    print(f"  {name:12s}: [{ci[i, 0]:7.4f}, {ci[i, 1]:7.4f}]")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
