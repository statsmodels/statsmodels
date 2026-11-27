"""
Fama-MacBeth two-stage regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.fama_macbeth import FamaMacBeth

print(f"doc: {__doc__}")
np.random.seed(42)

N_STOCKS = 25
N_MONTHS = 120
N_FACTORS = 3

stocks = np.repeat(np.arange(N_STOCKS), N_MONTHS)
months = np.tile(np.arange(N_MONTHS), N_STOCKS)

factor_returns = np.zeros((N_MONTHS, N_FACTORS))
factor_returns[:, 0] = np.random.randn(N_MONTHS) * 0.045 + 0.008
factor_returns[:, 1] = np.random.randn(N_MONTHS) * 0.032
factor_returns[:, 2] = np.random.randn(N_MONTHS) * 0.034

true_betas = np.zeros((N_STOCKS, N_FACTORS))
true_betas[:, 0] = np.random.uniform(0.7, 1.3, N_STOCKS)
true_betas[:, 1] = np.random.uniform(-0.5, 0.5, N_STOCKS)
true_betas[:, 2] = np.random.uniform(-0.5, 0.5, N_STOCKS)
stock_returns = np.zeros(N_STOCKS * N_MONTHS)
factors = np.zeros((N_STOCKS * N_MONTHS, N_FACTORS))

for i in range(N_STOCKS):
    idx = stocks == i
    stock_returns[idx] = (
        factor_returns @ true_betas[i] + np.random.randn(N_MONTHS) * 0.025
    )
    factors[idx] = factor_returns

exog = sm.add_constant(factors)
model = FamaMacBeth(stock_returns, exog, entity=stocks, time=months)

results_robust = model.fit(cov_type="robust")
print(results_robust.summary())

results_hac = model.fit(cov_type="HAC", bandwidth=6)
print("\n")
print(results_hac.summary())

comparison = pd.DataFrame(
    {
        "Parameter": ["Constant", "Mkt", "SMB", "HML"],
        "Estimate": results_robust.params,
        "SE_Robust": results_robust.bse,
        "SE_HAC": results_hac.bse,
    }
)
print("\n")
print(comparison)

lambdas_df = pd.DataFrame(
    results_robust.lambdas,
    columns=["Constant", "Mkt", "SMB", "HML"],
)
print("\n")
print(lambdas_df.describe())

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Fama-MacBeth Regression Results", fontsize=14, fontweight="bold")
ax = axes[0, 0]
ax.plot(lambdas_df.index, lambdas_df["Mkt"], label="Market", alpha=0.7)
ax.axhline(results_robust.params[1], color="C0", linestyle="--", label="Average")
ax.set_xlabel("Time Period")
ax.set_ylabel("Risk Premium")
ax.set_title("Market Risk Premium Over Time")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.hist(lambdas_df["Mkt"], bins=20, alpha=0.7, edgecolor="black")
ax.axvline(results_robust.params[1], color="red", linestyle="--", linewidth=2)
ax.set_xlabel("Risk Premium")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Market Risk Premium")
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
factor_names = ["Mkt", "SMB", "HML"]
estimates = results_robust.params[1:]  # Exclude constant
std_errors = results_robust.bse[1:]
x_pos = np.arange(len(factor_names))
ax.bar(
    x_pos, estimates, yerr=1.96 * std_errors, alpha=0.7, capsize=5, edgecolor="black"
)
ax.set_xticks(x_pos)
ax.set_xticklabels(factor_names)
ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
ax.set_ylabel("Risk Premium")
ax.set_title("Factor Risk Premia with 95% CI")
ax.grid(True, alpha=0.3, axis="y")

ax = axes[1, 1]
x_pos = np.arange(len(results_robust.params))
width = 0.35
ax.bar(x_pos - width / 2, results_robust.bse, width, label="Robust", alpha=0.7)
ax.bar(x_pos + width / 2, results_hac.bse, width, label="HAC", alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(["Const", "Mkt", "SMB", "HML"])
ax.set_ylabel("Standard Error")
ax.set_title("Comparison: Robust vs HAC Standard Errors")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("fama_macbeth_results.png", dpi=300, bbox_inches="tight")
print("\nFigure saved as 'fama_macbeth_results.png'")

ci = results_robust.conf_int(alpha=0.05)
print("\n95% Confidence Intervals:")
for i, name in enumerate(["Constant", "Market", "SMB", "HML"]):
    print(f"{name:12s}: [{ci[i, 0]:.4f}, {ci[i, 1]:.4f}]")
