## Reference values for statsmodels.tsa.vector_ar.local_proj.LocalProjections
##
## Independently replicates the LocalProjections regression construction
## (statsmodels/tsa/vector_ar/local_proj.py: _build_regressors / fit) using
## base R's lm() for OLS and sandwich::NeweyWest() for the HAC covariance,
## on the real macrodata series already used elsewhere in the docs
## (docs/source/vector_ar.rst) and shipped with statsmodels at
## statsmodels/datasets/macrodata/macrodata.csv.
##
## Python test is: statsmodels/tsa/vector_ar/tests/test_local_projection.py
## ::test_against_r_lpirfs_base_case and ::test_against_r_lpirfs_with_exog
##
## install.packages(c("sandwich"))
library(sandwich)
options(digits = 10)

macro <- read.csv("../../../../datasets/macrodata/macrodata.csv")

## Same transform as docs/source/vector_ar.rst: log-difference, drop NA row.
log_diff <- function(x) diff(log(x))

# ---------------------------------------------------------------------------
# Newey-West HAC sandwich covariance, hand-rolled to match
# statsmodels.tsa.vector_ar.local_proj._nw_cov exactly (Bartlett kernel, no
# prewhitening, no small-sample adjustment, divide by T twice).
# ---------------------------------------------------------------------------
nw_cov <- function(X, resid, nlags) {
  Tn <- nrow(X)
  scores <- X * resid
  S <- t(scores) %*% scores / Tn
  for (lag in 1:nlags) {
    w <- 1 - lag / (nlags + 1)
    gamma <- t(scores[(lag + 1):Tn, , drop = FALSE]) %*%
      scores[1:(Tn - lag), , drop = FALSE] / Tn
    S <- S + w * (gamma + t(gamma))
  }
  XtX_inv <- solve(t(X) %*% X / Tn)
  (XtX_inv %*% S %*% XtX_inv) / Tn
}

# ---------------------------------------------------------------------------
# LocalProjections regression, replicated row-for-row: for lags L and a
# fixed common sample used across all horizons 0..H (t_start = L,
# t_end = T - H, matching local_proj.py's fit()), regress
# Y[t+h, response] on [shock_t, lag1(all), ..., lagL(all), (exog_t),
# const] and report the shock coefficient +/- its Newey-West SE.
# ---------------------------------------------------------------------------
fit_lp <- function(Y, shock_col, lags, horizons, nw_lag, exog = NULL) {
  Tn <- nrow(Y)
  n <- ncol(Y)
  t_start <- lags + 1L         # R row index of the first usable "t"
  t_end <- Tn - horizons       # R row index of the last usable "t"
  idx <- t_start:t_end
  n_obs <- length(idx)

  X <- Y[idx, shock_col, drop = FALSE]
  for (lag in 1:lags) {
    X <- cbind(X, Y[idx - lag, , drop = FALSE])
  }
  if (!is.null(exog)) {
    X <- cbind(X, exog[idx, , drop = FALSE])
  }
  X <- cbind(X, const = rep(1, n_obs))
  X <- as.matrix(X)

  irfs <- matrix(NA_real_, nrow = horizons + 1, ncol = n)
  ses <- matrix(NA_real_, nrow = horizons + 1, ncol = n)
  for (h in 0:horizons) {
    Yh <- Y[idx + h, , drop = FALSE]
    for (i in 1:n) {
      fit <- lm.fit(X, Yh[, i])
      beta <- fit$coefficients
      resid <- fit$residuals
      V <- nw_cov(X, resid, nw_lag)
      irfs[h + 1, i] <- beta[1]
      ses[h + 1, i] <- sqrt(V[1, 1])
    }
  }
  list(irfs = irfs, ses = ses)
}

# ---------------------------------------------------------------------------
# Base case: realgdp shock -> (realgdp, realcons, realinv), lags=2,
# horizons=4, trend="c" (constant only), fixed nw_lags=4.
# ---------------------------------------------------------------------------
endog <- data.frame(
  realgdp = log_diff(macro$realgdp),
  realcons = log_diff(macro$realcons),
  realinv = log_diff(macro$realinv)
)
Y <- as.matrix(endog)

res_base <- fit_lp(Y, shock_col = 1, lags = 2, horizons = 4, nw_lag = 4)
cat("Base case irfs (rows=horizon 0..4, cols=realgdp,realcons,realinv):\n")
print(res_base$irfs)
cat("Base case stderr:\n")
print(res_base$ses)

# Cross-check the hand-rolled HAC formula against sandwich::NeweyWest,
# using a non-degenerate response (realcons at h=1, i.e. not the shock's
# own contemporaneous response, which fits exactly by construction).
idx1 <- 3:(nrow(Y) - 4)
X1 <- cbind(shock = Y[idx1, 1], Y[idx1 - 1, ], Y[idx1 - 2, ], const = 1)
fit1 <- lm(Y[idx1 + 1, 2] ~ X1 - 1)
V_sw <- NeweyWest(fit1, lag = 4, prewhite = FALSE, adjust = FALSE)
cat("sandwich::NeweyWest se[shock] realcons h=1:", sqrt(V_sw[1, 1]), "\n")
cat("hand-rolled nw_cov se[shock] realcons h=1: ", res_base$ses[2, 2], "\n")

# ---------------------------------------------------------------------------
# Exogenous-controls case: realgdp shock -> realgdp, lags=1, horizons=3,
# contemporaneous exog = log-diff(realgovt), fixed nw_lags=3.
# ---------------------------------------------------------------------------
endog2 <- data.frame(realgdp = log_diff(macro$realgdp))
exog2 <- data.frame(realgovt = log_diff(macro$realgovt))
Y2 <- as.matrix(endog2)
E2 <- as.matrix(exog2)

res_exog <- fit_lp(Y2, shock_col = 1, lags = 1, horizons = 3, nw_lag = 3,
                    exog = E2)
cat("\nExog case irfs (rows=horizon 0..3, col=realgdp):\n")
print(res_exog$irfs)
cat("Exog case stderr:\n")
print(res_exog$ses)
