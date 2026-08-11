library(KFAS)
library(plyr)
options(digits=10)

# should run this from the statsmodels/statsmodels directory
setwd('~/projects/statsmodels-0.9/statsmodels/')
dta <- read.csv('datasets/macrodata/macrodata.csv')

cbind.fill <- function(...){
  nm <- list(...) 
  nm <- lapply(nm, as.matrix)
  n <- max(sapply(nm, nrow)) 
  do.call(cbind, lapply(nm, function (x) 
    rbind(x, matrix(, n-nrow(x), ncol(x))))) 
}

output <- function(kf) {
  # Dimensions
  n = nrow(kf$model$y)
  p = ncol(kf$model$y)
  m = nrow(kf$model$T)
  r = nrow(kf$model$Q)

  # Construct the output dataframe
  # r = scaled_smoothed_estimator
  # N = scaled_smoothed_estimator_cov
  # m = forecasts
  # v = forecasts_error
  # F = forecasts_error_cov
  # a = predicted_state
  # P = predicted_state_cov
  # mu = filtered_forecasts
  # alphahat = smoothed_state
  # V = smoothed_state_cov
  # muhat = smoothed_forecasts
  # etahat = smoothed_state_disturbance
  # V_eta = smoothed_state_disturbance_cov
  # epshat = smoothed_measurement_disturbance
  # V_eps = smoothed_measurement_disturbance_cov
  out <- as.data.frame(with(kf, cbind.fill(
    t(r), t(r0), t(r1),
    apply(N, 3, sum), apply(N0, 3, sum), apply(N1, 3, sum), apply(N2, 3, sum),
    m, v, t(F), t(Finf),
    a, apply(P, 3, sum), apply(Pinf, 3, sum),
    att, apply(Ptt, 3, sum),
    alphahat, apply(V, 3, sum),
    muhat, apply(V_mu, 3, sum),
    etahat, apply(V_eta, 3, sum),
    epshat, t(V_eps),
    logLik
  )))
  names(out) <- c(
    paste('r', 1:m, sep='_'), paste('r0', 1:m, sep='_'), paste('r1', 1:m, sep='_'),
    'sumN', 'sumN0', 'sumN1', 'sumN2',
    paste('m', 1:p, sep='_'), paste('v', 1:p, sep='_'), paste('F', 1:p, sep='_'), paste('Finf', 1:p, sep='_'),
    paste('a', 1:m, sep='_'), 'sumP', 'sumPinf',
    paste('att', 1:m, sep='_'), 'sumPtt',
    paste('alphahat', 1:m, sep='_'), 'sumV',
    paste('muhat', 1:p, sep='_'), 'sumVmu',
    paste('etahat', 1:r, sep='_'), 'sumVeta',
    paste('epshat', 1:p, sep='_'), paste('Veps', 1:p, sep='_'),
    'llf'
  )
  return(out)
}

llf <- function(kf) {
  # Dimensions
  n = nrow(kf$model$y)
  p = ncol(kf$model$y)
  m = nrow(kf$model$T)
  r = nrow(kf$model$Q)
  d = kf$d
  F = kf$F
  Finf = kf$Finf
  v = kf$v
  
  -((n * p / 2) * log(pi * 2) +
      0.5 * sum(log(Finf[1:d])) +
      0.5 * sum(log(F[(d+1):10]) + (v^2 / t(F))[(d+1):10]))
}

# Local level
y1 <- 10.2394
sigma2_y <- 1.993
sigma2_mu <- 8.253
obs <- c(c(y1), rep(1, 9))
mod <- SSModel(obs ~ -1 + SSMcustom(Z=diag(1), T=diag(1), R=diag(1), Q=diag(1) * sigma2_mu, P1=diag(1) * 1e6), H=diag(1) * sigma2_y)
kf <- KFS(mod, c("state", "signal", "mean"), c("state", "signal", "mean", "disturbance"), simplify=FALSE)
write.csv(output(kf), 'tsa/statespace/tests/results/results_exact_initial_local_level_R.csv', row.names=FALSE)
# Note: Apparent loglikelihood discrepancy
print(llf(kf))    # -22.97437545
print(kf$logLik)  # -21.13649839
print(-22.97437545 - -21.13649839) / kf$d  # -0.9189385332
print(-0.5 * log(pi * 2))  # -0.9189385332

# Local linear trend
y1 <- 10.2394
y2 <- 4.2039
y3 <- 6.123123
sigma2_y <- 1.993
sigma2_mu <- 8.253
sigma2_beta <- 2.334
obs <- c(c(y1, y2, y3), rep(1, 7))
mod <- SSModel(obs ~ -1 + SSMcustom(Z=t(as.matrix(c(1, 0))), T=as.matrix(cbind(c(1., 0), c(1, 1))), R=diag(2), Q=diag(c(sigma2_mu, sigma2_beta)), P1inf=diag(2)), H=diag(1) * sigma2_y)
kf <- KFS(mod, c("state", "signal", "mean"), c("state", "signal", "mean", "disturbance"), simplify=FALSE)
write.csv(output(kf), 'tsa/statespace/tests/results/results_exact_initial_local_linear_trend_R.csv', row.names=FALSE)

# Local linear trend - missing
obs[2] <- NA
mod <- SSModel(obs ~ -1 + SSMcustom(Z=t(as.matrix(c(1, 0))), T=as.matrix(cbind(c(1., 0), c(1, 1))), R=diag(2), Q=diag(c(sigma2_mu, sigma2_beta)), P1inf=diag(2)), H=diag(1) * sigma2_y)
kf <- KFS(mod, c("state", "signal", "mean"), c("state", "signal", "mean", "disturbance"), simplify=FALSE)
write.csv(output(kf), 'tsa/statespace/tests/results/results_exact_initial_local_linear_trend_missing_R.csv', row.names=FALSE)


# Common level
y11 <- 10.2394
y21 <- 8.2304
theta <- 0.1111
sigma2_1 <- 1
sigma_12 <- 0
sigma2_2 <- 1
sigma2_mu <- 3.2324
obs <- cbind(
  c(c(y11), rep(1, 9)),
  c(c(y21), rep(1, 9))
)
mod <- SSModel(obs ~ -1 + SSMcustom(
    Z=as.matrix(cbind(c(1., theta), c(0, 1))),
    T=diag(2),
    R=as.matrix(c(1, 0)),
    Q=diag(1) * sigma2_mu,
    P1inf=diag(2)),
  H=diag(2))
kf <- KFS(mod, c("state", "signal", "mean"), c("state", "signal", "mean", "disturbance"), simplify=FALSE)
write.csv(output(kf), 'tsa/statespace/tests/results/results_exact_initial_common_level_R.csv', row.names=FALSE)

# Common level - restricted
mod <- SSModel(obs ~ -1 + SSMcustom(
  Z=as.matrix(c(1, theta)),
  T=diag(1),
  R=diag(1),
  Q=diag(1) * sigma2_mu,
  P1inf=diag(1)),
  H=diag(2))
kf <- KFS(mod, c("state", "signal", "mean"), c("state", "signal", "mean", "disturbance"), simplify=FALSE)
write.csv(output(kf), 'tsa/statespace/tests/results/results_exact_initial_common_level_restricted_R.csv', row.names=FALSE)
