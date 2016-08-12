library(KFAS)
options(digits=10)

# should run this from the statsmodels/statsmodels directory
dta <- read.csv('tsa/statespace/tests/results/results_wpi1_ar3_stata.csv')
matlab <- read.csv('tsa/statespace/tests/results/results_wpi1_missing_ar3_matlab_ssm.csv')
names(matlab) <- c(
  'a1','a2','a3','detP','alphahat1','alphahat2','alphahat3',
  'detV','eps','epsvar','eta','etavar')

endog <- diff(dta$wpi)

Z <- matrix(c(1, 0, 0), nrow=1, ncol=3)
H <- matrix(0, nrow=1, ncol=1)
T <- t(matrix(
  c(.5270715, .0952613, .2580355,
    1,        0,        0,
    0,        1,        0), nrow=3, ncol=3))
Q <- matrix(.5307459, nrow=1, ncol=1)
R <- matrix(0, nrow=3, ncol=1)
R[1,1] <- 1
P <- t(matrix(
  c(1.58276997,  1.24351589,  1.12706975,
    1.24351589,  1.58276997,  1.24351589,
    1.12706975,  1.24351589,  1.58276997), nrow=3, ncol=3))

mod <- SSModel(endog ~ -1 + SSMcustom(Z=Z, T=T, R=R, Q=Q, P1=P),  H=H)
kf <- KFS(mod, c("state", "signal", "mean"), c("state", "signal", "mean", "disturbance"), simplify=FALSE)

# kf$logLik # -130.0310409

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

out <- as.data.frame(with(kf, cbind(
  t(r)[2:124,], apply(N, 3, det)[2:124],
  m, apply(P_mu, 3, det), v, t(F),
  a[2:124,], apply(P, 3, det)[2:124],
  alphahat, apply(V, 3, det),
  muhat, apply(V_mu, 3, det),
  etahat, apply(V_eta, 3, det), epshat, t(V_eps)
)))
names(out) <- c(
  "r1", "r2", "r3", "detN",
  "m", "detPmu",
  "v", "F",
  "a1", "a2", "a3", "detP",
  "alphahat1", "alphahat2", "alphahat3", "detV",
  "muhat", "detVmu",
  "etahat", "detVeta",
  "epshat", "Veps"
)
write.csv(out, 'tsa/statespace/tests/results/results_smoothing3_R.csv', row.names=FALSE)
