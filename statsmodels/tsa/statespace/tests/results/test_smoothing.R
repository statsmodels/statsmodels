library(KFAS)
options(digits=10)

# should run this from the statsmodels/statsmodels directory
dta <- read.csv('datasets/macrodata/macrodata.csv')

obs <- diff(data.matrix(dta[c('realgdp','realcons','realinv')]))
obs[1:50,1] <- NaN
obs[20:70,2] <- NaN
obs[40:90,3] <- NaN
obs[120:130,1] <- NaN
obs[120:130,3] <- NaN

mod <- SSModel(obs ~ -1 + SSMcustom(Z=diag(3), T=diag(3), R=diag(3), Q=diag(3), P1=diag(3)*1e6),  H=diag(3))
kf <- KFS(mod, c("state", "signal", "mean"), c("state", "signal", "mean", "disturbance"), simplify=FALSE)

# kf$logLik = -205310.9767

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
  t(r)[2:203,], apply(N, 3, det)[2:203],
  m, apply(P_mu, 3, det), v, t(F),
  a[2:203,], apply(P, 3, det)[2:203],
  alphahat, apply(V, 3, det),
  muhat, apply(V_mu, 3, det),
  etahat, apply(V_eta, 3, det), epshat, t(V_eps)
)))
names(out) <- c(
  "r1", "r2", "r3", "detN",
  "m1", "m2", "m3", "detPmu",
  "v1", "v2", "v3", "F1", "F2", "F3",
  "a1", "a2", "a3", "detP",
  "alphahat1", "alphahat2", "alphahat3", "detV",
  "muhat1", "muhat2", "muhat3", "detVmu",
  "etahat1", "etahat2", "etahat3", "detVeta",
  "epshat1", "epshat2", "epshat3", "Veps1", "Veps2", "Veps3"
)
write.csv(out, 'tsa/statespace/tests/results/results_smoothing_R.csv', row.names=FALSE)
