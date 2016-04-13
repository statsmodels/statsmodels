library(KFAS)
library(MARSS)
options(digits=20)

# should run this from the statsmodels/statsmodels directory
dta <- read.csv('datasets/macrodata/macrodata.csv')

obs <- data.matrix(dta[c('realgdp','realcons','realinv')])
obs[,1] <- obs[,1] / sd(obs[,1], na.rm=TRUE)
obs[,2] <- obs[,2] / sd(obs[,2], na.rm=TRUE)
obs[,3] <- obs[,3] / sd(obs[,3], na.rm=TRUE)

obs[1:50,1] <- NA
obs[20:70,2] <- NA
obs[40:90,3] <- NA
obs[120:130,1] <- NA
obs[120:130,3] <- NA

Z <- diag(3)
H <- diag(3)
T <- diag(3)
R <- diag(3)
Q <- diag(3)
P0 <- diag(3) * 1e6

d <- matrix(c(1,2,3),3,1)
c <- matrix(c(4,5,6),3,1)

ss <- list(Z=Z, R=H, B=T, Q=Q, V0=P0, A=d, U=c, x0="zero", tinitx=1)
marssmod <- MARSS(t(obs), model=ss)
marssout <- MARSSkfas(marssmod, return.kfas.model=TRUE)
kf <- KFS(marssout$kfas.model, c("state", "signal", "mean"), c("state", "signal", "mean", "disturbance"), simplify=FALSE)

# kf$logLik  # -7924.03893566

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
  t(r)[2:204,1:3], apply(N[1:3,1:3,], 3, det)[2:204],
  m, apply(P_mu, 3, det), v, t(F),
  a[2:204,1:3], apply(P[1:3,1:3,], 3, det)[2:204],
  alphahat[,1:3], apply(V[1:3,1:3,], 3, det),
  muhat, apply(V_mu, 3, det),
  etahat[,1:3], apply(V_eta[1:3,1:3,], 3, det), epshat, t(V_eps)
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
write.csv(out, 'tsa/statespace/tests/results/results_intercepts_R.csv', row.names=FALSE)
