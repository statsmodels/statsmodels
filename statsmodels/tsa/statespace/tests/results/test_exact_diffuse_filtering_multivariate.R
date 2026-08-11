library(KFAS)
options(digits=10)

# should run this from the statsmodels/statsmodels directory
setwd('~/projects/statsmodels-0.9/statsmodels/')
dta <- read.csv('datasets/macrodata/macrodata.csv')
source('tsa/statespace/tests/results/kfas_helpers.R')

# We use the following two observation datasets
obs <- (diff(log(data.matrix(dta[c('realgdp', 'realcons')]))) * 400)[1:20,]
obs_missing <- obs
obs_missing[1:5,1] <- NA
obs_missing[9:12,] <- NA

# VAR(1)
Z <- diag(2)
T <- as.matrix(cbind(c(0.5, 0.2), c(0.3, 0.4)))
R <- diag(2)
Q <- diag(c(2, 3))
H <- diag(2) * 0
mod <- SSModel(obs ~ -1 + SSMcustom(Z=Z, T=T, R=R, Q=Q, P1inf=diag(2)), H=H)
kf <- KFS(mod, c("state", "signal", "mean"), c("state", "signal", "mean", "disturbance"), simplify=FALSE)
write.csv(kfas.output(kf), 'tsa/statespace/tests/results/results_exact_initial_var1_R.csv', row.names=FALSE)

# VAR(1) + missing
mod <- SSModel(obs_missing ~ -1 + SSMcustom(Z=Z, T=T, R=R, Q=Q, P1inf=diag(2)), H=H)
kf <- KFS(mod, c("state", "signal", "mean"), c("state", "signal", "mean", "disturbance"), simplify=FALSE)
write.csv(kfas.output(kf), 'tsa/statespace/tests/results/results_exact_initial_var1_missing_R.csv', row.names=FALSE)

# VAR(1) + mixed initialization
stationary_init <- 3.5714285714285716
mod <- SSModel(obs ~ -1 + SSMcustom(Z=Z, T=T, R=R, Q=Q, P1inf=diag(c(1, 0)), P1=diag(c(0, stationary_init))), H=H)
kf <- KFS(mod, c("state", "signal", "mean"), c("state", "signal", "mean", "disturbance"), simplify=FALSE)
write.csv(kfas.output(kf), 'tsa/statespace/tests/results/results_exact_initial_var1_mixed_R.csv', row.names=FALSE)

# VAR(1) + measurement error
H <- diag(c(4, 5))
mod <- SSModel(obs ~ -1 + SSMcustom(Z=Z, T=T, R=R, Q=Q, P1inf=diag(2)), H=H)
kf <- KFS(mod, c("state", "signal", "mean"), c("state", "signal", "mean", "disturbance"), simplify=FALSE)
write.csv(kfas.output(kf), 'tsa/statespace/tests/results/results_exact_initial_var1_measurement_error_R.csv', row.names=FALSE)

# DFM
Z <- matrix(c(0.5, 1, 0, 0), nrow=2)
T <- matrix(c(0.9, 1, 0.1, 0), nrow=2)
R <- matrix(c(1, 0), nrow=2)
Q <- diag(1)
H <- diag(c(1.5, 2))
mod <- SSModel(obs ~ -1 + SSMcustom(Z=Z, T=T, R=R, Q=Q, P1inf=diag(2)), H=H)
kf <- KFS(mod, c("state", "signal", "mean"), c("state", "signal", "mean", "disturbance"), simplify=FALSE)
write.csv(kfas.output(kf), 'tsa/statespace/tests/results/results_exact_initial_dfm_R.csv', row.names=FALSE)
