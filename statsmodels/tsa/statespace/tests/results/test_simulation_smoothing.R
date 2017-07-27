library(KFAS)
options(digits=10)

# should run this from the statsmodels/statsmodels directory
dta <- read.csv('datasets/macrodata/macrodata.csv')
obs <- diff(log(data.matrix(dta[c('realgdp','realcons','realinv')])))[1:9,]

T <- t(matrix(
  c(-0.1119908792, 0.8441841604,  0.0238725303,
    0.2629347724, 0.4996718412, -0.0173023305,
    -3.2192369082, 4.1536028244,  0.4514379215), nrow=3, ncol=3))
Q <- t(matrix(
  c(0.0000640649, 0.0000388496, 0.0002148769,
    0.0000388496, 0.0000572802, 0.000001555,
    0.0002148769, 0.000001555,  0.0017088585), nrow=3, ncol=3))
H <- t(matrix(
  c(0.0000640649, 0.,           0.,
    0.,           0.0000572802, 0.,
    0.,           0.,           0.0017088585), nrow=3, ncol=3))

mod <- SSModel(obs ~ -1 + SSMcustom(Z=diag(3), T=T, R=diag(3), Q=Q, P1=diag(3)*1e6),  H=H)
kf <- KFS(mod, c("state", "signal", "mean"), c("state", "signal", "mean", "disturbance"), simplify=FALSE)
kf$logLik # 39.01246166

# - Test 0 ------------------------------------------------------------------ #

# use fixInNamespace(simHelper, 'KFAS') to set u to zeros
# u <- rep(0, dfu * nsim)
set.seed(1234)
sim_eta <- simulateSSM(mod, type=c("eta"))
sim_eps <- simulateSSM(mod, type=c("epsilon"))
sim_state <- simulateSSM(mod, type=c("states"))
sim_signal <- simulateSSM(mod, type=c("signals"))

out <- as.data.frame(list(sim_eta, sim_eps, sim_state, sim_signal))
names(out) <- c(
  'eta1', 'eta2', 'eta3',
  'eps1', 'eps2', 'eps3',
  'state1', 'state2', 'state3',
  'signal1', 'signal2', 'signal3'
)

write.csv(out, 'tsa/statespace/tests/results/results_simulation_smoothing0.csv', row.names=FALSE)

# - Test 1 ------------------------------------------------------------------ #

# use fixInNamespace(simHelper, 'KFAS') to set u to zeros, except set eps
# variates to arange / 10
# u[1:(dfeps * nsim)] = (0:(dfeps * nsim - 1))/10
set.seed(1234)
sim_eta <- simulateSSM(mod, type=c("eta"))
sim_eps <- simulateSSM(mod, type=c("epsilon"))
sim_state <- simulateSSM(mod, type=c("states"))
sim_signal <- simulateSSM(mod, type=c("signals"))

out <- as.data.frame(list(sim_eta, sim_eps, sim_state, sim_signal))
names(out) <- c(
  'eta1', 'eta2', 'eta3',
  'eps1', 'eps2', 'eps3',
  'state1', 'state2', 'state3',
  'signal1', 'signal2', 'signal3'
)

write.csv(out, 'tsa/statespace/tests/results/results_simulation_smoothing1.csv', row.names=FALSE)

# - Test 2 ------------------------------------------------------------------ #

# use fixInNamespace(simHelper, 'KFAS') to set a1plus to zeros, set eps and eta
# variates to arange / 10
# u[1:(dfeps * nsim)] = (0:(dfeps * nsim - 1))/10
# u[(dfeps * nsim + 1):(dfeps * nsim + dfeta * nsim)] = (0:(dfeps * nsim - 1))/10
set.seed(1234)
sim_eta <- simulateSSM(mod, type=c("eta"))
sim_eps <- simulateSSM(mod, type=c("epsilon"))
sim_state <- simulateSSM(mod, type=c("states"))
sim_signal <- simulateSSM(mod, type=c("signals"))

out <- as.data.frame(list(sim_eta, sim_eps, sim_state, sim_signal))
names(out) <- c(
  'eta1', 'eta2', 'eta3',
  'eps1', 'eps2', 'eps3',
  'state1', 'state2', 'state3',
  'signal1', 'signal2', 'signal3'
)

write.csv(out, 'tsa/statespace/tests/results/results_simulation_smoothing2.csv', row.names=FALSE)

# - Test 3 ------------------------------------------------------------------ #

# Reset obs to full dataset
obs <- diff(log(data.matrix(dta[c('realgdp','realcons','realinv')])))

T <- t(matrix(
  c(-0.1119908792, 0.8441841604,  0.0238725303,
    0.2629347724, 0.4996718412, -0.0173023305,
    -3.2192369082, 4.1536028244,  0.4514379215), nrow=3, ncol=3))
Q <- t(matrix(
  c(0.0000640649, 0.0000388496, 0.0002148769,
    0.0000388496, 0.0000572802, 0.000001555,
    0.0002148769, 0.000001555,  0.0017088585), nrow=3, ncol=3))
H <- t(matrix(
  c(0.0000640649, 0.,           0.,
    0.,           0.0000572802, 0.,
    0.,           0.,           0.0017088585), nrow=3, ncol=3))

mod <- SSModel(obs ~ -1 + SSMcustom(Z=diag(3), T=T, R=diag(3), Q=Q, P1=diag(3)*1e6),  H=H)
kf <- KFS(mod, c("state", "signal", "mean"), c("state", "signal", "mean", "disturbance"), simplify=FALSE)
kf$logLik # 1695.34872

# if run sequentially, use fixInNamespace(simHelper, 'KFAS') to remove the
# modifications to u
# Instead, capture the actual variates drawn here
set.seed(1234)
variates <- rnorm(6 * nrow(obs) + 3, mean = 0, sd = 1)
# Note: need to reset the seed before each call to make sure we always use
# the same variates.
set.seed(1234)
sim_eta <- simulateSSM(mod, type=c("eta"))
set.seed(1234)
sim_eps <- simulateSSM(mod, type=c("epsilon"))
set.seed(1234)
sim_state <- simulateSSM(mod, type=c("states"))
set.seed(1234)
sim_signal <- simulateSSM(mod, type=c("signals"))

out <- as.data.frame(list(sim_eta, sim_eps, sim_state, sim_signal))
names(out) <- c(
  'eta1', 'eta2', 'eta3',
  'eps1', 'eps2', 'eps3',
  'state1', 'state2', 'state3',
  'signal1', 'signal2', 'signal3'
)

write.csv(variates, 'tsa/statespace/tests/results/results_simulation_smoothing3_variates.csv', row.names=FALSE)
write.csv(out, 'tsa/statespace/tests/results/results_simulation_smoothing3.csv', row.names=FALSE)

# - Test 4 ------------------------------------------------------------------ #

# Now with some fully missing observations
obs <- diff(log(data.matrix(dta[c('realgdp','realcons','realinv')])))
obs[1:50,] <- NaN

mod <- SSModel(obs ~ -1 + SSMcustom(Z=diag(3), T=T, R=diag(3), Q=Q, P1=diag(3)*1e6),  H=H)
kf <- KFS(mod, c("state", "signal", "mean"), c("state", "signal", "mean", "disturbance"), simplify=FALSE)
kf$logLik # 1305.739288

# The actual variates will be the same as in the Test 3
# Note: need to reset the seed before each call to make sure we always use
# the same variates.
set.seed(1234)
sim_eta <- simulateSSM(mod, type=c("eta"))
set.seed(1234)
sim_eps <- simulateSSM(mod, type=c("epsilon"))
set.seed(1234)
sim_state <- simulateSSM(mod, type=c("states"))
set.seed(1234)
sim_signal <- simulateSSM(mod, type=c("signals"))

out <- as.data.frame(list(sim_eta, sim_eps, sim_state, sim_signal))
names(out) <- c(
  'eta1', 'eta2', 'eta3',
  'eps1', 'eps2', 'eps3',
  'state1', 'state2', 'state3',
  'signal1', 'signal2', 'signal3'
)

write.csv(out, 'tsa/statespace/tests/results/results_simulation_smoothing4.csv', row.names=FALSE)

# - Test 5 ------------------------------------------------------------------ #

# Now with some partially missing observations
obs <- diff(log(data.matrix(dta[c('realgdp','realcons','realinv')])))
obs[1:50,1] <- NaN

# The actual variates will be the same as in the Test 3
mod <- SSModel(obs ~ -1 + SSMcustom(Z=diag(3), T=T, R=diag(3), Q=Q, P1=diag(3)*1e6),  H=H)
kf <- KFS(mod, c("state", "signal", "mean"), c("state", "signal", "mean", "disturbance"), simplify=FALSE)
kf$logLik # 1518.449598

# The actual variates will be the same as in the Test 3
# Note: need to reset the seed before each call to make sure we always use
# the same variates.
set.seed(1234)
sim_eta <- simulateSSM(mod, type=c("eta"))
set.seed(1234)
sim_eps <- simulateSSM(mod, type=c("epsilon"))
set.seed(1234)
sim_state <- simulateSSM(mod, type=c("states"))
set.seed(1234)
sim_signal <- simulateSSM(mod, type=c("signals"))

out <- as.data.frame(list(sim_eta, sim_eps, sim_state, sim_signal))
names(out) <- c(
  'eta1', 'eta2', 'eta3',
  'eps1', 'eps2', 'eps3',
  'state1', 'state2', 'state3',
  'signal1', 'signal2', 'signal3'
)

write.csv(out, 'tsa/statespace/tests/results/results_simulation_smoothing5.csv', row.names=FALSE)

# - Test 6 ------------------------------------------------------------------ #

# Now with both fully and partially missing observations
obs <- diff(log(data.matrix(dta[c('realgdp','realcons','realinv')])))

obs[1:50,1] <- NaN
obs[20:70,2] <- NaN
obs[40:90,3] <- NaN
obs[120:130,1] <- NaN
obs[120:130,3] <- NaN
obs[193:202,] <- NaN

mod <- SSModel(obs ~ -1 + SSMcustom(Z=diag(3), T=T, R=diag(3), Q=Q, P1=diag(3)*1e6),  H=H)
kf <- KFS(mod, c("state", "signal", "mean"), c("state", "signal", "mean", "disturbance"), simplify=FALSE)
kf$logLik # 1108.341725

# The actual variates will be the same as in the Test 3
# Note: need to reset the seed before each call to make sure we always use
# the same variates.
set.seed(1234)
sim_eta <- simulateSSM(mod, type=c("eta"))
set.seed(1234)
sim_eps <- simulateSSM(mod, type=c("epsilon"))
set.seed(1234)
sim_state <- simulateSSM(mod, type=c("states"))
set.seed(1234)
sim_signal <- simulateSSM(mod, type=c("signals"))

out <- as.data.frame(list(sim_eta, sim_eps, sim_state, sim_signal))
names(out) <- c(
  'eta1', 'eta2', 'eta3',
  'eps1', 'eps2', 'eps3',
  'state1', 'state2', 'state3',
  'signal1', 'signal2', 'signal3'
)

write.csv(out, 'tsa/statespace/tests/results/results_simulation_smoothing6.csv', row.names=FALSE)
