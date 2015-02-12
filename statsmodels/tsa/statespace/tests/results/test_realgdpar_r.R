library(FKF)

# Observations
df <- read.csv("results_kalman_filter_stata.csv")
gdp = df$value
lgdp = log(gdp)
dlgdp = diff(lgdp)

# Stata parameters
params <- c(
  0.40725515, 0.18782621, -0.01514009, -0.01027267, -0.03642297, 0.11576416,
  0.02573029, -.00766572, 0.13506498, 0.08649569, 0.06942822, -0.10685783,
  0.00008
)

# Measurement equation
n = 1
H = matrix(rep(0, 12), nrow=1)
H[1,1] = 1
R = matrix(c(0), nrow=1)

# Transition equation
k = 12
mu = matrix(rep(0,k), nrow=k)
F = matrix(rep(0,k*k), nrow=k)
for (i in 1:11) {
  F[i+1,i] = 1
}

# Q = G Q_star G'
Q = matrix(rep(0,k*k), nrow=k)

# Update matrices with given parameters
F[1,1:12] = params[1:12]
Q[1,1] = params[13]

# Initialization: Unconditional mean priors
initial_state = c(solve(diag(k) - F) %*% mu)
initial_state_cov = solve(diag(k^2) - F %x% F)  %*% matrix(c(Q))
dim(initial_state_cov) <- c(k,k)

# Filter
ans <- fkf(a0=initial_state, P0=initial_state_cov,
           dt=mu, ct=matrix(0), Tt=F, Zt=H,
           HHt=Q, GGt=R, yt=rbind(dlgdp))

write.csv(t(ans$att), 'results_states_gdp_R.csv', row.names=FALSE)
unlink('results_states_cov_gdp_R.csv')
for (t in 1:length(dlgdp)) {
  write.table(t(ans$Ptt[,,t]), 'results_states_cov_gdp_R.csv', sep=",", append=TRUE, row.names=FALSE)
}