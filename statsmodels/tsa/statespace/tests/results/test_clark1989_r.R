library(FKF)
library(KFAS)
options(digits=10)

# Observations
df <- read.csv("clark1989.csv", header=FALSE)
lgdp = log(df$V1[5:nrow(df)])
unemp = (df$V2 / 100)[5:nrow(df)]

# True parameters
params <- c(
  0.004863, 0.00668, 0.000295, 0.001518, 0.000306, 1.43859, -0.517385,
  -0.336789, -0.163511, -0.072012
)

# Dimensions
n = 2
k = 6

# Measurement equation
H = matrix(rep(0, n*k), nrow=n)
H[1,1] = 1
H[1,2] = 1
H[2,6] = 1
obs_intercept = matrix(rep(0,n), nrow=n)
R = matrix(rep(0, n^2), nrow=n)

# Transition equation

mu = matrix(rep(0, k), nrow=k)
F = matrix(rep(0, k^2), nrow=k)
F[1,1] = 1
F[1,5] = 1
F[3,2] = 1
F[4,3] = 1
F[5,5] = 1
F[6,6] = 1

# Q = G Q_star G'
Q = matrix(rep(0, k^2), nrow=k)

# Update matrices with given parameters
H[2,2] = params[8]
H[2,3] = params[9]
H[2,4] = params[10]
F[2,2] = params[6]
F[2,3] = params[7]
R[2,2] = params[5]^2
Q[1,1] = params[1]^2
Q[2,2] = params[2]^2
Q[5,5] = params[3]^2
Q[6,6] = params[4]^2

# Initialization: Diffuse priors
initial_state = c(mu)
initial_state_cov = diag(k) * 100

initial_state_cov = (F %*% initial_state_cov) %*% (t(F))

# - FKF --------------------------------------------------------------------- #

# Filter, complete dataset
dta <- rbind(lgdp,unemp)
ans <- fkf(a0=initial_state, P0=initial_state_cov,
           dt=mu, ct=obs_intercept, Tt=F, Zt=H,
           HHt=Q, GGt=R, yt=dta)

# Filter, partially missing dataset (using KFAS)
unemp2 = unemp
n = length(unemp2)
unemp2[(n-50):n] <- NaN
dta2 <- rbind(lgdp,unemp2)
ans2 <- fkf(a0=initial_state, P0=initial_state_cov,
            dt=mu, ct=obs_intercept, Tt=F, Zt=H,
            HHt=Q, GGt=R, yt=dta2)

mod <- SSModel(t(dta2) ~ -1+SSMcustom(H, F, diag(6), Q, P1=initial_state_cov), H=R)
fit <- KFS(mod, filtering="state", smoothing="state")

# Recalculate Kalman gain
# FKF defines the Kalman gain as P Z' F^{-1}, whereas Durbin and Koopman (2012)
# use T P Z' F^{-1}, so premultiply the ans$Kt by the transition matrix
Kt = ans$Kt
for (i in 1:n) {
  Kt[,,i] = F %*% Kt[,,i]
}

# Create output columns
# 1: sum of complete dataset Kalman gain
sum_complete_Kt = colSums(Kt, dims=2)
# 2-7: filtered states from incomplete dataset
# t(ans2$att)

# Print likelihood from partial missing model
print(fit$logLik)

out <- matrix(cbind(sum_complete_Kt, t(ans2$att)))
write.csv(out, 'results_clark1989_R.csv', row.names=FALSE)
