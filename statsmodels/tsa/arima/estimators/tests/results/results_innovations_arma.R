library(itsmr)
options(digits=10)

# Reference values for the innovations estimates of ARMA models on the lake
# dataset, with recursion level m = 17.
# Python test is: test_innovations::test_innovations_arma_itsmr
#
# itsmr::ia(x, q, m) returns the coefficients theta_{m,1}, ..., theta_{m,q}
# of the MA(m) model fitted with the innovations algorithm. itsmr does not
# implement the ARMA step, so it is computed here from section 5.1.3 of
# Brockwell and Davis (2016): theta_{m,j} = sum_i phi_i theta_{m,j-i} for
# j = q+1, ..., q+p (with theta_{m,0} = 1), then
# theta_j = theta_{m,j} - sum_i phi_i theta_{m,j-i} for j = 1, ..., q.
m <- 17

# MA(2)
print(ia(lake, 2, m = m)$theta)
# [1] 1.0830783033 0.7835383743

# ARMA(1, 1)
th <- ia(lake, 2, m = m)$theta
phi <- th[2] / th[1]
theta <- th[1] - phi
print(c(phi, theta))
# [1] 0.7234364976 0.3596418057

# White noise variance as reported by itsmr for the fitted ARMA(1, 1) model:
# the mean of the squared standardized one-step prediction errors
x <- lake - mean(lake)
print(itsmr:::.innovation.update(x, specify(ar = phi, ma = theta))$sigma2)
# [1] 0.475680222

# ARMA(2, 1)
th <- ia(lake, 3, m = m)$theta
A <- matrix(c(th[1], th[2], 1, th[1]), 2, 2)
phi <- solve(A, th[2:3])
theta <- th[1] - phi[1]
print(c(phi, theta))
# [1]  0.75102521032 -0.02988073619  0.33205309296
