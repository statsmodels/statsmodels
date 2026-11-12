# Reference values for QuantReg.fit(method="fn").
#
# Run from this directory with
#
#   Rscript results_quantreg_fn.R
#
# Generated with R 4.6.2 and quantreg 6.1. The script writes
#
# - quantreg_sim.csv: a simulated data set with an intercept and five
#   regressors on different scales and heteroskedastic, heavy tailed errors.
# - results_quantreg_fn.csv: the coefficients of rq(method="fn") (Frisch-Newton
#   interior point) and rq(method="br") (Barrodale-Roberts simplex) for the
#   engel data of the quantreg package and for the simulated data, and the
#   value of the objective function sum(rho_tau(residuals)).
#
# quantreg is only used to compute the numbers. The default tolerance of
# rq.fit.fnb is eps = 1e-6 on the duality gap.

library(quantreg)

taus <- c(0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98)

set.seed(3770)
n <- 500
x1 <- rnorm(n)
x2 <- runif(n, 0, 10)
x3 <- 100 * rnorm(n)
x4 <- rbinom(n, 1, 0.3)
x5 <- rexp(n)
err <- (1 + 0.5 * abs(x1)) * rt(n, df = 3)
y <- 1 + x1 - 0.5 * x2 + 0.01 * x3 + 2 * x4 - x5 + err
sim <- data.frame(y = y, x1 = x1, x2 = x2, x3 = x3, x4 = x4, x5 = x5)
fmt <- function(v) sprintf("%.17g", v)
sim_out <- as.data.frame(lapply(sim, fmt))
write.csv(sim_out, "quantreg_sim.csv", row.names = FALSE, quote = FALSE)

data(engel)
datasets <- list(
  engel = list(formula = foodexp ~ income, data = engel),
  sim = list(formula = y ~ x1 + x2 + x3 + x4 + x5, data = sim)
)

rho <- function(u, tau) sum(u * (tau - (u < 0)))

rows <- list()
for (name in names(datasets)) {
  ds <- datasets[[name]]
  for (tau in taus) {
    for (method in c("fn", "br")) {
      fit <- rq(ds$formula, tau = tau, data = ds$data, method = method)
      cf <- coef(fit)
      rows[[length(rows) + 1]] <- data.frame(
        dataset = name,
        method = method,
        q = tau,
        term = names(cf),
        index = seq_along(cf) - 1,
        value = fmt(cf),
        objective = fmt(rho(resid(fit), tau))
      )
    }
  }
}
out <- do.call(rbind, rows)
write.csv(out, "results_quantreg_fn.csv", row.names = FALSE, quote = FALSE)
