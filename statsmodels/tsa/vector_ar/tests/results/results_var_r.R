# Reference values for test_var.py from the R package vars (1.6.1, R 4.6.2)
#
# Rscript results_var_r.R <path to statsmodels/datasets/macrodata/macrodata.csv>
library(vars)

args <- commandArgs(trailingOnly = TRUE)
macro <- read.csv(args[1])
y <- sapply(log(macro[c("realgdp", "realcons", "realinv")]), diff)
n <- nrow(y)
p <- 2

# VAR(2) with a constant: moduli of the eigenvalues of the companion matrix
# and the forecast error variance decomposition
est <- VAR(y, p = p, type = "const")
cat("roots_modulus =", deparse(roots(est, modulus = TRUE), control = "digits17"), "\n")
dec <- fevd(est, n.ahead = 5)
for (nm in names(dec)) {
  cat("fevd", nm, "=", deparse(as.vector(t(dec[[nm]])), control = "digits17"), "\n")
}

# VAR(2) with a constant, a linear and a quadratic time trend, which is
# trend="ctt" in statsmodels. vars has no quadratic trend, so t ** 2 enters
# as an exogenous variable, with t = 1, ..., n as the time trend of vars.
tt2 <- matrix((1:n)^2, ncol = 1, dimnames = list(NULL, "tt2"))
est_ctt <- VAR(y, p = p, type = "both", exogen = tt2)
fut <- matrix((n + 1:5)^2, ncol = 1, dimnames = list(NULL, "tt2"))
fc <- predict(est_ctt, n.ahead = 5, dumvar = fut)
for (nm in names(fc$fcst)) {
  cat("fcst_ctt", nm, "=", deparse(fc$fcst[[nm]][, "fcst"], control = "digits17"), "\n")
}
