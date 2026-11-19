# Reference values for the Fay-Graubard and Morel-Bokossa-Neerchal GEE
# covariances in test_gee.py.
# R 4.5.2, gee 4.13-30, geesmv 1.3, saws 0.9-7.0. geesmv and saws are
# installed from the GitHub mirror of CRAN (github.com/cran/geesmv, tag 1.3;
# github.com/cran/saws, version 0.9-7.0).
#
# geesmv only supports equal cluster sizes for the Gaussian family and for
# Morel-Bokossa-Neerchal with Poisson, so these use gee_fg_mbn.csv, which this
# script simulates: 12 clusters of 4 observations. Its first 8 clusters give
# the other branch of the Morel-Bokossa-Neerchal delta (8 <= (d + 1) p = 9).
# saws (by M. P. Fay) gives the full Fay-Graubard covariance, also for the
# unequal cluster sizes of gee_small_sample.csv. geesmv and saws return the
# same Fay-Graubard variances where both run. The 30 further rows of
# gee_fg_mbn.csv (ids 101-110, 10 clusters of 3) and the "hilev" row of the
# reference file, a cluster with a leverage above the Fay-Graubard bound, were
# added with R 4.6.1, which reproduces all earlier rows of both files exactly.
# Columns v0-v8 hold the 3 x 3 covariance row by row for fg_saws, and the
# three variances (v0-v2) for mbn_geesmv, which does not return covariances.
suppressMessages({
  library(gee)
  library(geesmv)
  library(saws)
})

# Converge tightly and fix the scale at 1 for Poisson and binomial, as in
# statsmodels. gee stops at tol = 0.001 by default, which leaves the
# exchangeable fits about 1e-4 from the solution. geesmv calls gee itself,
# so its calls are routed through the same settings.
gee_tight <- function(formula, data, id, family, corstr, ...) {
  fam <- if (is.function(family)) family() else family
  fixed <- fam$family != "gaussian"
  gee::gee(formula, data = data, id = id, family = family, corstr = corstr,
    tol = 1e-12, maxiter = 200, scale.fix = fixed, scale.value = 1)
}
imports <- parent.env(asNamespace("geesmv"))
unlockBinding("gee", imports)
assign("gee", gee_tight, envir = imports)

set.seed(10793)
n_groups <- 12
m <- 4
id <- rep(seq_len(n_groups), each = m)
x1 <- rnorm(n_groups * m)
x2 <- runif(n_groups * m, -1, 1)
u <- rep(rnorm(n_groups, sd = 0.5), each = m)
eta <- 0.5 + 0.4 * x1 - 0.3 * x2 + u
sim <- data.frame(
  id = id, x1 = round(x1, 6), x2 = round(x2, 6),
  y_gauss = round(eta + rnorm(n_groups * m), 6),
  y_pois = rpois(n_groups * m, exp(eta)),
  y_bin = rbinom(n_groups * m, 1, plogis(eta))
)

# Ten more clusters of 3 observations (ids 101-110) for the Fay-Graubard
# bound. x1 is about 15 in the first cluster, so that this cluster has a
# leverage of 0.985 for x1, far above the bound b = 0.75 (checked below). The
# draws come after those above, which keep their values.
set.seed(10794)
n_hi <- 10
id <- 100 + rep(seq_len(n_hi), each = 3)
x1 <- rnorm(n_hi * 3)
x1[1:3] <- 15 + c(-0.5, 0, 0.5)
x2 <- runif(n_hi * 3, -1, 1)
u <- rep(rnorm(n_hi, sd = 0.5), each = 3)
eta <- 0.5 + 0.4 * x1 - 0.3 * x2 + u
sim <- rbind(sim, data.frame(
  id = id, x1 = round(x1, 6), x2 = round(x2, 6),
  y_gauss = round(eta + rnorm(n_hi * 3), 6),
  y_pois = rpois(n_hi * 3, exp(eta)),
  y_bin = rbinom(n_hi * 3, 1, plogis(eta))
))
write.csv(sim, "gee_fg_mbn.csv", row.names = FALSE, quote = FALSE)

sim <- read.csv("gee_fg_mbn.csv")
unequal <- read.csv("gee_small_sample.csv")
families <- list(y_gauss = gaussian, y_pois = poisson, y_bin = binomial)
fmt <- function(v) sprintf("%.12e", v)

rows <- list()
add <- function(data_name, yname, corstr, method, values) {
  rows[[length(rows) + 1]] <<- c(data_name, yname, corstr, method, fmt(values))
}
for (data_name in c("equal", "equal8", "unequal")) {
  dat <- switch(data_name,
    equal = sim[sim$id <= 12, ], equal8 = sim[sim$id <= 8, ],
    unequal = unequal)
  for (yname in names(families)) {
    for (corstr in c("independence", "exchangeable")) {
      f <- as.formula(paste(yname, "~ x1 + x2"))
      fam <- families[[yname]]
      invisible(capture.output(
        fit <- gee(f, id = id, data = dat, family = fam, corstr = corstr,
          tol = 1e-12, maxiter = 200, scale.fix = yname != "y_gauss",
          scale.value = 1)
      ))
      # full Fay-Graubard covariance, bound b = 0.75
      add(data_name, yname, corstr, "fg_saws",
          saws(geeUOmega(fit), method = "d4", bound = 0.75)$V)
      if (data_name != "unequal") {
        invisible(capture.output(mbn <- GEE.var.mbn(
          f, id = "id", family = fam, data = dat, corstr = corstr)$cov.beta))
        add(data_name, yname, corstr, "mbn_geesmv", c(mbn, rep(NA, 6)))
      }
    }
  }
}

# one cluster with a leverage above the bound b = 0.75: saws caps it at 0.75
dat <- sim[sim$id > 100, ]
invisible(capture.output(
  fit <- gee(y_gauss ~ x1 + x2, id = id, data = dat, family = gaussian,
    corstr = "exchangeable", tol = 1e-12, maxiter = 200, scale.fix = FALSE,
    scale.value = 1)
))
uo <- geeUOmega(fit)
omega_inv <- solve(apply(uo$omega, c(2, 3), sum))
lev <- t(sapply(seq_len(dim(uo$omega)[1]),
  function(i) diag(uo$omega[i, , ] %*% omega_inv)))
stopifnot(max(lev) > 0.95)
add("hilev", "y_gauss", "exchangeable", "fg_saws",
    saws(uo, method = "d4", bound = 0.75)$V)

out <- do.call(rbind, rows)
colnames(out) <- c("data", "endog", "corstr", "method", paste0("v", 0:8))
write.csv(out, "gee_fg_mbn_ref.csv", row.names = FALSE, quote = FALSE)
