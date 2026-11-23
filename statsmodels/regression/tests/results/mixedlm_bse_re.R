# Reference standard errors of the variance parameters for test_bse_re_nlme
# in test_lme.py.  Run from statsmodels/regression/tests with
#
#   Rscript results/mixedlm_bse_re.R
#
# m$apVar is the inverse of the numerical Hessian of the log-likelihood in the
# log standard deviations, the log((1 + r) / (1 - r)) of the correlations and
# log(sigma).  The delta method maps it to the variances and covariances, in
# the order of MixedLM, and sigma^2.  The default .relStep of the numerical
# Hessian is too small for these models, 1e-4 is used.
library(nlme)

natural <- function(p, dims) {
  out <- c()
  pos <- 0
  for (k in dims) {
    sd <- exp(p[pos + seq_len(k)])
    pos <- pos + k
    cv <- diag(sd^2, k)
    if (k > 1) {
      z <- p[pos + seq_len(k * (k - 1) / 2)]
      pos <- pos + k * (k - 1) / 2
      r <- diag(k)
      r[lower.tri(r)] <- (exp(z) - 1) / (exp(z) + 1)
      r[upper.tri(r)] <- t(r)[upper.tri(r)]
      cv <- r * outer(sd, sd)
    }
    for (i in seq_len(k)) out <- c(out, cv[i, seq_len(i)])
  }
  c(out, exp(2 * p[length(p)]))
}

delta_se <- function(m) {
  pars <- attr(m$apVar, "Pars")
  dims <- unlist(lapply(m$modelStruct$reStruct, function(x) Dim(x)[1]))
  h <- 1e-6
  jac <- sapply(seq_along(pars), function(j) {
    e <- replace(numeric(length(pars)), j, h)
    (natural(pars + e, dims) - natural(pars - e, dims)) / (2 * h)
  })
  sqrt(diag(jac %*% m$apVar %*% t(jac)))
}

ctl <- lmeControl(tolerance = 1e-10, msTol = 1e-12, niterEM = 0, .relStep = 1e-4)
dietox <- read.csv("results/dietox.csv")
pastes <- read.csv("results/pastes.csv")
for (method in c("REML", "ML")) {
  m <- list(
    dietox = lme(Weight ~ Time, random = ~ 1 | Pig, data = dietox,
                 method = method, control = ctl),
    dietox_slope = lme(Weight ~ Time, random = ~ 1 + Time | Pig, data = dietox,
                       method = method, control = ctl),
    # nlme lists the inner grouping (cask) first, MixedLM the group (batch)
    pastes = lme(strength ~ 1, random = ~ 1 | batch / cask, data = pastes,
                 method = method, control = ctl)
  )
  for (name in names(m)) {
    se <- delta_se(m[[name]])
    if (name == "pastes") se <- se[c(2, 1, 3)]
    cat(sprintf("%s %s: [%s]\n", name, method,
                paste(sprintf("%.6g", se), collapse = ", ")))
  }
}
