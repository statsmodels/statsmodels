# Reference values for the MixedLM fit tests against the lme4 optimum.
#
# Run generate_mixedlm_optimum.py first, then run this script from the tests
# directory (statsmodels/regression/tests) with
#
#   Rscript results/mixedlm_optimum.R
#
# It writes results/mixedlm_optimum_r_results.py.  The likelihood is flat near the
# optimum, so lme4 is run with a tight tolerance of the bobyqa optimizer and
# with the convergence checks off: only the optimum is used.
library(lme4)

data <- read.csv("results/mixedlm_optimum.csv")

forms <- list(
  slope = y ~ x + z + (1 + x | g),
  vc = y ~ x + z + (0 + x | g) + (0 + z | g)
)
control <- lmerControl(
  optimizer = "bobyqa",
  optCtrl = list(rhobeg = 0.2, rhoend = 1e-10, maxfun = 20000),
  check.conv.grad = .makeCC("ignore", 0),
  check.conv.singular = .makeCC("ignore", 0),
  check.conv.hess = .makeCC("ignore", 0)
)

num <- function(x) sprintf("%.15g", x)
vec <- function(x) paste0("[", paste(num(x), collapse = ", "), "]")

out <- c(
  '"""Reference values of lme4 for the MixedLM fit tests, see mixedlm_optimum.R."""',
  "",
  sprintf("# lme4 %s, R %s", packageVersion("lme4"), getRversion()),
  "mixedlm_optimum = {"
)
for (case in unique(data$case)) {
  d <- data[data$case == case, ]
  d$g <- factor(d$g)
  type <- sub("_.*", "", case)
  reml <- grepl("_reml$", case)
  m <- lmer(forms[[type]], d, REML = reml, control = control)
  if (type == "slope") {
    vc <- VarCorr(m)$g
    re <- sprintf('"cov_re": [%s, %s]',
                  vec(vc[1, ]), vec(vc[2, ]))
  } else {
    vc <- VarCorr(m)
    re <- sprintf('"vcomp": %s', vec(c(vc[[1]][1], vc[[2]][1])))
  }
  out <- c(out, sprintf('    "%s": {', case),
           sprintf('        "llf": %s,', num(as.numeric(logLik(m)))),
           sprintf('        "fe_params": %s,', vec(fixef(m))),
           sprintf('        %s,', re),
           sprintf('        "scale": %s,', num(sigma(m)^2)),
           sprintf('        "singular": %s,', ifelse(isSingular(m), "True", "False")),
           "    },")
}
out <- c(out, "}")
con <- file("results/mixedlm_optimum_r_results.py", open = "wb")
writeLines(out, con, sep = "\n")
close(con)
