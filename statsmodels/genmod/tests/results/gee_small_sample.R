# Reference values for the GEE small-sample covariances in test_gee.py.
# R 4.5.2, gee 4.13-30, geesmv 1.3.
# geesmv is archived on CRAN; install version 1.3 from the archive:
#   install.packages(
#     "https://cran.r-project.org/src/contrib/Archive/geesmv/geesmv_1.3.tar.gz",
#     repos = NULL, type = "source")
# gee_small_sample.csv is simulated data: 64 observations in 15 clusters of
# 2 to 6, with covariates x1 and x2 and Gaussian, Poisson and binary
# responses. It is stored so the reference values do not depend on an RNG.
# All reference values use the independence working correlation; other
# working correlations are checked in Python only.
# geesmv's Kauermann-Carroll uses a symmetric matrix root, which is only
# valid when I - H is symmetric (Gaussian, independence), so it is used
# only there. Mancl-DeRouen is used for all three families.
library(geesmv)
dat <- read.csv("gee_small_sample.csv")
f <- function(fun, y, fam) {
  r <- fun(as.formula(paste(y, "~ x1 + x2")), id = "id", family = fam,
           data = dat, corstr = "independence")
  sprintf("%.12e", r$cov.beta)
}
out <- rbind(
  c("gaussian", "kc", f(GEE.var.kc, "y_gauss", gaussian)),
  c("gaussian", "md", f(GEE.var.md, "y_gauss", gaussian)),
  c("poisson", "md", f(GEE.var.md, "y_pois", poisson)),
  c("binomial", "md", f(GEE.var.md, "y_bin", binomial))
)
colnames(out) <- c("family", "method", "var0", "var1", "var2")
write.csv(out, "gee_small_sample_geesmv.csv", row.names = FALSE, quote = FALSE)
