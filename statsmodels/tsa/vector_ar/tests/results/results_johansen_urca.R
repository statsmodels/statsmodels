## Reference values for statsmodels.tsa.vector_ar.vecm.coint_johansen
##
## urca::ca.jo with ecdet = "none" (an unrestricted constant, the same model
## as det_order=0 in coint_johansen) on the Danish data that ships with
## statsmodels as statsmodels/datasets/danish_data/data.csv (identical to
## urca::denmark). K is k_ar_diff + 1. The statistics of ca.jo are ordered from
## the largest rank to the smallest, so they are reversed here. The critical
## values of ca.jo for ecdet = "none" come from a different table than the
## ones of coint_johansen and are not compared.
##
## Python test is: statsmodels/tsa/vector_ar/tests/test_coint.py
## ::test_johansen_urca_danish
##
## Run with R 4.6.1 and urca 1.3.4 (on Windows):
##   Rscript results_johansen_urca.R
## install.packages("urca")
library(urca)
options(digits = 16)

danish <- read.csv("../../../../datasets/danish_data/data.csv")
y <- as.matrix(danish[, c("lrm", "lry", "ibo", "ide")])
data(denmark)
stopifnot(isTRUE(all.equal(unname(y), unname(as.matrix(denmark[, c("LRM", "LRY", "IBO", "IDE")])),
                           tolerance = 1e-12)))

for (K in c(2, 3)) {
  trace <- ca.jo(y, type = "trace", ecdet = "none", K = K)
  eigen <- ca.jo(y, type = "eigen", ecdet = "none", K = K)
  cat("k_ar_diff =", K - 1, "\n")
  cat("eig =", format(trace@lambda[1:4], digits = 16), "\n")
  cat("lr1 =", format(rev(trace@teststat), digits = 16), "\n")
  cat("lr2 =", format(rev(eigen@teststat), digits = 16), "\n")
}
