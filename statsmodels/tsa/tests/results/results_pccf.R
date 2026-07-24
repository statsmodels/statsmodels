## Generate PCCF reference values via OLS regression
##
## The partial cross-correlation at lag h between x and y is defined as
## the Pearson correlation between the residuals from regressing x[t]
## and y[t+h] on all intervening observations {x[t+j], y[t+j]: j=1..h-1}
## plus intercept.  At lag 1 there are no intervening observations,
## so pccf(1) = cor(x[1:(n-1)], y[2:n]).
##
## Data: macrodata realgdp and realcons from statsmodels.
## To export: python -c "from statsmodels.datasets import macrodata; \
##   macrodata.load_pandas().data.to_csv('macrodata.csv', index=False)"

options(digits = 17, scipen = 999)

macro <- read.csv("macrodata.csv")
x <- macro$realgdp
y <- macro$realcons
n <- length(x)
nlags <- 20

pccf_vals <- numeric(nlags)

for (h in 1:nlags) {
  if (h == 1) {
    rx <- x[1:(n - 1)]
    ry <- y[2:n]
  } else {
    idx <- 1:(n - h)
    carriers <- matrix(1, nrow = length(idx), ncol = 1)
    for (j in 1:(h - 1)) {
      carriers <- cbind(carriers, x[idx + j], y[idx + j])
    }
    targets_x <- x[idx]
    targets_y <- y[idx + h]
    fit_x <- lm.fit(carriers, targets_x)
    fit_y <- lm.fit(carriers, targets_y)
    rx <- fit_x$residuals
    ry <- fit_y$residuals
  }
  pccf_vals[h] <- cor(rx, ry)
}

out <- data.frame(pccf = pccf_vals)
write.csv(out, "results_pccf.csv", row.names = FALSE)
cat("PCCF values:\n")
print(pccf_vals)
