# One-sided HP filter results for test_hpfilter_one_sided_r
# R 4.5.3, base R only
# Run from statsmodels/tsa/filters/tests/results
# The one-sided trend at t is the last value of the two-sided HP trend of
# x[1:t], computed with a dense solve
hp_last <- function(x, lamb) {
  n <- length(x)
  if (n < 3) {
    return(x[n])
  }
  D <- diff(diag(n), differences = 2)
  solve(diag(n) + lamb * crossprod(D), x)[n]
}

macrodata <- read.csv("../../../../datasets/macrodata/macrodata.csv")
x <- macrodata$realgdp
lambdas <- c(6.25, 1600, 129600)
trend <- sapply(lambdas, function(lamb) {
  sapply(seq_along(x), function(t) hp_last(x[1:t], lamb))
})
colnames(trend) <- paste0("lamb_", lambdas)
write.csv(format(as.data.frame(trend), digits = 17, trim = TRUE),
          "hpfilter_one_sided.csv", row.names = FALSE, quote = FALSE)
