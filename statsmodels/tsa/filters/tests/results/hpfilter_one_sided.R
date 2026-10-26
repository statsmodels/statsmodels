# One-sided HP filter results for test_hpfilter_one_sided_r
# R 4.5.3, hpfilter 1.0.2
# Run from statsmodels/tsa/filters/tests/results
library(hpfilter)

macrodata <- read.csv("../../../../datasets/macrodata/macrodata.csv")
y <- data.frame(realgdp = macrodata$realgdp)
lambdas <- c(6.25, 1600, 129600)
trend <- sapply(lambdas, function(lamb) hp1(y, lambda = lamb)[, 1])
colnames(trend) <- paste0("lamb_", lambdas)
write.csv(format(as.data.frame(trend), digits = 17, trim = TRUE), "hpfilter_one_sided.csv",
          row.names = FALSE, quote = FALSE)
