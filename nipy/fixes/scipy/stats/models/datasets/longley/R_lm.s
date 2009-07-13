d <- read.table('./longley.csv', sep=',', header=T)
attach(d)

library(nlme)   # to be able to get BIC
m1 <- lm(TOTEMP ~ GNPDEFL + GNP + UNEMP + ARMED + POP + YEAR)
results <-summary(m1)
