d <- read.csv('./ccard.csv')
attach(d)


m1 <- lm(AVGEXP ~ AGE + INCOME + INCOMESQ + OWNRENT, weights=1/INCOMESQ)
results <- summary(m1)

m2 <- lm(AVGEXP ~ AGE + INCOME + INCOMESQ + OWNRENT - 1, weights=1/INCOMESQ)
results2 <- summary(m2)

print('m1 has a constant, which theoretically should be INCOME')
print('m2 include -1 for no constant')
print('See ccard/R_wls.s')
