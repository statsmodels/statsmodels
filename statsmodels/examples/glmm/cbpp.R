library('lme4')
m <- glmer(incidence ~ period + (1 | herd), family = poisson, data = cbpp)
print(summary(m))
print(logLik(m))
print(summary(m)$coef)
print(VarCorr(m))

