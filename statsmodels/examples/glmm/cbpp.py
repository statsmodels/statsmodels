import statsmodels.api as sm
import pandas as pd

# R code used to generated cbpp_R_mm.csv as the design matrix
# -----------------------------------------------------------
# library('lme4')
# m <- glmer(incidence ~ period + (1 | herd), family = poisson, data = cbpp)
# dmat <- cbind(cbpp$incidence,model.matrix(m)[,2:4],cbpp$herd)
# colnames(dmat) <- c("incidence","period2","period3","period4","herd")
# write.csv(dmat,"cbpp_R_mm.csv", row.names=FALSE, quote=FALSE)
cbpp = pd.read_csv('cbpp_R_mm.csv')

fam = sm.families.Poisson()
m = sm.MixedGLM.from_formula("incidence ~ 1+period2+period3+period4", cbpp, groups=cbpp['herd'],family=fam)

mf = m.fit()

print(mf.summary())
print(mf.llf)
print(mf.params)

