import statsmodels.regression.linear_model as lm
import statsmodels.discrete.discrete_model as dm
import statsmodels.regression.mixed_linear_model as mlm
import statsmodels.genmod.generalized_linear_model as glm
import statsmodels.robust.robust_linear_model as roblm
import statsmodels.regression.quantile_regression as qr
import statsmodels.duration.hazard_regression as hr
import statsmodels.genmod.generalized_estimating_equations as gee

gls = lm.GLS.from_formula
wls = lm.WLS.from_formula
ols = lm.OLS.from_formula
glsar = lm.GLSAR.from_formula
mixedlm = mlm.MixedLM.from_formula
glm = glm.GLM.from_formula
rlm = roblm.RLM.from_formula
mnlogit = dm.MNLogit.from_formula
logit = dm.Logit.from_formula
probit = dm.Probit.from_formula
poisson = dm.Poisson.from_formula
negativebinomial = dm.NegativeBinomial.from_formula
quantreg = qr.QuantReg.from_formula
phreg = hr.PHReg.from_formula
ordinal_gee = gee.OrdinalGEE.from_formula
nominal_gee = gee.NominalGEE.from_formula
gee = gee.GEE.from_formula

del lm, dm, mlm, glm, roblm, qr, hr
