from statsmodels.regression.linear_model import OLS, WLS, GLS
from statsmodels.genmod.generalized_linear_model import GLM

ols = OLS.from_formula
wls = WLS.from_formula
gls = WLS.from_formula
glm = GLM.from_formula
