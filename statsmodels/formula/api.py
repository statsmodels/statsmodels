from statsmodels.regression.linear_model import GLS
from statsmodels.regression.linear_model import WLS
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.linear_model import GLSAR
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.discrete.discrete_model import MNLogit
from statsmodels.discrete.discrete_model import Logit
from statsmodels.discrete.discrete_model import Probit
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.genmod.generalized_estimating_equations import (GEE,
     OrdinalGEE, NominalGEE)

gls = GLS.from_formula
wls = WLS.from_formula
ols = OLS.from_formula
glsar = GLSAR.from_formula
mixedlm = MixedLM.from_formula
glm = GLM.from_formula
rlm = RLM.from_formula
mnlogit = MNLogit.from_formula
logit = Logit.from_formula
probit = Probit.from_formula
poisson = Poisson.from_formula
negativebinomial = NegativeBinomial.from_formula
quantreg = QuantReg.from_formula
phreg = PHReg.from_formula
gee = GEE.from_formula
ordinal_gee = OrdinalGEE.from_formula
nominal_gee = NominalGEE.from_formula

del GLS, WLS, OLS, GLSAR, MixedLM, GLM, \
RLM, MNLogit, Logit, Probit, Poisson, \
NegativeBinomial, QuantReg, PHReg, \
GEE, OrdinalGEE, NominalGEE
