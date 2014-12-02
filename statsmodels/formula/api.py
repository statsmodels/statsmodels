from statsmodels.regression.linear_model import GLS
gls = GLS.from_formula
from statsmodels.regression.linear_model import WLS
wls = WLS.from_formula
from statsmodels.regression.linear_model import OLS
ols = OLS.from_formula
from statsmodels.regression.linear_model import GLSAR
glsar = GLSAR.from_formula
from statsmodels.regression.mixed_linear_model import MixedLM
mixedlm = MixedLM.from_formula
from statsmodels.genmod.generalized_linear_model import GLM
glm = GLM.from_formula
from statsmodels.robust.robust_linear_model import RLM
rlm = RLM.from_formula
from statsmodels.discrete.discrete_model import MNLogit
mnlogit = MNLogit.from_formula
from statsmodels.discrete.discrete_model import Logit
logit = Logit.from_formula
from statsmodels.discrete.discrete_model import Probit
probit = Probit.from_formula
from statsmodels.discrete.discrete_model import Poisson
poisson = Poisson.from_formula
from statsmodels.discrete.discrete_model import NegativeBinomial
negativebinomial = NegativeBinomial.from_formula
from statsmodels.regression.quantile_regression import QuantReg
quantreg = QuantReg.from_formula
from statsmodels.duration.hazard_regression import PHReg
phreg = PHReg.from_formula
from statsmodels.genmod.generalized_estimating_equations import (GEE,
     OrdinalGEE, NominalGEE)
gee = GEE.from_formula
ordinal_gee = OrdinalGEE.from_formula
nominal_gee = NominalGEE.from_formula
