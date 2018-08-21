from statsmodels.regression.linear_model import (GLS,
                                                 WLS,
                                                 OLS,
                                                 GLSAR)
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.discrete.discrete_model import (MNLogit,
                                                 Logit,
                                                 Probit,
                                                 Poisson,
                                                 NegativeBinomial)
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.genmod.generalized_estimating_equations import (GEE,
                                                                 OrdinalGEE,
                                                                 NominalGEE)

GLS = GLS.from_formula
WLS = WLS.from_formula
OLS = OLS.from_formula
GLSAR = GLSAR.from_formula
MixedML = MixedLM.from_formula
GLM = GLM.from_formula
RLM = RLM.from_formula
MNLogit = MNLogit.from_formula
Logit = Logit.from_formula
Probit = Probit.from_formula
Poisson = Poisson.from_formula
NegativeBinomial = NegativeBinomial.from_formula
QuantReg = QuantReg.from_formula
PHReg = PHReg.from_formula
GEE = GEE.from_formula
OrdinalGEE = OrdinalGEE.from_formula
NominalGEE = NominalGEE.from_formula
