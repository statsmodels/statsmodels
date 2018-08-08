__all__ = ["gls", "wls", "ols", "glsar", "mixedlm", "glm", "rlm",
           "mnlogit", "logit", "probit", "poisson",
           "negativebinomial", "quantreg", "phreg",
           "gee", "ordinal_gee", "nominal_gee"]

from statsmodels.regression.linear_model import GLS
gls = GLS.from_formula
from statsmodels.regression.linear_model import WLS  # noqa:E402
wls = WLS.from_formula
from statsmodels.regression.linear_model import OLS  # noqa:E402
ols = OLS.from_formula
from statsmodels.regression.linear_model import GLSAR  # noqa:E402
glsar = GLSAR.from_formula
from statsmodels.regression.mixed_linear_model import MixedLM  # noqa:E402
mixedlm = MixedLM.from_formula
from statsmodels.genmod.generalized_linear_model import GLM  # noqa:E402
glm = GLM.from_formula
from statsmodels.robust.robust_linear_model import RLM  # noqa:E402
rlm = RLM.from_formula
from statsmodels.discrete.discrete_model import MNLogit  # noqa:E402
mnlogit = MNLogit.from_formula
from statsmodels.discrete.discrete_model import Logit  # noqa:E402
logit = Logit.from_formula
from statsmodels.discrete.discrete_model import Probit  # noqa:E402
probit = Probit.from_formula
from statsmodels.discrete.discrete_model import Poisson  # noqa:E402
poisson = Poisson.from_formula
from statsmodels.discrete.discrete_model import NegativeBinomial  # noqa:E402
negativebinomial = NegativeBinomial.from_formula
from statsmodels.regression.quantile_regression import QuantReg  # noqa:E402
quantreg = QuantReg.from_formula
from statsmodels.duration.hazard_regression import PHReg  # noqa:E402
phreg = PHReg.from_formula
from statsmodels.genmod.generalized_estimating_equations import (  # noqa:E402
    GEE, OrdinalGEE, NominalGEE)
gee = GEE.from_formula
ordinal_gee = OrdinalGEE.from_formula
nominal_gee = NominalGEE.from_formula
