from statsmodels.regression.linear_model import GLS
gls = GLS.from_formula
from statsmodels.regression.linear_model import WLS
wls = WLS.from_formula
from statsmodels.regression.linear_model import OLS
ols = OLS.from_formula
from statsmodels.regression.linear_model import GLSAR
glsar = GLSAR.from_formula
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
from statsmodels.discrete.discrete_model import NBin
nbin = NBin.from_formula
from statsmodels.discrete.discrete_model import NbReg
nbreg = NbReg.from_formula
from statsmodels.tsa.ar_model import AR
ar = AR.from_formula
from statsmodels.tsa.arima_model import ARMA
arma = ARMA.from_formula
from statsmodels.tsa.arima_model import ARIMA
arima = ARIMA.from_formula
from statsmodels.tsa.vector_ar.var_model import VAR
var = VAR.from_formula
from statsmodels.tsa.vector_ar.svar_model import SVAR
svar = SVAR.from_formula
