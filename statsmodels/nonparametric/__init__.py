from kde import KDE
#from lowess import lowess #don't do that, shadowing the module
import bandwidths
from kernel_density import KDEMultivariate, \
    KDEMultivariateConditional, EstimatorSettings
from kernel_regression import Reg, CensoredReg

from statsmodels import NoseWrapper as Tester
test = Tester().test
