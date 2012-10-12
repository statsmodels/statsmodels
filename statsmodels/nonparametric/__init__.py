from kde import KDE
#from lowess import lowess #don't do that, shadowing the module
import bandwidths
from nonparametric2 import KDE, ConditionalKDE, Reg, CensoredReg

from statsmodels import NoseWrapper as Tester
test = Tester().test
