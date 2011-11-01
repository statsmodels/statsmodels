from kde import KDE
#from lowess import lowess #don't do that, shadowing the module
import bandwidths

from scikits.statsmodels import NoseWrapper as Tester
test = Tester().test
