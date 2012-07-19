import numpy as np
import statsmodels.api as sm
from statsmodels.sysreg.syssem import *

data = sm.datasets.kmenta.load().data

y = data['consump']
x1 = np.column_stack((data['price'], data['income']))
x2 = np.column_stack((data['price'], data['farmPrice'], data['trend']))
x1 = sm.add_constant(x1, prepend=True)
x2 = sm.add_constant(x2, prepend=True)

eq1 = {'endog' : y, 'exog' : x1, 'indep_endog' : [1]}
eq2 = {'endog' : y, 'exog' : x2, 'indep_endog' : [1]}
sys = [eq1, eq2]

mod = Sys2SLS(sys)
print mod.fit() # parameters estimates, same as systemfit

