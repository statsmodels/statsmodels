import numpy as np
import statsmodels.api as sm
from statsmodels.sysreg.syssem import *

# This follows the simple macroeconomic model given in
# Greene Example 15.1 (5th Edition)
# The data however is from statsmodels and is not the same as
# Greene's

# The model is
# consumption: c_{t} = \alpha_{0} + \alpha_{1}y_{t} + \alpha_{2}c_{t-1} + \epsilon_{t1}
# investment: i_{t} = \beta_{0} + \beta_{1}r_{t} + \beta_{2}\left(y_{t}-y_{t-1}\right) + \epsilon_{t2}
# demand: y_{t} = c_{t} + I_{t} + g_{t}

# See Greene's Econometric Analysis for more information

macrodata = sm.datasets.macrodata.load().data

y = macrodata['realcons'] + macrodata['realinv'] + macrodata['realgovt']

y1 = macrodata['realcons'][1:] # leave off first date
y2 = macrodata['realinv'][1:] # idem

x1 = np.column_stack((y[1:], macrodata['realcons'][:-1]))
x1 = sm.add_constant(x1, prepend=True)
x2 = np.column_stack((macrodata['tbilrate'][1:], np.diff(y)))
x2 = sm.add_constant(x2, prepend=True)

eq1 = {'endog' : y1, 'exog' : x1, 'indep_endog' : [1]}
# np.diff(y) is partially endogenous and should not be included in instruments
eq2 = {'endog' : y2, 'exog' : x2, 'indep_endog' : [2]}
sys = [eq1, eq2]

# y_{t-1} needs to be specified as additional instrument
instruments = np.column_stack((macrodata['realgovt'][1:], y[:-1]))
mod = Sys2SLS(sys, instruments=instruments)
print mod.fit() # identical to those in systemfit. See sandbox/test/macrodata.s

