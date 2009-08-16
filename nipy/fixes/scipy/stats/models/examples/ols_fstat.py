"""examples for usage of F-test on linear restrictions in OLS

linear restriction is R \beta = 0
R is (nr,nk), beta is (nk,1) (in matrix notation)
"""

import numpy as np
import numpy.testing as npt
from models.datasets.longley.data import load
from models.regression import OLS
from models import tools

data = load()
data.exog = tools.add_constant(data.exog)
res = OLS(data.endog, data.exog).fit()

# test pairwise equality of some coefficients
R = [[0,1,-1,0,0,0,0],[0, 0, 0, 0, 1, -1, 0]]
Ftest = res.Fcontrast(R)
print repr((Ftest.F, Ftest.p_val)) #use repr to get more digits
# 9.740461873303655 0.0056052885317360301

##Compare to R (after running R_lm.s in the longley folder) looks good.
##
##> library(car)
##> linear.hypothesis(m1, c("GNP = UNEMP","POP = YEAR"))
##Linear hypothesis test
##
##Hypothesis:
##GNP - UNEMP = 0
##POP - YEAR = 0
##
##Model 1: TOTEMP ~ GNPDEFL + GNP + UNEMP + ARMED + POP + YEAR
##Model 2: restricted model
##
## Res.Df      RSS Df Sum of Sq      F   Pr(>F)
##1      9   836424
##2     11  2646903 -2  -1810479 9.7405 0.005605 **

# test all variables have zero effect
R = np.eye(7)[:-1,:]
Ftest0 = res.Fcontrast(R)
print repr((Ftest0.F, Ftest0.p_val))
print '%r' % res.F
npt.assert_almost_equal(res.F, Ftest0.F, decimal=10)
# values differ in 11th decimal
