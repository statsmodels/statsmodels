import numpy as np
import statsmodels.api as sm
from statsmodels.sysreg.syssem import *

# See sysreg/tests/results/kmenta.R

data = sm.datasets.kmenta.load().data

y = data['consump']
x1 = np.column_stack((data['price'], data['income']))
x2 = np.column_stack((data['price'], data['farmPrice'], data['trend']))
x1 = sm.add_constant(x1, prepend=True)
x2 = sm.add_constant(x2, prepend=True)

eq1 = {'endog' : y, 'exog' : x1, 'indep_endog' : [1]}
eq2 = {'endog' : y, 'exog' : x2, 'indep_endog' : [1]}
sys = [eq1, eq2]

mod2sls = Sys2SLS(sys)
res2sls = mod2sls.fit()
print res2sls.summary(yname=['consump1', 'consump2'], xname=['const', 'price',
    'income', 'const', 'price', 'farmPrice', 'trend'])

mod3sls = Sys3SLS(sys)
res3sls = mod3sls.fit(igls=True)
#print res3sls.summary()

