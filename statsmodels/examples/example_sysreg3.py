import numpy as np
#import scipy as sp
import statsmodels.api as sm
from statsmodels.sysreg.syssem import *

data = sm.datasets.kmenta.load().data

y = data['consump']
x1 = np.column_stack((data['price'], data['income']))
x2 = np.column_stack((data['price'], data['farmPrice'], data['trend']))
x1 = sm.add_constant(x1, prepend=True)
x2 = sm.add_constant(x2, prepend=True)
#inst = np.column_stack((data['income'], data['farmPrice'], data['trend']))
#inst = sm.add_constant(inst, prepend=True)

eq1 = {'endog' : y, 'exog' : x1}
eq2 = {'endog' : y, 'exog' : x2}
sys = [eq1, eq2]

mod = SysSEM(sys)
