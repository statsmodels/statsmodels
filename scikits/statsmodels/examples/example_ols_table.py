"""Example: scikits.statsmodels.OLS
"""

from scikits.statsmodels.datasets.longley import Load
import scikits.statsmodels as sm
from scikits.statsmodels.iolib.table import (SimpleTable, default_txt_fmt,
                        default_latex_fmt, default_html_fmt)
import numpy as np

data = Load()
data.exog = sm.tools.add_constant(data.exog)

ols_model = sm.OLS(data.endog, data.exog)
ols_results = ols_model.fit()

# the Longley dataset is well known to have high multicollinearity
# one way to find the condition number is as follows


#Find OLS parameters for model with one explanatory variable dropped

resparams = np.nan * np.ones((7,7))
res =  sm.OLS(data.endog, data.exog).fit()
resparams[:,0] = res.params

indall = range(7)
for i in range(6):
    ind = indall[:]
    del ind[i]
    res =  sm.OLS(data.endog, data.exog[:,ind]).fit()
    resparams[ind,i+1] = res.params

txt_fmt1 = default_txt_fmt
numformat = '%10.4f'
txt_fmt1 = dict(data_fmts = [numformat])
rowstubs = data._names[2:] + ['const']
headers = ['all'] + ['drop %s' % name for name in data._names[2:]]
tabl = SimpleTable(resparams, headers, rowstubs, txt_fmt=txt_fmt1)

nanstring = numformat%np.nan
nn = len(nanstring)
nanrep = ' '*(nn-1)
nanrep = nanrep[:nn//2] + '-' + nanrep[nn//2:]

print 'Longley data - sensitivity to dropping an explanatory variable'
#print tabl
print str(tabl).replace(nanstring, nanrep)



