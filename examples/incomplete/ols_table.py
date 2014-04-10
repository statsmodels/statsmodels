"""Example: statsmodels.OLS
"""

from statsmodels.datasets.longley import load
import statsmodels.api as sm
from statsmodels.iolib.table import SimpleTable, default_txt_fmt
import numpy as np

data = load()

data_orig = (data.endog.copy(), data.exog.copy())

#.. Note: In this example using zscored/standardized variables has no effect on
#..   regression estimates. Are there no numerical problems?

rescale = 0
#0: no rescaling, 1:demean, 2:standardize, 3:standardize and transform back
rescale_ratio = data.endog.std() / data.exog.std(0)
if rescale > 0:
    # rescaling
    data.endog -= data.endog.mean()
    data.exog -= data.exog.mean(0)
if rescale > 1:
    data.endog /= data.endog.std()
    data.exog /= data.exog.std(0)

#skip because mean has been removed, but dimension is hardcoded in table
data.exog = sm.tools.add_constant(data.exog, prepend=False)


ols_model = sm.OLS(data.endog, data.exog)
ols_results = ols_model.fit()

# the Longley dataset is well known to have high multicollinearity
# one way to find the condition number is as follows


#Find OLS parameters for model with one explanatory variable dropped

resparams = np.nan * np.ones((7, 7))
res = sm.OLS(data.endog, data.exog).fit()
resparams[:, 0] = res.params

indall = range(7)
for i in range(6):
    ind = indall[:]
    del ind[i]
    res = sm.OLS(data.endog, data.exog[:, ind]).fit()
    resparams[ind, i + 1] = res.params

if rescale == 1:
    pass
if rescale == 3:
    resparams[:-1, :] *= rescale_ratio[:, None]

txt_fmt1 = default_txt_fmt
numformat = '%10.4f'
txt_fmt1 = dict(data_fmts=[numformat])
rowstubs = data.names[1:] + ['const']
headers = ['all'] + ['drop %s' % name for name in data.names[1:]]
tabl = SimpleTable(resparams, headers, rowstubs, txt_fmt=txt_fmt1)

nanstring = numformat % np.nan
nn = len(nanstring)
nanrep = ' ' * (nn - 1)
nanrep = nanrep[:nn // 2] + '-' + nanrep[nn // 2:]

print('Longley data - sensitivity to dropping an explanatory variable')
print(str(tabl).replace(nanstring, nanrep))
