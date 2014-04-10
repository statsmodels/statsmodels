# -*- coding: utf-8 -*-
"""groupmean, groupby in pandas, la and tabular from a scikits.timeseries

after a question on the scipy-user mailing list I tried to do
groupmeans, which in this case are duplicate dates, in the 3 packages.

I'm using the versions that I had installed, which are all based on
repository checkout, but are not fully up-to-date

some brief comments

* la.larry and pandas.DataFrame require unique labels/index so
  groups have to represented in a separate data structure
* pandas is missing GroupBy in the docs, but the docstring is helpful
* both la and pandas handle datetime objects as object arrays
* tabular requires conversion to structured dtype, but easy helper
  functions or methods are available in scikits.timeseries and tabular

* not too bad for a first try

Created on Sat Jan 30 08:33:11 2010
Author: josef-pktd
"""
from statsmodels.compat.python import lrange, zip
import numpy as np
import scikits.timeseries as ts

s = ts.time_series([1,2,3,4,5],
            dates=ts.date_array(["2001-01","2001-01",
            "2001-02","2001-03","2001-03"],freq="M"))

print('\nUsing la')
import la

dta = la.larry(s.data, label=[lrange(len(s.data))])
dat = la.larry(s.dates.tolist(), label=[lrange(len(s.data))])
s2 = ts.time_series(dta.group_mean(dat).x,dates=ts.date_array(dat.x,freq="M"))
s2u = ts.remove_duplicated_dates(s2)
print(repr(s))
print(dat)
print(repr(s2))
print(repr(s2u))

print('\nUsing pandas')
import pandas
pdta = pandas.DataFrame(s.data, np.arange(len(s.data)), [1])
pa = pdta.groupby(dict(zip(np.arange(len(s.data)),
            s.dates.tolist()))).aggregate(np.mean)
s3 = ts.time_series(pa.values.ravel(),
            dates=ts.date_array(pa.index.tolist(),freq="M"))

print(pa)
print(repr(s3))

print('\nUsing tabular')
import tabular as tb
X = tb.tabarray(array=s.torecords(), dtype=s.torecords().dtype)
tabx = X.aggregate(On=['_dates'], AggFuncDict={'_data':np.mean,'_mask':np.all})
s4 = ts.time_series(tabx['_data'],dates=ts.date_array(tabx['_dates'],freq="M"))
print(tabx)
print(repr(s4))

from finance import *  #hack to make it run as standalone
#after running pandas/examples/finance.py
larmsft = la.larry(msft.values, [msft.index.tolist(), msft.columns.tolist()])
laribm = la.larry(ibm.values, [ibm.index.tolist(), ibm.columns.tolist()])
lar1 = la.larry(np.dstack((msft.values,ibm.values)), [ibm.index.tolist(), ibm.columns.tolist(), ['msft', 'ibm']])
print(lar1.mean(0))


y = la.larry([[1.0, 2.0], [3.0, 4.0]], [['a', 'b'], ['c', 'd']])
ysr = np.empty(y.x.shape[0],dtype=([('index','S1')]+[(i,np.float) for i in y.label[1]]))
ysr['index'] = y.label[0]
for i in ysr.dtype.names[1:]:
    ysr[i] = y[y.labelindex(i, axis=1)].x
