# -*- coding: utf-8 -*-
"""Playing with correlation of DJ-30 stock returns

this uses pickled data that needs to be created with findow.py
to see graphs, uncomment plt.show()


Created on Sat Jan 30 16:30:18 2010
Author: josef-pktd
"""

import numpy as np
import matplotlib.finance as fin
import matplotlib.pyplot as plt
import datetime as dt

import pandas as pa
import pickle

import scikits.statsmodels.api as sm
import scikits.statsmodels.sandbox as sb
import scikits.statsmodels.sandbox.tools as sbtools

try:
    rrdm = pickle.load(file('dj30rr','rb'))
except Exception: #blanket for any unpickling error
    print "Error with unpickling, a new pickle file can be created with findow_1"
    raise

ticksym = rrdm.columns.tolist()
rr = rrdm.values[1:400]

rrcorr = np.corrcoef(rr, rowvar=0)



plot_corr(rrcorr, xnames=ticksym)
nvars = rrcorr.shape[0]
plt.figure()
plt.hist(rrcorr[np.triu_indices(nvars,1)])
plt.title('Correlation Coefficients')

xreda, facta, evaa, evea  = sbtools.pcasvd(rr)
evallcs = (evaa).cumsum()
print evallcs/evallcs[-1]
xred, fact, eva, eve  = sbtools.pcasvd(rr, keepdim=4)
pcacorr = np.corrcoef(xred, rowvar=0)

resid = rr-xred
residcorr = np.corrcoef(resid, rowvar=0)
plot_corr(residcorr, xnames=ticksym, title='Correlation Residuals')

plt.matshow(residcorr)
plt.imshow(residcorr, cmap=plt.cm.jet, interpolation='nearest',
           extent=(0,30,0,30), vmin=-1.0, vmax=1.0)
plt.colorbar()
#plt.show()
#plt.close('all')

