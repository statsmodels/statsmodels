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
from statsmodels.compat.python import cPickle

import statsmodels.api as sm
import statsmodels.sandbox as sb
import statsmodels.sandbox.tools as sbtools

from statsmodels.graphics.correlation import plot_corr, plot_corr_grid

try:
    rrdm = cPickle.load(file('dj30rr','rb'))
except Exception: #blanket for any unpickling error
    print("Error with unpickling, a new pickle file can be created with findow_1")
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
print(evallcs/evallcs[-1])
xred, fact, eva, eve  = sbtools.pcasvd(rr, keepdim=4)
pcacorr = np.corrcoef(xred, rowvar=0)

plot_corr(pcacorr, xnames=ticksym, title='Correlation PCA')

resid = rr-xred
residcorr = np.corrcoef(resid, rowvar=0)
plot_corr(residcorr, xnames=ticksym, title='Correlation Residuals')

plt.matshow(residcorr)
plt.imshow(residcorr, cmap=plt.cm.jet, interpolation='nearest',
           extent=(0,30,0,30), vmin=-1.0, vmax=1.0)
plt.colorbar()

normcolor = (0,1) #False #True
fig = plt.figure()
ax = fig.add_subplot(2,2,1)
plot_corr(rrcorr, xnames=ticksym, normcolor=normcolor, ax=ax)
ax2 = fig.add_subplot(2,2,3)
#pcacorr = np.corrcoef(xred, rowvar=0)
plot_corr(pcacorr, xnames=ticksym, title='Correlation PCA',
          normcolor=normcolor, ax=ax2)
ax3 = fig.add_subplot(2,2,4)
plot_corr(residcorr, xnames=ticksym, title='Correlation Residuals',
          normcolor=normcolor, ax=ax3)

import matplotlib as mpl
images = [c for ax in fig.axes for c in ax.get_children() if isinstance(c, mpl.image.AxesImage)]
print(images)
print(ax.get_children())
#cax = fig.add_subplot(2,2,2)
#[0.85, 0.1, 0.075, 0.8]
fig. subplots_adjust(bottom=0.1, right=0.9, top=0.9)
cax = fig.add_axes([0.9, 0.1, 0.025, 0.8])
fig.colorbar(images[0], cax=cax)
fig.savefig('corrmatrixgrid.png', dpi=120)

has_sklearn = True
try:
    import sklearn
except ImportError:
    has_sklearn = False
    print('sklearn not available')


def cov2corr(cov):
    std_ = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_, std_)
    return corr

if has_sklearn:
    from sklearn.covariance import LedoitWolf, OAS, MCD

    lw = LedoitWolf(store_precision=False)
    lw.fit(rr, assume_centered=False)
    cov_lw = lw.covariance_
    corr_lw = cov2corr(cov_lw)

    oas = OAS(store_precision=False)
    oas.fit(rr, assume_centered=False)
    cov_oas = oas.covariance_
    corr_oas = cov2corr(cov_oas)

    mcd = MCD()#.fit(rr, reweight=None)
    mcd.fit(rr, assume_centered=False)
    cov_mcd = mcd.covariance_
    corr_mcd = cov2corr(cov_mcd)

    titles = ['raw correlation', 'lw', 'oas', 'mcd']
    normcolor = None
    fig = plt.figure()
    for i, c in enumerate([rrcorr, corr_lw, corr_oas, corr_mcd]):
    #for i, c in enumerate([np.cov(rr, rowvar=0), cov_lw, cov_oas, cov_mcd]):
        ax = fig.add_subplot(2,2,i+1)
        plot_corr(c, xnames=None, title=titles[i],
              normcolor=normcolor, ax=ax)

    images = [c for ax in fig.axes for c in ax.get_children() if isinstance(c, mpl.image.AxesImage)]
    fig. subplots_adjust(bottom=0.1, right=0.9, top=0.9)
    cax = fig.add_axes([0.9, 0.1, 0.025, 0.8])
    fig.colorbar(images[0], cax=cax)

    corrli = [rrcorr, corr_lw, corr_oas, corr_mcd, pcacorr]
    diffssq = np.array([[((ci-cj)**2).sum() for ci in corrli]
                            for cj in corrli])
    diffsabs = np.array([[np.max(np.abs(ci-cj)) for ci in corrli]
                            for cj in corrli])
    print(diffssq)
    print('\nmaxabs')
    print(diffsabs)
    fig.savefig('corrmatrix_sklearn.png', dpi=120)

    fig2 = plot_corr_grid(corrli+[residcorr], ncols=3,
                          titles=titles+['pca', 'pca-residual'],
                          xnames=[], ynames=[])
    fig2.savefig('corrmatrix_sklearn_2.png', dpi=120)

#plt.show()
#plt.close('all')

