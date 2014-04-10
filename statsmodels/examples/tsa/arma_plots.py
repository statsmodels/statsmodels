'''Plot acf and pacf for some ARMA(1,1)

'''


from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.arima_process as tsp
from statsmodels.sandbox.tsa.fftarma import ArmaFft as FftArmaProcess
import statsmodels.tsa.stattools as tss
from statsmodels.graphics.tsaplots import plotacf

np.set_printoptions(precision=2)


arcoefs = [0.9, 0., -0.5] #[0.9, 0.5, 0.1, 0., -0.5]
macoefs = [0.9, 0., -0.5] #[0.9, 0.5, 0.1, 0., -0.5]
nsample = 1000
nburnin = 1000
sig = 1

fig = plt.figure(figsize=(8, 13))
fig.suptitle('ARMA: Autocorrelation (left) and Partial Autocorrelation (right)')
subplotcount = 1
nrows = 4
for arcoef in arcoefs[:-1]:
    for macoef in macoefs[:-1]:
        ar = np.r_[1., -arcoef]
        ma = np.r_[1.,  macoef]

        #y = tsp.arma_generate_sample(ar,ma,nsample, sig, burnin)
        #armaprocess = FftArmaProcess(ar, ma, nsample) #TODO: make n optional
        #armaprocess.plot4()
        armaprocess = tsp.ArmaProcess(ar, ma)
        acf = armaprocess.acf(20)[:20]
        pacf = armaprocess.pacf(20)[:20]
        ax = fig.add_subplot(nrows, 2, subplotcount)
        plotacf(acf, ax=ax)
##        ax.set_title('Autocorrelation \nar=%s, ma=%rs' % (ar, ma),
##                     size='xx-small')
        ax.text(0.7, 0.6, 'ar =%s \nma=%s' % (ar, ma),
                transform=ax.transAxes,
                horizontalalignment='left', #'right',
                size='xx-small')
        ax.set_xlim(-1,20)
        subplotcount +=1
        ax = fig.add_subplot(nrows, 2, subplotcount)
        plotacf(pacf, ax=ax)
##        ax.set_title('Partial Autocorrelation \nar=%s, ma=%rs' % (ar, ma),
##                     size='xx-small')
        ax.text(0.7, 0.6, 'ar =%s \nma=%s' % (ar, ma),
                transform=ax.transAxes,
                horizontalalignment='left', #'right',
                size='xx-small')
        ax.set_xlim(-1,20)
        subplotcount +=1

axs = fig.axes
### turn of the 2nd column y tick labels
##for ax in axs[1::2]:#[:,1].flat:
##   for label in ax.get_yticklabels(): label.set_visible(False)

# turn off all but the bottom xtick labels
for ax in axs[:-2]:#[:-1,:].flat:
   for label in ax.get_xticklabels(): label.set_visible(False)


# use a MaxNLocator on the first column y axis if you have a bunch of
# rows to avoid bunching; example below uses at most 3 ticks
import matplotlib.ticker as mticker
for ax in axs: #[::2]:#[:,1].flat:
   ax.yaxis.set_major_locator( mticker.MaxNLocator(3 ))



plt.show()
