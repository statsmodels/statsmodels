# -*- coding: utf-8 -*-
"""A quick look at volatility of stock returns for 2009

Just an exercise to find my way around the pandas methods.
Shows the daily rate of return, the square of it (volatility) and
a 5 day moving average of the volatility.
No guarantee for correctness.
Assumes no missing values.
colors of lines in graphs are not great

uses DataFrame and WidePanel to hold data downloaded from yahoo using matplotlib.
I haven't figured out storage, so the download happens at each run
of the script.

getquotes is from pandas\examples\finance.py

Created on Sat Jan 30 16:30:18 2010
Author: josef-pktd
"""
from statsmodels.compat.python import lzip
import numpy as np
import matplotlib.finance as fin
import matplotlib.pyplot as plt
import datetime as dt

import pandas as pa


def getquotes(symbol, start, end):
    quotes = fin.quotes_historical_yahoo(symbol, start, end)
    dates, open, close, high, low, volume = lzip(*quotes)

    data = {
        'open' : open,
        'close' : close,
        'high' : high,
        'low' : low,
        'volume' : volume
    }

    dates = pa.Index([dt.datetime.fromordinal(int(d)) for d in dates])
    return pa.DataFrame(data, index=dates)


start_date = dt.datetime(2007, 1, 1)
end_date = dt.datetime(2009, 12, 31)

dj30 = ['MMM', 'AA', 'AXP', 'T', 'BAC', 'BA', 'CAT', 'CVX', 'CSCO',
       'KO', 'DD', 'XOM', 'GE', 'HPQ', 'HD', 'INTC', 'IBM', 'JNJ',
       'JPM', 'KFT', 'MCD', 'MRK', 'MSFT', 'PFE', 'PG', 'TRV',
       'UTX', 'VZ', 'WMT', 'DIS']
mysym = ['msft', 'ibm', 'goog']
indexsym = ['gspc', 'dji']


# download data
dmall = {}
for sy in dj30:
   dmall[sy]  = getquotes(sy, start_date, end_date)

# combine into WidePanel
pawp = pa.WidePanel.fromDict(dmall)
print(pawp.values.shape)

# select closing prices
paclose = pawp.getMinorXS('close')

# take log and first difference over time
paclose_ratereturn = paclose.apply(np.log).diff()

import os
if not os.path.exists('dj30rr'):
    #if pandas is updated, then sometimes unpickling fails, and need to save again
    paclose_ratereturn.save('dj30rr')

plt.figure()
paclose_ratereturn.plot()
plt.title('daily rate of return')

# square the returns
paclose_ratereturn_vol = paclose_ratereturn.apply(lambda x:np.power(x,2))
plt.figure()
plt.title('volatility (with 5 day moving average')
paclose_ratereturn_vol.plot()

# use convolution to get moving average
paclose_ratereturn_vol_mov = paclose_ratereturn_vol.apply(
                        lambda x:np.convolve(x,np.ones(5)/5.,'same'))
paclose_ratereturn_vol_mov.plot()



#plt.show()
