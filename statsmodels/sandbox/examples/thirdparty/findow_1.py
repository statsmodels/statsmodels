"""A quick look at volatility of stock returns for 2009

Just an exercise to find my way around the pandas methods.
Shows the daily rate of return, the square of it (volatility) and
a 5 day moving average of the volatility.
No guarantee for correctness.
Assumes no missing values.
colors of lines in graphs are not great

uses DataFrame and WidePanel to hold data downloaded from yahoo using matplotlib.
I have not figured out storage, so the download happens at each run
of the script.

Created on Sat Jan 30 16:30:18 2010
Author: josef-pktd
"""

from statsmodels.compat.python import lzip

import datetime as dt
from pathlib import Path

import matplotlib.finance as fin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def getquotes(symbol, start, end):
    quotes = fin.quotes_historical_yahoo(symbol, start, end)
    dates, open, close, high, low, volume = lzip(*quotes)
    data = {"open": open, "close": close, "high": high, "low": low, "volume": volume}
    dates = pd.Index([dt.datetime.fromordinal(int(d)) for d in dates])
    return pd.DataFrame(data, index=dates)


start_date = dt.datetime(2007, 1, 1)
end_date = dt.datetime(2009, 12, 31)
dj30 = [
    "MMM",
    "AA",
    "AXP",
    "T",
    "BAC",
    "BA",
    "CAT",
    "CVX",
    "CSCO",
    "KO",
    "DD",
    "XOM",
    "GE",
    "HPQ",
    "HD",
    "INTC",
    "IBM",
    "JNJ",
    "JPM",
    "KFT",
    "MCD",
    "MRK",
    "MSFT",
    "PFE",
    "PG",
    "TRV",
    "UTX",
    "VZ",
    "WMT",
    "DIS",
]
mysym = ["msft", "ibm", "goog"]
indexsym = ["gspc", "dji"]
dmall = {}
for sy in dj30:
    dmall[sy] = getquotes(sy, start_date, end_date)
pawp = pd.WidePanel.fromDict(dmall)
print(pawp.values.shape)
paclose = pawp.getMinorXS("close")
paclose_ratereturn = paclose.apply(np.log).diff()
if not Path("dj30rr").exists():
    paclose_ratereturn.save("dj30rr")
plt.figure()
paclose_ratereturn.plot()
plt.title("daily rate of return")
paclose_ratereturn_vol = paclose_ratereturn.apply(lambda x: np.power(x, 2))
plt.figure()
plt.title("volatility (with 5 day moving average")
paclose_ratereturn_vol.plot()
paclose_ratereturn_vol_mov = paclose_ratereturn_vol.apply(
    lambda x: np.convolve(x, np.ones(5) / 5.0, "same")
)
paclose_ratereturn_vol_mov.plot()
