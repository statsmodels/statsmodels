# this will trigger an InfeasibleTestError exception in 9 lags
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np

X = [1,2,3,4,5,6,7,8,9,10] * 100
X = [float(x) for x in X]
Y = [x+1 for x in X]

both = [(a,b) for a,b in zip(X,Y)]

grangercausalitytests(both, maxlag=9)
