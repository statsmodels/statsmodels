

import numpy as np
from scikits.statsmodels.sandbox import tsa
from maketests_mlabwrap import HoldIt

filen = 'savedrvs.py'
np.set_printoptions(precision=14, linewidth=100)


# check arma to return same as random.normal
np.random.seed(10000)
xo = tsa.arma_generate_sample([1], [1], nsample=100)
xo2 = np.round(xo*1000).astype(int)
np.random.seed(10000)
rvs = np.random.normal(size=100)
rvs2 = np.round(xo*1000).astype(int)
assert (xo2==rvs2).all()

nsample = 1000
data =  HoldIt('rvsdata')

np.random.seed(10000)
xo = tsa.arma_generate_sample([1, -0.8, 0.5], [1], nsample=nsample)
data.xar2 = np.round(xo*1000).astype(int)
np.random.seed(10000)
xo = np.random.normal(size=nsample)
data.xnormal = np.round(xo*1000).astype(int)
np.random.seed(10000)
xo = tsa.arma_generate_sample([1, -0.8, 0.5, -0.3], [1, 0.3, 0.2], nsample=nsample)
data.xarma32 = np.round(xo*1000).astype(int)

data.save(filename=filen, comment='generated data, divide by 1000, see savervs')
