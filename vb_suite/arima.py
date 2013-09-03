from vbench.api import Benchmark
from datetime import datetime

common_setup = """from sm_vb_common import *
"""

#-----------------------------------------------------------------------------
# basic arima fit

setup = common_setup + """

from statsmodels.tsa.arima_process import arma_generate_sample
np.random.seed(12345)
arparams = np.array([.75, -.25])
maparams = np.array([.65, .35])
arparams = np.r_[1, -arparams]
maparam = np.r_[1, maparams]
nobs = 500
y = arma_generate_sample(arparams, maparams, nobs)
import pandas
dates = sm.tsa.datetools.dates_from_range('1980m1', length=nobs)
y = pandas.TimeSeries(y, index=dates)
arma_mod = sm.tsa.ARMA(y, order=(2, 2))
"""

stmt1 = "arma_res = arma_mod.fit(trend='nc', disp=-1)"
arima_basic = Benchmark(stmt1, setup, start_date=datetime(2013, 6, 1))
