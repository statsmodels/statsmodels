import numpy as np
from numpy.testing import dec

import scikits.statsmodels.api as sm
from scikits.statsmodels.graphics.tsaplots import plotacf
import scikits.statsmodels.tsa.arima_process as tsp


try:
    import matplotlib.pyplot as plt
    have_matplotlib = True
except:
    have_matplotlib = False


@dec.skipif(not have_matplotlib)
def test_plotacf():
    # Just test that it runs.
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ar = np.r_[1., -0.9]
    ma = np.r_[1.,  0.9]
    armaprocess = tsp.ArmaProcess(ar, ma)
    acf = armaprocess.acf(20)[:20]
    plotacf(acf, ax=ax)

    plt.close(fig)

