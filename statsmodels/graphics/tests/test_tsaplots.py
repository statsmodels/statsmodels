import numpy as np
from numpy.testing import dec

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.tsa.arima_process as tsp


try:
    import matplotlib.pyplot as plt
    have_matplotlib = True
except:
    have_matplotlib = False


@dec.skipif(not have_matplotlib)
def test_plot_acf():
    # Just test that it runs.
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ar = np.r_[1., -0.9]
    ma = np.r_[1.,  0.9]
    armaprocess = tsp.ArmaProcess(ar, ma)
    acf = armaprocess.acf(20)[:20]
    plot_acf(acf, ax=ax)

    plt.close(fig)

