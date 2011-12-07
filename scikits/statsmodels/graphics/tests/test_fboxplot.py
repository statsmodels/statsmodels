import numpy as np
from numpy.testing import assert_allclose

from scikits.statsmodels.graphics.fboxplot import banddepth


def test_banddepth():
    xx = np.arange(500) / 150.
    y1 = 1 + 0.5 * np.sin(xx)
    y2 = 0.3 + np.sin(xx + np.pi/6)
    y3 = -0.5 + np.sin(xx + np.pi/6)
    y4 = -1 + 0.3 * np.cos(xx + np.pi/6)

    data = np.asarray([y1, y2, y3, y4])
    depth = banddepth(data, method='BD2')
    expected_depth = [0.5, 5./6, 5./6, 0.5]
    assert_allclose(depth, expected_depth)


## Plot to visualize why we expect this output
#import matplotlib.pyplot as plt
#fig = plt.figure()
#ax = fig.add_subplot(111)
#for ii, yy in enumerate([y1, y2, y3, y4]):
#    ax.plot(xx, yy, label="y%s" % ii)

#ax.legend()

#plt.show()
