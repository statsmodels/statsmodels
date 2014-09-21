from statsmodels.compat.python import range
import numpy as np
from numpy.testing import dec, assert_equal, assert_almost_equal

from statsmodels.graphics.functional import \
            banddepth, fboxplot, rainbowplot


try:
    import matplotlib.pyplot as plt
    import matplotlib
    have_matplotlib = True
except ImportError:
    have_matplotlib = False


def test_banddepth_BD2():
    xx = np.arange(500) / 150.
    y1 = 1 + 0.5 * np.sin(xx)
    y2 = 0.3 + np.sin(xx + np.pi/6)
    y3 = -0.5 + np.sin(xx + np.pi/6)
    y4 = -1 + 0.3 * np.cos(xx + np.pi/6)

    data = np.asarray([y1, y2, y3, y4])
    depth = banddepth(data, method='BD2')
    expected_depth = [0.5, 5./6, 5./6, 0.5]
    assert_almost_equal(depth, expected_depth)

    ## Plot to visualize why we expect this output
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #for ii, yy in enumerate([y1, y2, y3, y4]):
    #    ax.plot(xx, yy, label="y%s" % ii)

    #ax.legend()
    #plt.show()


def test_banddepth_MBD():
    xx = np.arange(5001) / 5000.
    y1 = np.zeros(xx.shape)
    y2 = 2 * xx - 1
    y3 = np.ones(xx.shape) * 0.5
    y4 = np.ones(xx.shape) * -0.25

    data = np.asarray([y1, y2, y3, y4])
    depth = banddepth(data, method='MBD')
    expected_depth = [5./6, (2*(0.75-3./8)+3)/6, 3.5/6, (2*3./8+3)/6]
    assert_almost_equal(depth, expected_depth, decimal=4)


@dec.skipif(not have_matplotlib)
def test_fboxplot_rainbowplot():
    # Test fboxplot and rainbowplot together, is much faster.
    def harmfunc(t):
        """Test function, combination of a few harmonic terms."""
        # Constant, 0 with p=0.9, 1 with p=1 - for creating outliers
        ci = int(np.random.random() > 0.9)
        a1i = np.random.random() * 0.05
        a2i = np.random.random() * 0.05
        b1i = (0.15 - 0.1) * np.random.random() + 0.1
        b2i = (0.15 - 0.1) * np.random.random() + 0.1

        func = (1 - ci) * (a1i * np.sin(t) + a2i * np.cos(t)) + \
               ci * (b1i * np.sin(t) + b2i * np.cos(t))

        return func

    np.random.seed(1234567)
    # Some basic test data, Model 6 from Sun and Genton.
    t = np.linspace(0, 2 * np.pi, 250)
    data = []
    for ii in range(20):
        data.append(harmfunc(t))

    # fboxplot test
    fig = plt.figure()
    ax = fig.add_subplot(111)
    _, depth, ix_depth, ix_outliers = fboxplot(data, wfactor=2, ax=ax)

    ix_expected = np.array([13, 4, 15, 19, 8, 6, 3, 16, 9, 7, 1, 5, 2,
                            12, 17, 11, 14, 10, 0, 18])
    assert_equal(ix_depth, ix_expected)
    ix_expected2 = np.array([2, 11, 17, 18])
    assert_equal(ix_outliers, ix_expected2)

    plt.close(fig)

    # rainbowplot test (re-uses depth variable)
    xdata = np.arange(data[0].size)
    fig = rainbowplot(data, xdata=xdata, depth=depth, cmap=plt.cm.rainbow)
    plt.close(fig)
