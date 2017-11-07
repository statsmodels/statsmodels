from statsmodels.compat.testing import skipif

import numpy as np

from statsmodels.graphics.bland_altman import bland_altman_plot


try:
    import matplotlib.pyplot as plt
    have_matplotlib = True
except:
    have_matplotlib = False


@skipif(not have_matplotlib, reason='matplotlib not available')
def test_bland_altman():

    # Seed the random number generator.
    # This ensures that the results below are reproducible.
    np.random.seed(11111)
    m1 = np.random.random(20)
    m2 = np.random.random(20)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    bland_altman_plot(m1, m2)
    plt.close(fig)

    fig, ax = plt.subplots(2)
    bland_altman_plot(m1, m2, ax = ax[0])
    plt.close(fig)

    fig, ax = plt.subplots(1)
    bland_altman_plot(m1, m2, sd_limit = 0)
    plt.close(fig)

    fig, ax = plt.subplots(1)
    bland_altman_plot(m1, m2, scatter_kwds={'color':'green', 's':10})
    plt.close(fig)

    fig, ax = plt.subplots(1)
    bland_altman_plot(m1, m2, mean_line_kwds={'color':'green','lw':5})
    plt.close(fig)

    fig, ax = plt.subplots(1)
    bland_altman_plot(m1, m2, limit_lines_kwds={'color':'green',
                                                'lw':5,
                                                'ls':'dotted'})
    plt.close(fig)
