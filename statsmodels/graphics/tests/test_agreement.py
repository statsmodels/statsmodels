from statsmodels.compat.testing import skipif

import numpy as np
import pandas as pd
from statsmodels.graphics.agreement import mean_diff_plot


try:
    import matplotlib.pyplot as plt
    have_matplotlib = True
except:
    have_matplotlib = False


@skipif(not have_matplotlib, reason='matplotlib not available')
def test_mean_diff_plot():

    # Seed the random number generator.
    # This ensures that the results below are reproducible.
    np.random.seed(11111)
    m1 = np.random.random(20)
    m2 = np.random.random(20)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # basic test.
    fig = mean_diff_plot(m1, m2, ax=ax)
    plt.close(fig)

    # Test with pandas Series.
    p1 = pd.Series(m1)
    p2 = pd.Series(m2)
    fig = mean_diff_plot(p1, p2)
    plt.close(fig)

    # Test plotting on assigned axis.
    fig, ax = plt.subplots(2)
    mean_diff_plot(m1, m2, ax = ax[0])
    plt.close(fig)

    # Test the setting of confidence intervals.
    fig = mean_diff_plot(m1, m2, sd_limit = 0)
    plt.close(fig)

    # Test asethetic controls.
    fig = mean_diff_plot(m1, m2, scatter_kwds={'color':'green', 's':10})
    plt.close(fig)

    fig = mean_diff_plot(m1, m2, mean_line_kwds={'color':'green', 'lw':5})
    plt.close(fig)

    fig = mean_diff_plot(m1, m2, limit_lines_kwds={'color':'green',
                                                'lw':5,
                                                'ls':'dotted'})
    plt.close(fig)
