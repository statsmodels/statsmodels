import numpy as np
import pandas as pd
import pytest

from statsmodels.graphics.agreement import mean_diff_plot


@pytest.mark.thread_unsafe(reason="Uses matplotlib")
@pytest.mark.matplotlib
def test_mean_diff_plot(close_figures):
    import matplotlib.pyplot as plt

    # Seed the random number generator.
    # This ensures that the results below are reproducible.
    rs = np.random.RandomState(11111)
    m1 = rs.random(20)
    m2 = rs.random(20)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # basic test.
    mean_diff_plot(m1, m2, ax=ax)

    # Test with pandas Series.
    p1 = pd.Series(m1)
    p2 = pd.Series(m2)
    mean_diff_plot(p1, p2)

    # Test plotting on assigned axis.
    fig, ax = plt.subplots(2)
    mean_diff_plot(m1, m2, ax=ax[0])

    # Test the setting of confidence intervals.
    mean_diff_plot(m1, m2, sd_limit=0)

    # Test asethetic controls.
    mean_diff_plot(m1, m2, scatter_kwds={"color": "green", "s": 10})

    mean_diff_plot(m1, m2, mean_line_kwds={"color": "green", "lw": 5})

    mean_diff_plot(m1, m2, limit_lines_kwds={"color": "green", "lw": 5, "ls": "dotted"})


@pytest.mark.thread_unsafe(reason="Uses matplotlib")
@pytest.mark.matplotlib
def test_mean_diff_plot_linestyles(close_figures):
    import matplotlib.pyplot as plt

    rs = np.random.RandomState(11111)
    m1 = rs.random(20)
    m2 = rs.random(20)

    # The mean line defaults to dashed and the limit lines to dotted
    fig, ax = plt.subplots()
    mean_diff_plot(m1, m2, ax=ax)
    mean_line, lower, upper = ax.lines
    assert mean_line.get_linestyle() == "--"
    assert lower.get_linestyle() == ":"
    assert upper.get_linestyle() == ":"

    # An explicit linestyle is used for the line it was supplied for
    fig, ax = plt.subplots()
    mean_diff_plot(
        m1,
        m2,
        ax=ax,
        mean_line_kwds={"linestyle": "-."},
        limit_lines_kwds={"linestyle": "-"},
    )
    mean_line, lower, upper = ax.lines
    assert mean_line.get_linestyle() == "-."
    assert lower.get_linestyle() == "-"
    assert upper.get_linestyle() == "-"
