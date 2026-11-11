import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import stats

from statsmodels.graphics.plot_grids import scatter_ellipse


@pytest.mark.thread_unsafe(reason="Uses matplotlib")
@pytest.mark.matplotlib
def test_scatter_ellipse_matches_covariance(close_figures):
    from matplotlib.patches import Ellipse

    rng = np.random.default_rng(0)
    nobs = 300
    x1 = rng.normal(size=nobs)
    x2 = 0.7 * x1 + rng.normal(scale=0.6, size=nobs)
    data = np.column_stack([x1, x2])

    level = 0.9
    fig = scatter_ellipse(data, level=level)
    assert len(fig.axes) == 1
    ax = fig.axes[0]

    ellipses = ax.findobj(Ellipse)
    assert len(ellipses) == 1
    ell = ellipses[0]

    # independently recompute the confidence ellipse from the data's own
    # mean/covariance: eigen-decomposition of the covariance matrix, scaled
    # by the chi-squared quantile -- the textbook definition of a confidence
    # ellipse also used internally by plot_grids._make_ellipse.
    mean = data.mean(0)
    cov = np.cov(data, rowvar=0)
    v, w = np.linalg.eigh(cov)
    u = w[0] / np.linalg.norm(w[0])
    angle = 180 * np.arctan(u[1] / u[0]) / np.pi
    width_height = 2 * np.sqrt(v * stats.chi2.ppf(level, 2))

    assert_allclose(ell.center, mean[:2])
    assert_allclose(ell.width, width_height[0])
    assert_allclose(ell.height, width_height[1])
    assert_allclose(ell.angle, 180 + angle)


@pytest.mark.thread_unsafe(reason="Uses matplotlib")
@pytest.mark.matplotlib
def test_scatter_ellipse_multiple_variables_and_levels(close_figures):
    from matplotlib.patches import Ellipse

    rng = np.random.default_rng(1)
    nobs = 100
    data = rng.normal(size=(nobs, 3))
    varnames = ["a", "b", "c"]

    fig = scatter_ellipse(
        data,
        level=[0.5, 0.9],
        varnames=varnames,
        add_titles=True,
        keep_ticks=True,
    )
    # nvars=3 -> only the (nvars - 1) * nvars / 2 = 3 lower-triangular
    # variable pairs are plotted: (b,a), (c,a), (c,b)
    assert len(fig.axes) == 3
    # subplots are added in row-major order over (i, j) with j < i, so the
    # first one is for the (b, a) pair
    assert fig.axes[0].get_title() == "b-a"

    for ax in fig.axes:
        # two confidence levels were requested, so each pair gets 2 ellipses
        ellipses = ax.findobj(Ellipse)
        assert len(ellipses) == 2
