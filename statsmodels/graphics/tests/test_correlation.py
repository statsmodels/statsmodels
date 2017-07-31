from statsmodels.compat.testing import skipif
import numpy as np

from statsmodels.graphics.correlation import plot_corr, plot_corr_grid
from statsmodels.datasets import randhie


try:
    import matplotlib.pyplot as plt
    have_matplotlib = True
except:
    have_matplotlib = False


@skipif(not have_matplotlib, reason='matplotlib not available')
def test_plot_corr():
    hie_data = randhie.load_pandas()
    corr_matrix = np.corrcoef(hie_data.data.values.T)

    fig = plot_corr(corr_matrix, xnames=hie_data.names)
    plt.close(fig)

    fig = plot_corr(corr_matrix, xnames=[], ynames=hie_data.names)
    plt.close(fig)

    fig = plot_corr(corr_matrix, normcolor=True, title='', cmap='jet')
    plt.close(fig)


@skipif(not have_matplotlib, reason='matplotlib not available')
def test_plot_corr_grid():
    hie_data = randhie.load_pandas()
    corr_matrix = np.corrcoef(hie_data.data.values.T)

    fig = plot_corr_grid([corr_matrix] * 2, xnames=hie_data.names)
    plt.close(fig)

    fig = plot_corr_grid([corr_matrix] * 5, xnames=[], ynames=hie_data.names)
    plt.close(fig)

    fig = plot_corr_grid([corr_matrix] * 3, normcolor=True, titles='', cmap='jet')
    plt.close(fig)

