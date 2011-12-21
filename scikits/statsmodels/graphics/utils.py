"""Helper functions for graphics with Matplotlib."""


__all__ = ['create_mpl_ax', 'create_mpl_fig']


def _import_mpl():
    """This function is not needed outside this utils module."""
    try:
        import matplotlib.pyplot as plt
    except:
        raise ImportError("Matplotlib is not found.")

    return plt


def create_mpl_ax(ax=None):
    """Helper function for when a single plot axis is needed.

    Parameters
    ----------
    ax : Matplotlib AxesSubplot instance, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.

    Returns
    -------
    fig : Matplotlib figure instance
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.
    ax : Matplotlib AxesSubplot instance
        The created axis if `ax` is None, otherwise the axis that was passed
        in.

    Notes
    -----
    This function imports `matplotlib.pyplot`, which should only be done to
    create (a) figure(s) with ``plt.figure``.  All other functionality exposed
    by the pyplot module can and should be imported directly from its
    Matplotlib module.

    See Also
    --------
    create_mpl_fig

    Examples
    --------
    A plotting function has a keyword ``ax=None``.  Then calls:

    >>> from scikits.statsmodels.graphics import utils
    >>> fig, ax = utils.create_mpl_ax(ax)

    """
    if ax is None:
        plt = _import_mpl()
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    return fig, ax


def create_mpl_fig(fig=None):
    """Helper function for when multiple plot axes are needed.

    Those axes should be created in the functions they are used in, with
    ``fig.add_subplot()``.

    Parameters
    ----------
    fig : Matplotlib figure instance, optional
        If given, this figure is simply returned.  Otherwise a new figure is
        created.

    Returns
    -------
    fig : Matplotlib figure instance
        If `fig` is None, the created figure.  Otherwise the input `fig` is
        returned.

    See Also
    --------
    create_mpl_ax

    """
    if fig is None:
        plt = _import_mpl()
        fig = plt.figure()

    return fig
