"""
Descriptive Statistics for Time Series

Created on Sat Oct 30 14:24:08 2010

Author: josef-pktd
License: BSD(3clause)
"""

import warnings

from statsmodels.graphics.utils import _import_mpl

from . import stattools as stt


# TODO: check subclassing for descriptive stats classes
class TsaDescriptive:
    """
    Collection of descriptive statistical methods for time series

    Parameters
    ----------
    data : array_like
        The time series data.
    label : str, optional
        A label used to identify the series.
    name : str, optional
        A name used as the base for labeling series derived from `data`,
        such as those produced by `filter` and `detrend`.

    Attributes
    ----------
    data : array_like
        The time series data.
    label : str
        The label identifying the series.
    name : str
        The base name used for derived series.
    mod : None
        Placeholder for the model instance created by `fit`.
    res : None
        Placeholder for the results instance created by `fit`.
    """

    def __init__(self, data, label=None, name=""):
        warnings.warn(
            "TsaDescriptive is deprecated. Although documented, it has had "
            "no test coverage and no internal callers, and its behavior is "
            "not guaranteed. It will be removed after statsmodels 0.16 is "
            "released. If you rely on this class, please open an issue at "
            "https://github.com/statsmodels/statsmodels/issues.",
            FutureWarning,
            stacklevel=2,
        )
        self.data = data
        self.label = label
        self.name = name
        self.mod = None
        self.res = None

    def filter(self, num, den):
        """
        Filter the time series using a linear filter.

        Parameters
        ----------
        num : array_like
            The numerator coefficient vector of the filter.
        den : array_like
            The denominator coefficient vector of the filter.

        Returns
        -------
        TsaDescriptive
            A new instance containing the filtered data.
        """
        from scipy.signal import lfilter
        xfiltered = lfilter(num, den, self.data)
        return self.__class__(xfiltered, self.label, self.name + "_filtered")

    def detrend(self, order=1):
        """
        Detrend the time series.

        Parameters
        ----------
        order : int, optional
            The polynomial order of the trend to remove.

        Returns
        -------
        TsaDescriptive
            A new instance containing the detrended data.
        """
        from . import tsatools
        xdetrended = tsatools.detrend(self.data, order=order)
        return self.__class__(xdetrended, self.label, self.name + "_detrended")

    def fit(self, order=(1, 0, 1), **kwds):
        """
        Fit an ARMA model to the time series.

        Parameters
        ----------
        order : tuple of int, optional
            The (p, q) order, or (p, d, q) order, of the ARMA model to fit.
        **kwds
            Additional keyword arguments passed to the model's `fit` method.

        Returns
        -------
        Results
            The results instance from fitting the model.
        """
        from statsmodels.tsa.arima.model import ARIMA
        self.mod = ARIMA(self.data, order=order)
        self.res = self.mod.fit(**kwds)

        return self.res

    def acf(self, nlags=40):
        """
        Compute the autocorrelation function of the time series.

        Parameters
        ----------
        nlags : int, optional
            The number of lags to include.

        Returns
        -------
        ndarray
            The autocorrelation function.
        """
        return stt.acf(self.data, nlags=nlags)

    def pacf(self, nlags=40):
        """
        Compute the partial autocorrelation function of the time series.

        Parameters
        ----------
        nlags : int, optional
            The number of lags to include.

        Returns
        -------
        ndarray
            The partial autocorrelation function.
        """
        return stt.pacf(self.data, nlags=nlags)

    def periodogram(self):
        """
        Compute the periodogram of the time series.

        Returns
        -------
        ndarray
            The periodogram values.
        """
        # does not return frequencies
        return stt.periodogram(self.data)

    # copied from fftarma.py
    def plot4(self, fig=None, nobs=100, nacf=20, nfreq=100):
        """
        Plot the series, its ACF, PACF, and power spectrum.

        Parameters
        ----------
        fig : Figure, optional
            An existing matplotlib figure to draw the plots in. If not
            provided, a new figure is created.
        nobs : int, optional
            The number of observations to use. Not used.
        nacf : int, optional
            The number of lags to include in the ACF and PACF plots.
        nfreq : int, optional
            The number of frequencies to include in the power spectrum
            plot.

        Returns
        -------
        Figure
            The figure containing the plots.
        """
        data = self.data
        acf = self.acf(nacf)
        pacf = self.pacf(nacf)
        spdr = self.periodogram()[:nfreq]  # (w)

        if fig is None:
            plt = _import_mpl()
            fig = plt.figure()
        ax = fig.add_subplot(2, 2, 1)
        namestr = f" for {self.name}" if self.name else ""
        ax.plot(data)
        ax.set_title("Time series" + namestr)

        ax = fig.add_subplot(2, 2, 2)
        ax.plot(acf)
        ax.set_title("Autocorrelation" + namestr)

        ax = fig.add_subplot(2, 2, 3)
        ax.plot(spdr)  # (wr, spdr)
        ax.set_title("Power Spectrum" + namestr)

        ax = fig.add_subplot(2, 2, 4)
        ax.plot(pacf)
        ax.set_title("Partial Autocorrelation" + namestr)

        return fig
