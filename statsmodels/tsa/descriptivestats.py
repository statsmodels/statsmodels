"""
Descriptive Statistics for Time Series

Created on Sat Oct 30 14:24:08 2010

Author: josef-pktd
License: BSD(3clause)
"""

import warnings

import numpy as np

from statsmodels.graphics.utils import _import_mpl
from statsmodels.tools.validation import array_like

from . import stattools as stt


# TODO: check subclassing for descriptive stats classes
class TsaDescriptive:
    """
    Collection of descriptive statistical methods for time series

    Parameters
    ----------
    data : ndarray or Series
        The time series data, 1-D.
    label : str, optional
        A label used to identify the series.
    name : str, optional
        A name used as the base for labeling series derived from `data`,
        such as those produced by `filter` and `detrend`.

    Attributes
    ----------
    data : ndarray or Series
        The time series data, as passed.
    label : str
        The label identifying the series.
    name : str
        The base name used for derived series.
    mod : ARIMA or None
        The model created by `fit`, None until `fit` is called.
    res : ARIMAResults or None
        The results returned by `fit`, None until `fit` is called.
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
            A new instance containing the filtered data, with the same
            label and the name ``name + "_filtered"``.

        Notes
        -----
        The filter is applied by ``scipy.signal.lfilter`` with zero initial
        conditions.
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
            The polynomial order of the trend to remove. The default, 1,
            removes a linear trend; 0 removes the mean.

        Returns
        -------
        TsaDescriptive
            A new instance containing the detrended data, with the same
            label and the name ``name + "_detrended"``.
        """
        from . import tsatools
        xdetrended = tsatools.detrend(self.data, order=order)
        return self.__class__(xdetrended, self.label, self.name + "_detrended")

    def fit(self, order=(1, 0, 1), **kwds):
        """
        Fit an ARIMA model to the time series.

        Parameters
        ----------
        order : tuple of int, optional
            The (p, d, q) order of the ARIMA model to fit. The default is
            (1, 0, 1).
        **kwds
            Additional keyword arguments passed to ``ARIMA.fit``.

        Returns
        -------
        ARIMAResults
            The results instance from fitting the model. It is also stored
            as `res`, and the model as `mod`.
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
            The autocorrelation function at lags 0 to `nlags`.
        """
        return stt.acf(self.data, nlags=nlags)

    def pacf(self, nlags=40):
        """
        Compute the partial autocorrelation function of the time series.

        Parameters
        ----------
        nlags : int, optional
            The number of lags to include, at most half the number of
            observations.

        Returns
        -------
        ndarray
            The partial autocorrelation function at lags 0 to `nlags`.
        """
        return stt.pacf(self.data, nlags=nlags)

    def periodogram(self):
        """
        Compute the periodogram of the time series.

        Returns
        -------
        ndarray
            The periodogram ``abs(fft(data)) ** 2 / nobs`` at the Fourier
            frequencies ``2 * pi * k / nobs`` for ``k = 0, ..., nobs - 1``,
            with the value at frequency zero set to 0. The frequencies are
            not returned.

        Notes
        -----
        .. versionchanged:: 0.16.0
           Since statsmodels 0.12 this method, and `plot4`, raised
           AttributeError, because it called
           ``statsmodels.tsa.stattools.periodogram``, which was removed. It
           now computes the periodogram as that function did.
        """
        # The code of statsmodels.tsa.stattools.periodogram, removed in
        # statsmodels 0.12
        x = array_like(self.data, "x")
        pergr = 1.0 / len(x) * np.abs(np.fft.fft(x)) ** 2
        pergr[0] = 0.0
        return pergr

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

        Notes
        -----
        .. versionchanged:: 0.16.0
           Since statsmodels 0.12 this method raised AttributeError, from
           `periodogram`. It now draws all four plots.
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
