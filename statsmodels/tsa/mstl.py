"""
Author: Kishan Manani
License: BSD-3 Clause

An implementation of MSTL [1], an algorithm for time series decomposition when
there are multiple seasonal components.

This implementation has the following differences with the original algorithm:
- Missing data must be handled outside of this class.
- The algorithm proposed in the paper handles a case when there is no
seasonality. This implementation assumes that there is at least one seasonal
component.

[1] K. Bandura, R.J. Hyndman, and C. Bergmeir (2021)
MSTL: A Seasonal-Trend Decomposition Algorithm for Time Series with Multiple
Seasonal Patterns
https://arxiv.org/pdf/2107.13462.pdf
"""
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import boxcox

from statsmodels.tools.typing import ArrayLike1D
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.tsatools import freq_to_period


class MSTL:
    """
    MSTL(endog, periods=None, windows=None, lmbda=None, iterate=2, stl_kwargs=None)

    Season-Trend decomposition using LOESS for multiple seasonalities.

    Parameters
    ----------
    endog : array_like
        Data to be decomposed. Must be squeezable to 1-d.
    periods : {int, array_like, None}, optional
        Periodicity of the seasonal components. If None and endog is a pandas
        Series or DataFrame, attempts to determine from endog. If endog is a
        ndarray, periods must be provided.
    windows : {int, array_like, None}, optional
        Length of the seasonal smoothers for each corresponding period.
        Must be an odd integer, and should normally be >= 7 (default). If None
        then default values determined using 7 + 4 * np.arange(1, n + 1, 1)
        where n is number of seasonal components.
    lmbda : {float, str, None}, optional
        The lambda parameter for the Box-Cox transform to be applied to `endog`
        prior to decomposition. If None, no transform is applied. If "auto", a
        value will be estimated that maximizes the log-likelihood function.
    iterate : int, optional
        Number of iterations to use to refine the seasonal component.
    stl_kwargs: dict, optional
        Arguments to pass to STL.

    See Also
    --------
    statsmodels.tsa.seasonal.STL

    References
    ----------
    .. [1] K. Bandura, R.J. Hyndman, and C. Bergmeir (2021)
    MSTL: A Seasonal-Trend Decomposition Algorithm for Time Series with
    Multiple Seasonal Patterns. arXiv preprint arXiv:2107.13462.

    Examples
    --------
    Start by creating a toy dataset with multiple seasonal components.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> register_matplotlib_converters()
    >>> t = np.arange(1, 1000)
    >>> trend = 0.0001 * t ** 2 + 100
    >>> daily_seasonality = 5 * np.sin(2 * np.pi * t / 24)
    >>> weekly_seasonality = 10 * np.sin(2 * np.pi * t / (24 * 7))
    >>> noise = np.random.randn(len(t))
    >>> y = trend + daily_seasonality + weekly_seasonality + noise

    Use MSTL to decompose the time series into multiple seasonal components
    with periods 24 (daily) and 24*7 (weekly).

    >>> from statsmodels.tsa.seasonal import MSTL
    >>> res = MSTL(data, periods=(24, 24*7)).fit()
    >>> res.plot()
    >>> plt.show()
    """

    def __init__(
        self,
        endog: ArrayLike1D,
        *,
        periods: Optional[Union[int, Sequence[int]]] = None,
        windows: Optional[Union[int, Sequence[int]]] = None,
        lmbda: Optional[Union[float, str]] = None,
        iterate: int = 2,
        stl_kwargs: Optional[Dict[str, Union[int, bool, None]]] = None,
    ):
        self.endog = endog
        self._y = self._to_1d_array(endog)
        self.nobs = self._y.shape[0]
        self.lmbda = lmbda
        self.periods = self._process_periods(periods)
        self.windows = self._process_windows(windows)
        self.iterate = iterate
        self._stl_kwargs = self._remove_overloaded_stl_args(
            stl_kwargs if stl_kwargs else {}
        )

        if len(self.periods) != len(self.windows):
            raise ValueError("Periods and windows must have same length")

    def fit(self):
        """
        Estimate multiple season components as well as trend and residuals
        components.

        Returns
        -------
        DecomposeResult
            Estimation results.
        """
        periods, windows = self._sort_periods_and_windows()

        # Remove long periods from decomposition
        periods = tuple(
            period for period in self.periods if period < self.nobs / 2
        )
        windows = self.windows[: len(periods)]
        num_seasons = len(periods)

        iterate = 1 if num_seasons == 1 else self.iterate

        # Box Cox
        if self.lmbda == "auto":
            y, lmbda = boxcox(self._y, lmbda=None)
            self.est_lmbda = lmbda
        elif self.lmbda:
            y = boxcox(self._y, lmbda=self.lmbda)
        else:
            y = self._y

        # Iterate over each seasonal component to extract seasonalities
        seasonal = np.zeros(shape=(num_seasons, self.nobs))
        deseas = y
        for _ in range(iterate):
            for i in range(num_seasons):
                deseas = deseas + seasonal[i]
                res = STL(
                    endog=deseas,
                    period=periods[i],
                    seasonal=windows[i],
                    **self._stl_kwargs,
                ).fit()
                seasonal[i] = res.seasonal
                deseas = deseas - seasonal[i]

        seasonal = np.squeeze(seasonal.T)
        trend = res.trend
        rw = res.weights
        resid = deseas - trend

        # Return pandas if endog is pandas
        if isinstance(self.endog, (pd.Series, pd.DataFrame)):
            index = self.endog.index
            y = pd.Series(y, index=index, name="observed")
            trend = pd.Series(trend, index=index, name="trend")
            resid = pd.Series(resid, index=index, name="resid")
            rw = pd.Series(rw, index=index, name="robust_weight")
            cols = [f"seasonal_{period}" for period in periods]
            if seasonal.ndim == 1:
                seasonal = pd.Series(seasonal, index=index, name="seasonal")
            else:
                seasonal = pd.DataFrame(seasonal, index=index, columns=cols)

        # Avoid circular imports
        from statsmodels.tsa.seasonal import DecomposeResult

        return DecomposeResult(y, seasonal, trend, resid, rw)

    def __str__(self):
        return (
            f"MSTL(endog,"
            f" periods={self.periods},"
            f" windows={self.windows},"
            f" lmbda={self.lmbda},"
            f" iterate={self.iterate})"
        )

    def _sort_periods_and_windows(self) -> Tuple[Sequence[int], Sequence[int]]:
        periods, windows = zip(*sorted(zip(self.periods, self.windows)))
        return periods, windows

    def _process_periods(self, periods) -> Sequence[int]:
        if periods is None:
            periods = tuple(self._infer_period(self.endog))
        elif isinstance(periods, int):
            periods = (periods,)
        else:
            pass
        return periods

    def _process_windows(self, windows) -> Sequence[int]:
        if windows is None:
            windows = self._default_seasonal_windows(len(self.periods))
        elif isinstance(windows, int):
            windows = (windows,)
        else:
            pass
        return windows

    @staticmethod
    def _remove_overloaded_stl_args(stl_args: Dict) -> Dict:
        args = ["endog", "period", "seasonal"]
        for arg in args:
            stl_args.pop(arg, None)
        return stl_args

    @staticmethod
    def _default_seasonal_windows(n: int) -> ArrayLike1D:
        return 7 + 4 * np.arange(1, n + 1, 1)  # See [1]

    @staticmethod
    def _infer_period(endog):
        freq = None
        if isinstance(endog, (pd.Series, pd.DataFrame)):
            freq = getattr(endog.index, "inferred_freq", None)
        if freq is None:
            raise ValueError("Unable to determine period from endog")
        period = freq_to_period(freq)
        return period

    @staticmethod
    def _to_1d_array(x):
        y = np.ascontiguousarray(np.squeeze(np.asarray(x)), dtype=np.double)
        if y.ndim != 1:
            raise ValueError("y must be a 1d array")
        return y
