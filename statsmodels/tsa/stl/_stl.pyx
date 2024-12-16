#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False, initializedcheck=False

"""
(c) 2019 Kevin Sheppard
License: NCSA/BSD-3 Clause
Based on NETLIB STL code

Notes: See file _stl_py.py in Git history for a pure python port for
the STL FORTRAN code

R.B. Cleveland, W.S.Cleveland, J.E. McRae, and I. Terpenning,
STL: A Seasonal-Trend Decomposition Procedure Based on Loess, Statistics
Research Report, AT&T Bell Laboratories.
"""

"""
Notes: Docstring from STL

PURPOSE
STL decomposes a time series into seasonal and trend  components.
It returns the components and robustness weights.
SYNOPSIS
stl(y, n, np, ns, nt, nl, isdeg, itdeg, ildeg, nsjump, ntjump,
        nljump, ni, no, rw, season, trend, work)
integer n, np, ns, nt, nl, isdeg, itdeg, ildeg, nsjump, ntjump,
        nljump, ni, no
real y(n), rw(n), season(n), trend(n), work(n+2*np,5)
ARGUMENTS
y       input, time series to be decomposed.
n       input, number of values in y.
np      input, the period of the seasonal component. For example,
        if  the  time series is monthly with a yearly cycle, then
        np=12.
ns      input, length of the seasonal smoother.  The value of  ns
        should be an odd integer greater than or equal to 3; ns>6
        is recommended.   As  ns  increases  the  values  of  the
        seasonal component at a given point in the seasonal cycle
        (e.g., January values of a monthly series with  a  yearly
        cycle) become smoother.
nt      input, length of the trend smoother.   The  value  of  nt
        should  be  an  odd integer greater than or equal to 3; a
        value of nt between 1.5*np and 2*np is  recommended.   As
        nt  increases  the  values  of the trend component become
        smoother.
nl      input, length of the low-pass filter.  The  value  of  nl
        should  be an odd integer greater than or equal to 3; the
        smallest odd integer greater  than  or  equal  to  np  is
        recommended.
isdeg   input, degree of locally-fitted  polynomial  in  seasonal
        smoothing.  The value is 0 or 1.
itdeg   input,  degree  of  locally-fitted  polynomial  in  trend
        smoothing.  The value is 0 or 1.
ildeg   input, degree of locally-fitted  polynomial  in  low-pass
        smoothing.  The value is 0 or 1.
nsjump  input,  skipping  value  for  seasonal  smoothing.    The
        seasonal  smoother  skips  ahead  nsjump  points and then
        linearly interpolates in between.  The  value  of  nsjump
        should  be  a  positive  integer; if nsjump=1, a seasonal
        smooth is calculated  at  all  n  points.   To  make  the
        procedure  run  faster, a reasonable choice for nsjump is
        10%-20% of ns.
ntjump  input, skipping value for trend smoothing.
nljump  input, skipping value for the low-pass filter.
ni      input, number of loops  for  updating  the  seasonal  and
        trend  components.   The value of ni should be a positive
        integer.  See the next argument for advice on the  choice
        of ni.
no      input, number of iterations of robust fitting.  The value
        of  no  should be a nonnegative integer.  If the data are
        well behaved without outliers, then robustness iterations
        are not needed.  In this case set no=0, and set ni=2 to 5
        depending  on  how  much  security  you  want  that   the
        seasonal-trend   looping   converges.   If  outliers  are
        present then no=3 is  a  very  secure  value  unless  the
        outliers are radical, in which case no=5 or even 10 might
        be better.  If no>0 then set ni to 1 or 2.
rw      output, final robustness weights. All rw are 1 if no=0.
season  output, seasonal component.
trend   output, trend component.
work    workspace of (n+2*np)*5 locations.
"""
from typing import Dict, Union

import numpy as np
import pandas as pd

from libc.math cimport NAN, fabs, isnan, sqrt

from statsmodels.tools.validation import array_like
from statsmodels.tsa.seasonal._seasonal import DecomposeResult
from statsmodels.tsa.tsatools import freq_to_period


def _is_pos_int(x, odd):
    valid = (isinstance(x, (int, np.integer))
             and not isinstance(x, np.timedelta64))
    valid = valid and not isinstance(x, (float, np.floating))
    try:
        valid = valid and x > 0
    except Exception:
        valid = False
    if valid and odd:
        valid = valid & (x % 2) == 1
    return valid


cdef class STL(object):
    """
    STL(endog, period=None, seasonal=7, trend=None, low_pass=None,
        seasonal_deg=1, trend_deg=1, low_pass_deg=1, robust=False,
        seasonal_jump=1, trend_jump=1, low_pass_jump=1)

    Season-Trend decomposition using LOESS.

    Parameters
    ----------
    endog : array_like
        Data to be decomposed. Must be squeezable to 1-d.
    period : {int, None}, optional
        Periodicity of the sequence. If None and endog is a pandas Series or
        DataFrame, attempts to determine from endog. If endog is a ndarray,
        period must be provided.
    seasonal : int, optional
        Length of the seasonal smoother. Must be an odd integer, and should
        normally be >= 7 (default).
    trend : {int, None}, optional
        Length of the trend smoother. Must be an odd integer. If not provided
        uses the smallest odd integer greater than
        1.5 * period / (1 - 1.5 / seasonal), following the suggestion in
        the original implementation.
    low_pass : {int, None}, optional
        Length of the low-pass filter. Must be an odd integer >=3. If not
        provided, uses the smallest odd integer > period.
    seasonal_deg : int, optional
        Degree of seasonal LOESS. 0 (constant) or 1 (constant and trend).
    trend_deg : int, optional
        Degree of trend LOESS. 0 (constant) or 1 (constant and trend).
    low_pass_deg : int, optional
        Degree of low pass LOESS. 0 (constant) or 1 (constant and trend).
    robust : bool, optional
        Flag indicating whether to use a weighted version that is robust to
        some forms of outliers.
    seasonal_jump : int, optional
        Positive integer determining the linear interpolation step. If larger
        than 1, the LOESS is used every seasonal_jump points and linear
        interpolation is between fitted points. Higher values reduce
        estimation time.
    trend_jump : int, optional
        Positive integer determining the linear interpolation step. If larger
        than 1, the LOESS is used every trend_jump points and values between
        the two are linearly interpolated. Higher values reduce estimation
        time.
    low_pass_jump : int, optional
        Positive integer determining the linear interpolation step. If larger
        than 1, the LOESS is used every low_pass_jump points and values between
        the two are linearly interpolated. Higher values reduce estimation
        time.

    See Also
    --------
    statsmodels.tsa.seasonal.DecomposeResult
    statsmodels.tsa.seasonal.seasonal_decompose

    Notes
    -----
    Derived from the NETLIB fortran written by [1]_.  The original code
    contains a bug that appears in the determination of the median that is
    used in the robust weighting. This version matches the fixed version that
    uses a correct partitioned sort to determine the median.

    See the notebook `Seasonal Decomposition
    <../examples/notebooks/generated/stl_decomposition.html>`__ for an overview.

    References
    ----------
    .. [1] R. B. Cleveland, W. S. Cleveland, J.E. McRae, and I. Terpenning
        (1990) STL: A Seasonal-Trend Decomposition Procedure Based on LOESS.
        Journal of Official Statistics, 6, 3-73.

    Examples
    --------
    The original example uses STL to decompose CO2 data into level, season and a
    residual.

    Start by aggregating to monthly, and filling any missing values

    >>> from statsmodels.datasets import co2
    >>> import matplotlib.pyplot as plt
    >>> from pandas.plotting import register_matplotlib_converters
    >>> register_matplotlib_converters()
    >>> data = co2.load(True).data
    >>> data = data.resample('ME').mean().ffill()

    The period (12) is automatically detected from the data's frequency ('M').

    >>> from statsmodels.tsa.seasonal import STL
    >>> res = STL(data).fit()
    >>> res.plot()
    >>> plt.show()

    .. plot:: plots/stl_plot.py
    """
    cdef object endog
    cdef Py_ssize_t nobs
    cdef int _period, seasonal, trend, low_pass, seasonal_deg, trend_deg
    cdef int low_pass_deg, low_pass_jump, trend_jump, seasonal_jump
    cdef bint robust, _use_rw
    cdef double[::1] _ya, _trend, _season, _rw
    cdef double[:, ::1] _work

    def __init__(self, endog, period=None, seasonal=7, trend=None, low_pass=None,
                 seasonal_deg=1, trend_deg=1, low_pass_deg=1,
                 robust=False, seasonal_jump=1, trend_jump=1, low_pass_jump=1):
        self.endog = endog
        y = array_like(endog, "endog", dtype=np.double, contiguous=True, writeable=True, ndim=1)
        self._ya = y
        self.nobs = y.shape[0]  # n
        if period is None:
            freq = None
            if isinstance(endog, (pd.Series, pd.DataFrame)):
                freq = getattr(endog.index, 'inferred_freq', None)
            if freq is None:
                raise ValueError('Unable to determine period from endog')
            period = freq_to_period(freq)
        if not _is_pos_int(period, False) or period < 2:
            raise ValueError('period must be a positive integer >= 2')
        self._period = period  # np
        if not _is_pos_int(seasonal, True) or seasonal < 3:
            raise ValueError('seasonal must be an odd positive integer >= 3')
        self.seasonal = seasonal  # ns
        if trend is None:
            trend = int(np.ceil(1.5 * self._period / (1 - 1.5 / self.seasonal)))
            # ensure odd
            trend += ((trend % 2) == 0)
        if not _is_pos_int(trend, True) or trend < 3 or trend <= period:
            raise ValueError('trend must be an odd positive integer '
                             '>= 3 where trend > period')
        self.trend = trend  # nt
        if low_pass is None:
            low_pass = self._period + 1
            low_pass += ((low_pass % 2) == 0)
        if not _is_pos_int(low_pass, True) or \
                low_pass < 3 or low_pass <= period:
            raise ValueError('low_pass must be an odd positive integer >= 3 '
                             'where low_pass > period')
        self.low_pass = low_pass  # nl
        self.seasonal_deg = seasonal_deg  # isdeg
        self.trend_deg = trend_deg  # itdeg
        self.low_pass_deg = low_pass_deg  # ildeg
        self.robust = robust
        if not _is_pos_int(low_pass_jump, False):
            raise ValueError('low_pass_jump must be a positive integer')
        if not _is_pos_int(seasonal_jump, False):
            raise ValueError('seasonal_jump must be a positive integer')
        if not _is_pos_int(trend_jump, False):
            raise ValueError('trend_jump must be a positive integer')
        self.low_pass_jump = low_pass_jump
        self.seasonal_jump = seasonal_jump
        self.trend_jump = trend_jump

        self._use_rw = False
        self._trend = np.zeros(self.nobs)
        self._season = np.zeros(self.nobs)
        self._rw = np.ones(self.nobs)
        self._work = np.zeros((7, self.nobs + 2 * period))

    def __reduce__(self):
        args = (
            self.endog,
            self._period,
            self.seasonal,
            self.trend,
            self.low_pass,
            self.seasonal_deg,
            self.trend_deg,
            self.low_pass_deg,
            self.robust,
            self.seasonal_jump,
            self.trend_jump,
            self.low_pass_jump,
        )
        return (STL, args)

    @property
    def period(self) -> int:
        """The period length of the time series"""
        return self._period

    @property
    def config(self) -> Dict[str, Union[int, bool]]:
        """
        The parameters used in the model.

        Returns
        -------
        dict[str, Union[int, bool]]
            The values used in the STL decomposition.
        """
        return dict(period=self._period,
                    seasonal=self.seasonal,
                    seasonal_deg=self.seasonal_deg,
                    seasonal_jump=self.seasonal_jump,
                    trend=self.trend,
                    trend_deg=self.trend_deg,
                    trend_jump=self.trend_jump,
                    low_pass=self.low_pass,
                    low_pass_deg=self.low_pass_deg,
                    low_pass_jump=self.low_pass_jump,
                    robust=self.robust)

    def fit(self, inner_iter=None, outer_iter=None):
        """
        fit(inner_iter=None, outer_iter=None)

        Estimate season, trend and residuals components.

        Parameters
        ----------
        inner_iter : {int, None}, optional
            Number of iterations to perform in the inner loop. If not provided
            uses 2 if ``robust`` is True, or 5 if not.
        outer_iter : {int, None}, optional
            Number of iterations to perform in the outer loop. If not provided
            uses 15 if ``robust`` is True, or 0 if not.

        Returns
        -------
        DecomposeResult
            Estimation results.
        """
        cdef Py_ssize_t i

        if inner_iter is None:
            inner_iter = 2 if self.robust else 5
        if outer_iter is None:
            outer_iter = 15 if self.robust else 0

        self._use_rw = False
        k = 0
        for i in range(self.nobs):
            self._season[i] = self._trend[i] = 0.0
            self._rw[i] = 1.0
        while True:
            self._onestp(inner_iter)
            k = k + 1
            if k > outer_iter:
                break
            for i in range(self.nobs):
                self._work[0, i] = self._trend[i] + self._season[i]
            self._rwts()
            self._use_rw = True

        # Return pandas if pandas
        season = np.asarray(self._season)
        trend = np.asarray(self._trend)
        rw = np.asarray(self._rw)
        resid = self._ya - season - trend
        if isinstance(self.endog, (pd.Series, pd.DataFrame)):
            index = self.endog.index
            resid = pd.Series(resid, index=index, name='resid')
            season = pd.Series(season, index=index, name='season')
            trend = pd.Series(trend, index=index, name='trend')
            rw = pd.Series(rw, index=index, name='robust_weight')

        return DecomposeResult(self.endog, season, trend, resid, rw)

    cdef void _onestp(self, int inner_iter):
        """
        y, n, np, ns, nt, nl, isdeg, itdeg, ildeg, nsjump,
                ntjump, nljump, ni, userw, rw, season, trend, work
        ->
        self._ya, self.nobs, self._period, self.seasonal,
                         self.trend, self.low_pass, self.seasonal_deg,
                         self.trend_deg, self.low_pass_deg, self.seasonal_jump,
                         self.trend_jump, self.low_pass_jump, inner_iter,
                         userw, self._rw, self._season, self._trend,
                         self._work
        """
        cdef Py_ssize_t i, j, np
        cdef double[:, ::1] work
        cdef double[::1] y, season, trend, rw
        # Original variable names
        work = self._work
        y = self._ya
        trend = self._trend
        n = self.nobs
        nl = self.low_pass
        ildeg = self.low_pass_deg
        nljump = self.low_pass_jump
        np = self._period
        season = self._season
        nt = self.trend
        itdeg = self.trend_deg
        ntjump = self.trend_jump
        rw = self._rw
        for j in range(inner_iter):
            for i in range(self.nobs):
                work[0, i] = y[i] - trend[i]
            self._ss()
            self._fts()
            self._ess(work[2, :], n, nl, ildeg, nljump, False, work[3, :],
                      work[0, :], work[4, :])
            for i in range(self.nobs):
                season[i] = work[1, np+i] - work[0, i]
                work[0, i] = y[i] - season[i]
            self._ess(work[0, :], n, nt, itdeg, ntjump, self._use_rw, rw,
                      trend, work[2, :])

    cdef double _est(self, double[::1] y, int n, int len_, int ideg, int xs,
                     int nleft, int nright, double[::1] w, bint userw,
                     double[::1] rw):
        cdef double rng, a, b, c, h, h1, h9, r, ys
        cdef Py_ssize_t j

        # Removed ok and ys, which are scalar return values
        rng = n - 1.0
        h = max(xs - nleft, nright - xs)
        if len_ > n:
            h += (len_ - n) // 2.0
        h9 = .999 * h
        h1 = .001 * h
        a = 0.0
        for j in range(nleft - 1, nright):
            w[j] = 0.
            r = fabs(j + 1 - xs)
            if r <= h9:
                if r <= h1:
                    w[j] = 1.0
                else:
                    w[j] = (1.0 - (r / h) ** 3) ** 3
                if userw:
                    w[j] = w[j] * rw[j]
                a = a + w[j]
        if a <= 0:
            return NAN
        for j in range(nleft - 1, nright):
            w[j] = w[j] / a
        if h > 0 and ideg > 0:
            a = 0.0
            for j in range(nleft - 1, nright):
                a = a + w[j] * (j + 1)
            b = xs - a
            c = 0.0
            for j in range(nleft - 1, nright):
                c = c + w[j] * (j + 1 - a) ** 2
            if sqrt(c) > .001 * rng:
                b = b / c
                for j in range(nleft - 1, nright):
                    w[j] = w[j] * (b * (j + 1 - a) + 1.0)
        ys = 0.0
        for j in range(nleft - 1, nright):
            ys = ys + w[j] * y[j]

        return ys

    cdef void _ess(self, double[::1] y, int n, int len_, int ideg, int njump,
                   bint userw, double[::1] rw, double[::1] ys, double[::1] res):
        # TODO: Try with 1 data point!!? Establish minimums
        cdef Py_ssize_t i, j, k
        cdef double delta
        cdef int newnj, nleft, nright, nsh
        cdef bint ok

        if n < 2:
            ys[0] = y[0]
            return
        newnj = min(njump, n - 1)
        if len_ >= n:
            nleft = 1
            nright = n
            i = 0
            while i < n:
                # formerly: for i in range(0, n, newnj):
                ys[i] = self._est(y, n, len_, ideg, i + 1, nleft, nright,
                                      res, userw, rw)
                if isnan(ys[i]):
                    ys[i] = y[i]
                i += newnj
        elif newnj == 1:
            nsh = (len_ + 2) // 2
            nleft = 1
            nright = len_
            for i in range(n):
                if (i + 1) > nsh and nright != n:
                    nleft = nleft + 1
                    nright = nright + 1
                ys[i] = self._est(y, n, len_, ideg, i + 1, nleft, nright,
                                      res, userw, rw)
                if isnan(ys[i]):
                    ys[i] = y[i]
        else:
            nsh = (len_ + 1) // 2
            i = 0
            while i < n:
                # formerly: for i in range(0, n, newnj):
                if (i + 1) < nsh:
                    nleft = 1
                    nright = len_
                elif (i + 1) >= (n - nsh + 1):
                    nleft = n - len_ + 1
                    nright = n
                else:
                    nleft = i + 1 - nsh + 1
                    nright = len_ + i + 1 - nsh
                ys[i] = self._est(y, n, len_, ideg, i + 1, nleft, nright,
                                      res, userw, rw)
                if isnan(ys[i]):
                    ys[i] = y[i]
                i += newnj
        if newnj == 1:
            return
        # newnj > 1
        i = 0
        while i < (n - newnj):
            # Formerly: for i in range(0, n - newnj, newnj):
            delta = (ys[i + newnj] - ys[i]) / newnj
            for j in range(i, i + newnj):
                ys[j] = ys[i] + delta * ((j + 1) - (i + 1))
            i += newnj
        k = ((n - 1) // newnj) * newnj + 1
        if k != n:
            ys[n - 1] = self._est(y, n, len_, ideg, n, nleft, nright, res,
                                      userw, rw)
            if isnan(ys[n - 1]):
                ys[n - 1] = y[n - 1]
            if k != (n - 1):
                delta = (ys[n - 1] - ys[k - 1]) / (n - k)
                for j in range(k, n):
                    ys[j] = ys[k - 1] + delta * ((j + 1) - k)

    cdef void _ma(self, double[::1] x, int n, int len_, double[::1] ave):
        cdef int newn
        cdef double flen, v
        cdef Py_ssize_t i, j, k, m

        newn = n - len_ + 1
        flen = float(len_)
        v = 0.0
        for i in range(len_):
            v = v + x[i]
        ave[0] = v / flen
        k = len_
        m = 0
        for j in range(1, newn):
            v += x[k] - x[m]
            ave[j] = v / flen
            k += 1
            m += 1

    cdef void _fts(self):
        """
        Original def:
        _fts(self, x, n, np, trend, work)
        Only call:
        fts(work[1, :], n + 2 * np, np, work[2, :], work[0, :])
        """
        cdef double[::1] x, trend, work
        cdef int n, np

        x = self._work[1, :]
        n = self.nobs + 2 * self._period
        np = self._period
        trend = self._work[2, :]
        work = self._work[0, :]
        self._ma(x, n, np, trend)
        self._ma(trend, n - np + 1, np, work)
        self._ma(work, n - 2 * np + 2, 3, trend)

    cdef void _ss(self):
        """
        _ss(self, y, n, np, ns, isdeg, nsjump, userw, rw, season, work1, work2,
            work3, work4)

        ss(work[0, :], n, np, ns, isdeg, nsjump, userw, rw,
                     work[1, :], work[2, :], work[3, :], work[4, :], season)
        """
        cdef Py_ssize_t i, j, m
        cdef int n, np, ns, isdef, nsjump, k
        cdef bint userw
        cdef double[::1] y, work1, work2, work3, work4, rw, season

        # Original variable names
        y = self._work[0, :]
        n = self.nobs
        np = self._period
        ns = self.seasonal
        isdeg = self.seasonal_deg
        nsjump = self.seasonal_jump
        rw = self._rw
        season = self._work[1, :]
        work1 = self._work[2, :]
        work2 = self._work[3, :]
        work3 = self._work[4, :]
        work4 = self._season
        userw = self._use_rw
        for j in range(np):
            k = (n - (j + 1)) // np + 1
            for i in range(k):
                work1[i] = y[i * np + j]
            if userw:
                for i in range(k):
                    work3[i] = rw[i * np + j]

            self._ess(work1, k, ns, isdeg, nsjump, userw, work3, work2[1:],
                      work4)
            xs = 0
            nright = min(ns, k)
            work2[0] = self._est(work1, k, ns, isdeg, xs, 1, nright, work4,
                                     userw, work3)
            if isnan(work2[0]):
                work2[0] = work2[1]
            xs = k + 1
            nleft = max(1, k - ns + 1)
            work2[k + 1] = self._est(work1, k, ns, isdeg, xs, nleft, k,
                                         work4, userw, work3)
            if isnan(work2[k + 1]):
                work2[k + 1] = work2[k]
            for m in range(k + 2):
                season[m * np + j] = work2[m]

    cdef void _rwts(self):
        """
        y, n, fit, rw ->
        self._ya, self.nobs, self._work[0, :], self._rw
        """
        cdef Py_ssize_t i
        cdef double [::1] y, fit, rw
        cdef double cmad, c1, c9
        cdef int n
        # Original variable names
        y = self._ya
        n = self.nobs
        fit = self._work[0, :]
        rw = self._rw
        for i in range(self.nobs):
            rw[i] = fabs(y[i] - fit[i])
        mid = np.empty(2, dtype=int)
        mid[0] = n // 2
        mid[1] = n - mid[0] - 1
        rw_part = np.partition(rw, mid)
        cmad = 3.0 * (rw_part[mid[0]] + rw_part[mid[1]])
        if cmad == 0:
            for i in range(self.nobs):
                rw[i] = 1
            return
        c9 = .999 * cmad
        c1 = .001 * cmad
        for i in range(self.nobs):
            if rw[i] <= c1:
                rw[i] = 1.0
            elif rw[i] <= c9:
                rw[i] = (1.0 - (rw[i] / cmad) ** 2) ** 2
            else:
                rw[i] = 0.0
