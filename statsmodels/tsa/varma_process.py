"""
Helper and filter functions for VAR and VARMA, and basic VAR class

Created on Mon Jan 11 11:04:23 2010
Author: josef-pktd
License: BSD

This is a new version, I did not look at the old version again, but similar
ideas.

not copied/cleaned yet:
 * fftn based filtering, creating samples with fft
 * Tests: I ran examples but did not convert them to tests
   examples look good for parameter estimate and forecast, and filter functions

main TODOs:
* result statistics
* see whether Bayesian dummy observation can be included without changing
  the single call to linalg.lstsq
* impulse response function does not treat correlation, see Hamilton and jplv

Extensions
* constraints, Bayesian priors/penalization
* Error Correction Form and Cointegration
* Factor Models Stock-Watson,  ???


see also VAR section in Notes.txt

"""

import warnings

import numpy as np
from scipy import signal

from statsmodels.tools.validation import array_like, string_like
from statsmodels.tsa.tsatools import lagmat


def varfilter(x, a):
    """
    Apply an autoregressive filter to a series x

    x can be 2d, a can be 1d, 2d, or 3d

    Parameters
    ----------
    x : array_like
        data array, 1d or 2d, if 2d then observations in rows
    a : array_like
        autoregressive filter coefficients, ar lag polynomial
        see Notes

    Returns
    -------
    y : ndarray
        filtered array, 2d, of shape (nobs - nlags + 1, nvars) in cases 1
        and 2 and (nobs - nlags + 1, npoly) in case 3, see Notes. It holds
        the times t = nlags-1,...,nobs-1, at which every lag of x is
        available.

    Notes
    -----

    In general form this uses the linear filter ::

        y = a(L)x

    where
    x : nobs, nvars
    a : nlags, nvars, npoly

    Depending on the shape and dimension of a this uses different
    Lag polynomial arrays

    case 1 : a is 1d or (nlags,1)
        one lag polynomial is applied to all variables (columns of x)
    case 2 : a is 2d, (nlags, nvars)
        each series is independently filtered with its own
        lag polynomial, uses loop over nvar
    case 3 : a is 3d, (nlags, nvars, npoly)
        the ith column of the output array is given by the linear filter
        defined by the 2d array a[:,:,i], i.e., ::

            y[:,i] = a(.,.,i)(L) * x
            y[t,i] = sum_p sum_j a(p,j,i)*x(t-p,j)
                     for p = 0,...nlags-1, j = 0,...nvars-1,
                     for t = nlags-1,...,nobs-1

    .. versionchanged:: 0.16.0
       In case 3 the result previously used the variables of x in reverse
       order, and unless nlags was 2 it had the wrong number of rows, none
       when nlags was 1. It now follows the formula above. In case 2, an
       ``a`` of shape (1, nvars) with nvars > 1 (one lag) was taken as the
       single lag polynomial of case 1 and gave one column; it now gives one
       filter per variable.

    Note: maybe convert to axis=1, Not

    TODO: initial conditions

    """
    x = array_like(x, "x", dtype=None, mindim=0, maxdim=2)
    a = np.asarray(a)
    if x.ndim == 1:
        x = x[:, None]
    nobs, nvar = x.shape
    nlags = a.shape[0]
    # for x is 2d with ncols >1

    if a.ndim == 1:
        # case: identical ar filter (lag polynomial)
        return signal.convolve(x, a[:, None], mode="valid")
        # alternative:
        # return signal.lfilter(a,[1],x.astype(float),axis=0)
    elif a.ndim == 2:
        if a.shape[1] == 1:
            # case: identical ar filter (lag polynomial)
            return signal.convolve(x, a, mode="valid")

        # case: independent ar
        # (a bit like recserar in gauss, but no x yet)
        # (no, reserar is inverse filter)
        result = np.zeros((x.shape[0] - nlags + 1, nvar))
        for i in range(nvar):
            # could also use np.convolve, but easier for swiching to fft
            result[:, i] = signal.convolve(x[:, i], a[:, i], mode="valid")
        return result

    elif a.ndim == 3:
        # case: vector autoregressive with lag matrices
        # Note: we must have shape[1] == nvar
        # y[t] = sum_p x[t - p] @ a[p] for t = nlags - 1, ..., nobs - 1
        return sum(x[nlags - 1 - p : nobs - p] @ a[p] for p in range(nlags))


def varinversefilter(ar, nobs, version=1):
    """
    Creates inverse ar filter (MA representation) recursively

    The VAR lag polynomial is defined by ::

        ar(L) y_t = u_t  or
        y_t = -ar_{-1}(L) y_{t-1} + u_t

    the returned lagpolynomial is arinv(L)=ar^{-1}(L) in ::

        y_t = arinv(L) u_t

    Parameters
    ----------
    ar : ndarray, (nlags,nvars,nvars)
        matrix lagpolynomial, currently no exog
        first row should be identity
    nobs : int
        Number of observations (time points) for which to compute the
        inverse filter.
    version : int, optional
        Selects the implementation to use. If 1 (the default), the
        inverse filter is computed recursively. If 0, the alternative
        implementation is not yet finished and raises NotImplementedError.

    Returns
    -------
    arinv : ndarray
        The inverse (MA representation) lag polynomial array, of shape
        (nobs + 1, nvars, nvars).
    """
    nlags, nvars, nvarsex = ar.shape
    if nvars != nvarsex:
        print("exogenous variables not implemented not tested")
    arinv = np.zeros((nobs + 1, nvarsex, nvars))
    arinv[0, :, :] = ar[0]
    arinv[1:nlags, :, :] = -ar[1:]
    if version == 1:
        for i in range(2, nobs + 1):
            tmp = np.zeros((nvars, nvars))
            for p in range(1, nlags):
                tmp += np.dot(-ar[p], arinv[i - p, :, :])
            arinv[i, :, :] = tmp
    if version == 0:
        for i in range(nlags + 1, nobs + 1):
            print(ar[1:].shape, arinv[i - 1 : i - nlags : -1, :, :].shape)
            # arinv[i,:,:] = np.dot(-ar[1:],arinv[i-1:i-nlags:-1,:,:])
            # print(np.tensordot(-ar[1:],arinv[i-1:i-nlags:-1,:,:],axes=([2],[1])).shape
            # arinv[i,:,:] = np.tensordot(-ar[1:],arinv[i-1:i-nlags:-1,:,:],axes=([2],[1]))
            raise NotImplementedError("waiting for generalized ufuncs or something")

    return arinv


def vargenerate(ar, u, initvalues=None):
    """
    Generate a VAR process with errors u

    similar to gauss
    uses loop

    Parameters
    ----------
    ar : ndarray
        Matrix lagpolynomial, of shape (nlags, nvars, nvars).
    u : ndarray
        Error terms (innovations) for the VAR process, of shape
        (nobs, nvars).
    initvalues : ndarray, optional
        Initial (presample) values for the process. If None, the initial
        values are set to zero.

    Returns
    -------
    sar : ndarray
        Sample of the VAR process, the inverse filtered `u`. Has shape
        (nobs + max(nlags - 1, len(initvalues)), nvars); the presample
        values, including the initial condition y_0 = 0, are not
        trimmed.

    Examples
    --------
    # generate random sample of VAR
    nobs, nvars = 10, 2
    u = numpy.random.randn(nobs,nvars)
    a21 = np.array([[[ 1. ,  0. ],
                     [ 0. ,  1. ]],

                    [[-0.8,  0. ],
                     [ 0.,  -0.6]]])
    vargenerate(a21,u)

    # Impulse Response to an initial shock to the first variable
    imp = np.zeros((nobs, nvars))
    imp[0,0] = 1
    vargenerate(a21,imp)

    """
    nlags, nvars, nvarsex = ar.shape
    nlagsm1 = nlags - 1
    nobs = u.shape[0]
    if nvars != nvarsex:
        print("exogenous variables not implemented not tested")
    if u.shape[1] != nvars:
        raise ValueError("u needs to have nvars columns")
    if initvalues is None:
        sar = np.zeros((nobs + nlagsm1, nvars))
        start = nlagsm1
    else:
        start = max(nlagsm1, initvalues.shape[0])
        sar = np.zeros((nobs + start, nvars))
        sar[start - initvalues.shape[0] : start] = initvalues
    # sar[nlagsm1:] = u
    sar[start:] = u
    # if version == 1:
    for i in range(start, start + nobs):
        for p in range(1, nlags):
            sar[i] += np.dot(sar[i - p, :], -ar[p])

    return sar


def padone(x, front=0, back=0, axis=0, fillvalue=0):
    """
    Pad with zeros along one axis

    Can be used sequentially to pad several axes.

    Parameters
    ----------
    x : ndarray
        Array to pad.
    front : int, optional
        Number of `fillvalue` elements to add before the array along
        `axis`.
    back : int, optional
        Number of `fillvalue` elements to add after the array along
        `axis`.
    axis : int, optional
        Axis along which to pad.
    fillvalue : scalar, optional
        Value used to fill the padded elements.

    Returns
    -------
    ndarray
        The padded array.

    Examples
    --------
    >>> padone(np.ones((2,3)),1,3,axis=1)
    array([[ 0.,  1.,  1.,  1.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.,  0.,  0.]])

    >>> padone(np.ones((2,3)),1,1, fillvalue=np.nan)
    array([[ NaN,  NaN,  NaN],
           [  1.,   1.,   1.],
           [  1.,   1.,   1.],
           [ NaN,  NaN,  NaN]])
    """
    # primitive version
    shape = np.array(x.shape)
    shape[axis] += front + back
    shapearr = np.array(x.shape)
    out = np.empty(shape)
    out.fill(fillvalue)
    startind = np.zeros(x.ndim)
    startind[axis] = front
    endind = startind + shapearr
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    # print(myslice
    # print(out.shape
    # print(out[tuple(myslice)].shape
    out[tuple(myslice)] = x
    return out


def trimone(x, front=0, back=0, axis=0):
    """
    Trim a number of array elements along one axis

    Parameters
    ----------
    x : ndarray
        Array to trim.
    front : int, optional
        Number of elements to remove from the front along `axis`.
    back : int, optional
        Number of elements to remove from the back along `axis`.
    axis : int, optional
        Axis along which to trim.

    Returns
    -------
    ndarray
        The trimmed array.

    Examples
    --------
    >>> xp = padone(np.ones((2,3)),1,3,axis=1)
    >>> xp
    array([[ 0.,  1.,  1.,  1.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.,  0.,  0.]])
    >>> trimone(xp,1,3,1)
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]])
    """
    shape = np.array(x.shape)
    shape[axis] -= front + back
    # print(shape, front, back
    startind = np.zeros(x.ndim)
    startind[axis] = front
    endind = startind + shape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    # print(myslice
    # print(shape, endind
    # print(x[tuple(myslice)].shape
    return x[tuple(myslice)]


def ar2full(ar):
    """
    Make reduced lagpolynomial into a right side lagpoly array

    Parameters
    ----------
    ar : ndarray
        Reduced-form lag polynomial array, of shape (nlags, nvar, nvarex).

    Returns
    -------
    ndarray
        Full (right-hand side) lag polynomial array, of shape
        (nlags + 1, nvar, nvarex), with an identity matrix prepended.
    """
    nlags, nvar, nvarex = ar.shape
    return np.r_[np.eye(nvar, nvarex)[None, :, :], -ar]


def ar2lhs(ar):
    """
    Convert full (rhs) lagpolynomial into a reduced, left side lagpoly array

    This is mainly a reminder about the definition.

    Parameters
    ----------
    ar : ndarray
        Full (right-hand side) lag polynomial array.

    Returns
    -------
    ndarray
        Reduced, left-hand side lag polynomial array.
    """
    return -ar[1:]


class _Var:
    """
    Obsolete VAR class, use tsa.VAR instead, for internal use only

    Parameters
    ----------
    y : ndarray or DataFrame
        The data, of shape (nobs, nvars), with observations in rows.

    Examples
    --------
    >>> import numpy as np
    >>> from statsmodels.tsa.varma_process import _Var
    >>> rng = np.random.default_rng(0)
    >>> a = np.array([[0.8, 0.0], [-0.1, 0.8]])
    >>> y = np.zeros((500, 2))
    >>> for t in range(1, 500):
    ...     y[t] = a @ y[t - 1] + rng.standard_normal(2)
    >>> v = _Var(y)
    >>> v.fit(1)
    >>> v.arhat.round(2)
    array([[[ 1.  ,  0.  ],
            [ 0.  ,  1.  ]],
    <BLANKLINE>
           [[-0.83,  0.09],
            [ 0.02, -0.75]]])
    """

    def __init__(self, y):
        warnings.warn(
            "_Var is deprecated (its own docstring already called it "
            "\"Obsolete\" -- use tsa.VAR instead). It has had no test "
            "coverage and its behavior is not guaranteed. It will be "
            "removed after statsmodels 0.16 is released. If you rely on "
            "this class, please open an issue at "
            "https://github.com/statsmodels/statsmodels/issues.",
            FutureWarning,
            stacklevel=2,
        )
        self.y = y
        self.nobs, self.nvars = y.shape

    def fit(self, nlags):
        """
        Estimate parameters using OLS

        Parameters
        ----------
        nlags : int
            number of lags to include in regression, same for all variables

        Returns
        -------
        None
            Nothing is returned; the estimation results are attached to
            the instance, see Notes.

        Notes
        -----
        This currently assumes all parameters are estimated without restrictions.
        In this case SUR is identical to OLS.

        The following are attached to the instance:

        arhat : ndarray of shape (nlags + 1, nvar, nvar)
            Full lag polynomial array, the identity followed by ``-arlhs``.
        arlhs : ndarray of shape (nlags, nvar, nvar)
            Reduced lag polynomial for the left-hand side. ``arlhs[i]`` is
            the transpose of the coefficient matrix of lag ``i + 1``.
        estresults
            The full result tuple as returned by ``linalg.lstsq``; other
            statistics still need to be completed.
        rss : ndarray of shape (nvar,)
            The residual sum of squares of each equation.
        xredrank : int
            The rank of the regressor matrix.
        yred, xred : ndarray
            The dependent variables and the lagged regressors used in the
            regression.
        """
        self.nlags = nlags  # without current period
        nvars = self.nvars
        # TODO: ar2s looks like a module variable, bug?
        # lmat = lagmat(ar2s, nlags, trim='both', original='in')
        lmat = lagmat(self.y, nlags, trim="both", original="in")
        self.yred = lmat[:, :nvars]
        self.xred = lmat[:, nvars:]
        res = np.linalg.lstsq(self.xred, self.yred, rcond=-1)
        self.estresults = res
        self.arlhs = res[0].reshape(nlags, nvars, nvars)
        self.arhat = ar2full(self.arlhs)
        self.rss = res[1]
        self.xredrank = res[2]
        # Reset the fitted values cached by predict
        self.yhat = None

    def predict(self):
        """
        Calculate estimated timeseries (yhat) for sample

        Returns
        -------
        yhat : ndarray
            The fitted values ``xred @ estresults[0]`` of the regression in
            `fit`, of shape (nobs - nlags, nvars), for the observations
            ``nlags, ..., nobs - 1``. They are also attached as ``yhat``,
            and are computed again after `fit` is called.

        Notes
        -----
        .. versionchanged:: 0.16.0
           The result was previously neither the fitted values nor the
           residuals, and it had nobs - nlags rows only when nlags was 1.
           It was also not updated when `fit` was called again with a
           different number of lags.
        """
        if self.yhat is None:
            self.yhat = self.xred @ self.estresults[0]
        return self.yhat

    def covmat(self):
        """
        Covariance matrix of estimate

        Returns
        -------
        None
            Nothing is returned; the result is attached to the instance as
            ``paramcov``, of shape (nlags * nvar, nlags * nvar, nvar), where
            ``paramcov[:, :, i]`` belongs to equation ``i``.

        Notes
        -----
        ``paramcov[:, :, i]`` is ``rss[i] / df_resid * inv(xred.T @ xred)``,
        the covariance of the coefficients ``estresults[0][:, i]`` of
        equation ``i``. ``df_resid = nobs - nlags - nlags * nvar`` is the
        number of observations in the regression less the number of
        coefficients in each equation. These are the diagonal blocks of the
        ``cov_params`` of ``VAR(y).fit(nlags, trend="n")``; the covariances
        between the coefficients of different equations are not computed.

        .. versionchanged:: 0.16.0
           ``paramcov`` was previously too large by the factor
           ``df_resid``: it scaled ``inv(xred.T @ xred)`` by the residual
           sum of squares of each equation instead of its residual
           variance.

        Examples
        --------
        Continuing the example of the class, the block of the first
        equation is the matching block of the covariance of the VAR
        estimates:

        >>> from statsmodels.tsa.api import VAR
        >>> v.covmat()
        >>> cov = VAR(y).fit(1, trend="n").cov_params()
        >>> np.allclose(v.paramcov[:, :, 0], cov[0::2, 0::2])
        True
        """
        # The residual variance of each equation, with the degrees of freedom
        # of VAR: the observations in the regression less the coefficients of
        # each equation
        df_resid = self.xred.shape[0] - self.xred.shape[1]
        self.paramcov = (
            (self.rss / df_resid)[None, None, :]
            * np.linalg.inv(np.dot(self.xred.T, self.xred))[:, :, None]
        )

    def forecast(self, horiz=1, u=None):
        """
        Calculates forecast for horiz number of periods at end of sample

        Parameters
        ----------
        horiz : int, optional
            Forecast horizon. Ignored if `u` is given, in which case the
            horizon is the number of rows of `u`.
        u : ndarray, optional
            Error term for forecast periods, of shape (horiz, nvars). If
            None, then u is zero.

        Returns
        -------
        yforecast : ndarray
            Forecast array, of shape (nobs + horiz, nvars); this includes
            the sample and the forecasts.
        """
        if u is None:
            u = np.zeros((horiz, self.nvars))
        return vargenerate(self.arhat, u, initvalues=self.y)


class VarmaPoly:
    """
    Class to keep track of Varma polynomial format

    Parameters
    ----------
    ar : ndarray
        The autoregressive lag polynomial array, of shape
        (nlags, nvarall, nvars).
    ma : ndarray, optional
        The moving-average lag polynomial array, of shape
        (malags, nvars, nvars). If None, no moving-average terms are
        used and an identity matrix is used in their place.

    Examples
    --------

    ar23 = np.array([[[ 1. ,  0. ],
                     [ 0. ,  1. ]],

                    [[-0.6,  0. ],
                     [ 0.2, -0.6]],

                    [[-0.1,  0. ],
                     [ 0.1, -0.1]]])

    ma22 = np.array([[[ 1. ,  0. ],
                     [ 0. ,  1. ]],

                    [[ 0.4,  0. ],
                     [ 0.2, 0.3]]])


    """

    def __init__(self, ar, ma=None):
        self.ar = ar
        self.ma = ma
        nlags, nvarall, nvars = ar.shape
        self.nlags, self.nvarall, self.nvars = nlags, nvarall, nvars
        self.isstructured = not (ar[0, :nvars] == np.eye(nvars)).all()
        if self.ma is None:
            self.ma = np.eye(nvars)[None, ...]
            self.isindependent = True
        else:
            self.isindependent = not (ma[0] == np.eye(nvars)).all()
        self.malags = ar.shape[0]
        self.hasexog = nvarall > nvars
        self.arm1 = -ar[1:]

    # @property
    def vstack(self, a=None, name="ar"):
        """
        Stack lagpolynomial vertically in 2d array

        Parameters
        ----------
        a : ndarray, optional
            Lag polynomial array to stack. If None, uses ``self.ar`` or
            ``self.ma``, selected by `name`.
        name : {"ar", "ma"}, optional
            Which instance lag polynomial to use when `a` is None.

        Returns
        -------
        ndarray
            The lag polynomial stacked vertically into a 2d array.
        """
        if a is not None:
            _a = a
        else:
            name = string_like(name, "name", options=("ar", "ma"))
            _a = self.ar if name == "ar" else self.ma
        return _a.reshape(-1, self.nvarall)

    # @property
    def hstack(self, a=None, name="ar"):
        """
        Stack lagpolynomial horizontally in 2d array

        Parameters
        ----------
        a : ndarray, optional
            Lag polynomial array to stack. If None, uses ``self.ar`` or
            ``self.ma``, selected by `name`.
        name : {"ar", "ma"}, optional
            Which instance lag polynomial to use when `a` is None.

        Returns
        -------
        ndarray
            The lag polynomial stacked horizontally into a 2d array.
        """
        if a is not None:
            _a = a
        else:
            name = string_like(name, "name", options=("ar", "ma"))
            _a = self.ar if name == "ar" else self.ma
        return _a.swapaxes(1, 2).reshape(-1, self.nvarall).T

    # @property
    def stacksquare(self, a=None, name="ar", orientation="vertical"):
        """
        Stack lagpolynomial vertically in 2d square array with eye

        Parameters
        ----------
        a : ndarray, optional
            Lag polynomial array to stack. If None, uses ``self.ar`` or
            ``self.ma``, selected by `name`.
        name : {"ar", "ma"}, optional
            Which instance lag polynomial to use when `a` is None.
        orientation : str, optional
            Currently not used.

        Returns
        -------
        ndarray
            The lag polynomial stacked vertically into a 2d square array,
            with an identity block appended.
        """
        if a is not None:
            _a = a
        else:
            name = string_like(name, "name", options=("ar", "ma"))
            _a = self.ar if name == "ar" else self.ma
        astacked = _a.reshape(-1, self.nvarall)
        lenpk, nvars = astacked.shape  # [0]
        amat = np.eye(lenpk, k=nvars)
        amat[:, :nvars] = astacked
        return amat

    # @property
    def vstackarma_minus1(self):
        """Stack ar and lagpolynomial vertically in 2d array"""
        a = np.concatenate((self.ar[1:], self.ma[1:]), 0)
        return a.reshape(-1, self.nvarall)

    # @property
    def hstackarma_minus1(self):
        """
        Stack ar and lagpolynomial vertically in 2d array

        This is the Kalman Filter representation, I think.
        """
        a = np.concatenate((self.ar[1:], self.ma[1:]), 0)
        return a.swapaxes(1, 2).reshape(-1, self.nvarall)

    def getisstationary(self, a=None):
        """
        Check whether the auto-regressive lag-polynomial is stationary

        Parameters
        ----------
        a : ndarray, optional
            The lag polynomial array to check. If None, uses the reduced
            form of ``self.ar``.

        Returns
        -------
        isstationary : bool
            True if all eigenvalues of the lag-polynomial are less than one
            in absolute value.

        Notes
        -----
        Attaches ``areigenvalues``, the eigenvalues sorted by absolute
        value, as a complex array.

        References
        ----------
        Formula taken from NAG manual.
        """
        if a is not None:
            _a = a
        elif self.isstructured:
            _a = -self.reduceform(self.ar)[1:]
        else:
            _a = -self.ar[1:]
        amat = self.stacksquare(_a)
        ev = np.sort(np.linalg.eigvals(amat))[::-1]
        self.areigenvalues = ev
        return (np.abs(ev) < 1).all()

    def getisinvertible(self, a=None):
        """
        Check whether the moving-average lag-polynomial is invertible

        Parameters
        ----------
        a : ndarray, optional
            The lag polynomial array to check. If None, uses the reduced
            form of ``self.ma``.

        Returns
        -------
        isinvertible : bool
            True if all eigenvalues of the lag-polynomial are less than one
            in absolute value.

        Notes
        -----
        Attaches ``maeigenvalues``, the eigenvalues sorted by absolute
        value, as a complex array.

        References
        ----------
        Formula taken from NAG manual.
        """
        if a is not None:
            _a = a
        elif self.isindependent:
            _a = self.reduceform(self.ma)[1:]
        else:
            _a = self.ma[1:]
        if _a.shape[0] == 0:
            # no ma lags
            self.maeigenvalues = np.array([], dtype=complex)
            return True

        amat = self.stacksquare(_a)
        ev = np.sort(np.linalg.eigvals(amat))[::-1]
        self.maeigenvalues = ev
        return (np.abs(ev) < 1).all()

    def reduceform(self, apoly):
        """
        Convert a structural lag polynomial to reduced form

        This assumes no exog, todo

        Parameters
        ----------
        apoly : ndarray
            3d lag polynomial array, of shape (nlags, nvars, nvars).

        Returns
        -------
        ndarray
            The reduced-form lag polynomial array, with the same shape as
            `apoly`.
        """
        if apoly.ndim != 3:
            raise ValueError("apoly needs to be 3d")
        nlags, nvarsex, nvars = apoly.shape

        try:
            a0inv = np.linalg.inv(apoly[0, :nvars, :])
        except np.linalg.LinAlgError as la_err:
            raise ValueError(
                "matrix not invertible, ask for implementation of pinv"
            ) from la_err

        a = np.empty_like(apoly)
        for lag in range(nlags):
            a[lag] = np.dot(a0inv, apoly[lag])

        return a
