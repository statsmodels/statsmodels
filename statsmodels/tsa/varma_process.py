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

from statsmodels.tools.validation import string_like
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
        filtered array, 2d, number of columns determined by x and a

    Warnings
    --------
    convolve does not work as expected; this likely does not work
    correctly for nvars>3.

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
                     for all t >= nlags


    Note: maybe convert to axis=1, Not

    TODO: initial conditions

    """
    x = np.asarray(x)
    a = np.asarray(a)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim > 2:
        raise ValueError("x array has to be 1d or 2d")
    nvar = x.shape[1]
    nlags = a.shape[0]
    ntrim = nlags // 2
    # for x is 2d with ncols >1

    if a.ndim == 1:
        # case: identical ar filter (lag polynomial)
        return signal.convolve(x, a[:, None], mode="valid")
        # alternative:
        # return signal.lfilter(a,[1],x.astype(float),axis=0)
    elif a.ndim == 2:
        if min(a.shape) == 1:
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
        # Note: we must have shape[1] == shape[2] == nvar
        yf = signal.convolve(x[:, :, None], a)
        yvalid = yf[ntrim:-ntrim, yf.shape[1] // 2, :]
        return yvalid


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

    Examples
    --------

    >>> v = Var(ar2s)
    >>> v.fit(1)
    >>> v.arhat
    array([[[ 1.        ,  0.        ],
            [ 0.        ,  1.        ]],

           [[-0.77784898,  0.01726193],
            [ 0.10733009, -0.78665335]]])

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

        arhat : ndarray of shape (nlags, nvar, nvar)
            Full lag polynomial array.
        arlhs : ndarray of shape (nlags - 1, nvar, nvar)
            Reduced lag polynomial for the left-hand side.
        estresults
            The full result tuple as returned by ``linalg.lstsq``; other
            statistics still need to be completed.
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

    def predict(self):
        """Calculate estimated timeseries (yhat) for sample"""

        if not hasattr(self, "yhat"):
            self.yhat = varfilter(self.y, self.arhat)
        return self.yhat

    def covmat(self):
        """
        Covariance matrix of estimate

        Notes
        -----
        Not sure it's correct, need to check orientation everywhere.
        Looks ok, display needs getting used to.

        Examples
        --------
        >>> v.rss[None,None,:]*np.linalg.inv(np.dot(v.xred.T,v.xred))[:,:,None]
        array([[[ 0.37247445,  0.32210609],
                [ 0.1002642 ,  0.08670584]],

               [[ 0.1002642 ,  0.08670584],
                [ 0.45903637,  0.39696255]]])
        >>>
        >>> v.rss[0]*np.linalg.inv(np.dot(v.xred.T,v.xred))
        array([[ 0.37247445,  0.1002642 ],
               [ 0.1002642 ,  0.45903637]])
        >>> v.rss[1]*np.linalg.inv(np.dot(v.xred.T,v.xred))
        array([[ 0.32210609,  0.08670584],
               [ 0.08670584,  0.39696255]])
        """

        # check if orientation is same as self.arhat
        self.paramcov = (
            self.rss[None, None, :]
            * np.linalg.inv(np.dot(self.xred.T, self.xred))[:, :, None]
        )

    def forecast(self, horiz=1, u=None):
        """
        Calculates forecast for horiz number of periods at end of sample

        Parameters
        ----------
        horiz : int, optional
            Forecast horizon.
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
