#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Decompositions of (difference) stationary time series according
to Wold's Theorem.

"""
import warnings

import numpy as np
from pandas.util.decorators import deprecate_kwarg
import scipy.linalg
import scipy.signal

from statsmodels.tools.decorators import cache_readonly

# -----------------------------------------------------------------------
# Input Validation


def _shape_params(params):
    if np.ndim(params) == 1:
        # Univariate Case, the one dimension of params represents lags
        params = params[:, None, None]
    elif np.ndim(params) == 2:
        # i.e. 1 lag
        params = params[None, :, :]
    return params


# TODO: re-cover the ValueErrors here
def _check_is_poly(params):
    """
    Validate that the given params is in the form of a lag polynomial,
    i.e. is a 1-dimensional np.ndarray with first entry 1.
    """
    if np.ndim(params) != 1:  # pragma: no cover
        raise ValueError(params.shape, params)
    elif params[0] != 1:  # pragma: no cover
        msg = 'Params must be in the form of a Lag Polynomial'
        raise ValueError(msg, params)


# TODO: re-cover the ValueErrors here
def _check_param_dims(params):
    """
    `params` are either `ar` or `ma`
    """
    # if len(params.shape) == 2:
    #    params = params[None, :, :]
    if len(params.shape) != 3:  # pragma: no cover
        msg = 'AR and MA should each be 1 or 3-dimensional.'
        raise ValueError(params.shape, msg)
    if params.shape[1] != params.shape[2]:  # pragma: no cover
        msg = ('Dimensions 1 and 2 of AR coefficients should be equal.  '
               'i.e. K x N x N')
        raise ValueError(params.shape, msg)


# TODO: re-cover the ValueErrors here
def _check_intercept_shape(intercept, neqs):
    if intercept is None:
        intercept = np.zeros(neqs)

    intercept = np.array(intercept)
    if neqs == 1:
        accepted_shapes = [(), (1,)]
    else:
        accepted_shapes = [(neqs,)]
    if intercept.shape not in accepted_shapes:  # pragma: no cover
        raise ValueError(intercept.shape, accepted_shapes)
    return intercept


# TODO: re-cover the ValueErrors here
def _unpack_lags_and_neqs(ar, ma, intercept):
    # Check if these coefficients correspond to a vector-ARMA

    k_ar = ar.shape[0]
    if len(ar.shape) > 1:
        _check_param_dims(ar)
        neqs = ar.shape[1]
    else:
        # Default is univariate
        neqs = 1

    if ma is not None:
        k_ma = ma.shape[0]
        if len(ma.shape) > 1:
            _check_param_dims(ma)
            if len(ar.shape) != 3 or ar.shape[1] != ma.shape[1]:  # pragma: no cover
                msg = 'ar.shape[1:] must match ma.shape[1:]'
                raise ValueError(ar.shape, ma.shape, msg)
            neqs = ma.shape[1]
    else:
        k_ma = 0

    intercept = _check_intercept_shape(intercept, neqs)
    return (k_ar, k_ma, neqs, intercept)


# -----------------------------------------------------------------------
# Base/Mixin Classes

class _DimBase(object):
    @property
    def k_ma(self):
        try:
            macoefs = self.macoefs
            if macoefs is None:
                ret = 0
            else:
                ret = macoefs.shape[0]
        except AttributeError:
            # kludge
            maparams = self.maparams
            if maparams is None:
                ret = 0
            else:
                ret = maparams.shape[0] - 1

        return ret

    @property
    def k_ar(self):
        try:
            arcoefs = self.arcoefs
            ret = arcoefs.shape[0]
        except AttributeError:
            # kludge
            arparams = self.arparams
            ret = arparams.shape[0] - 1
        return ret

    @property
    def neqs(self):
        try:
            arcoefs = self.arcoefs
            ret = arcoefs.shape[1]
        except AttributeError:
            # kludge
            arparams = self.arparams
            ret = arparams.shape[1]
        return ret


class RootsMixin(object):
    @property
    def arroots(self):
        """Roots of autoregressive lag-polynomial"""
        # TODO: should docstring say "inverse"?
        return np.roots(np.r_[1, -self.arcoefs])**-1
        # Equiv: self.arpoly.roots()

    # TODO: trim 0 off the end of ma args, avoiding np.inf in cases where
    # you just passed [1, 0]
    @property
    def maroots(self):
        """Roots of moving average lag-polynomial"""
        # TODO: should docstring say "inverse"?
        return np.roots(np.r_[1, self.macoefs])**-1
        # Equiv: self.mapoly.roots()

    @property
    def arfreq(self):
        r"""Returns the frequency of the AR roots.

        This is the solution, x, to z = abs(z)*exp(2j*np.pi*x) where z
        are the roots.
        """
        z = self.arroots
        # if not z.size:
        #    return np.nan
        return np.arctan2(z.imag, z.real) / (2*np.pi)

    @property
    def mafreq(self):
        r"""Returns the frequency of the MA roots.

        This is the solution, x, to z = abs(z)*exp(2j*np.pi*x) where z
        are the roots.
        """
        z = self.maroots
        # if not z.size:
        #    return np.nan
        return np.arctan2(z.imag, z.real) / (2*np.pi)

    @property
    def isstationary(self):
        """Arma process is stationary if AR roots are outside unit circle

        Returns
        -------
        isstationary : boolean
            True if autoregressive roots are outside unit circle
        """
        return np.all(np.abs(self.arroots) > 1)

    @property
    def isinvertible(self):
        """ARMA process is invertible if MA roots are outside unit circle.

        From Powell (http://eml.berkeley.edu/~powell/e241b_f06/TS-StatInv.pdf):


        If the MA(q) process

            y_t = \mu + \epsilon_t + \theta_1\epsilon_{t-1} +...+ \theta_q\epsilon_{t-q}
                = \mu + \theta(L)\epsilon_t

        can be rewritten as a linear combination of its past
        values {y_{t-s}, s=1,2,...} plus the contemporaneous error
        term \epsilon_t, i.e.,

            y_t = \alpha + \sum_{s=1}^{\infty}\pi_s y_{t-s} + \epsilon_t

        for some \alpha and {\pi_j}, then the process is said
        to be \texit{invertible}.


        Returns
        -------
        isinvertible : boolean
            True if moving average roots are outside unit circle
        """
        return np.all(np.abs(self.maroots) > 1)

    def invertroots(self, retnew=False):
        """
        Make MA polynomial invertible by inverting roots inside unit circle

        Parameters
        ----------
        retnew : boolean
            If False (default), then return the lag-polynomial as array.
            If True, then return a new instance with invertible MA-polynomial

        Returns
        -------
        manew : array
            new invertible MA lag-polynomial, returned if retnew is false.
        wasinvertible : boolean
            True if the MA lag-polynomial was already invertible, returned if
            retnew is false.
        armaprocess : new instance of class
            If retnew is true, then return a new instance with invertible
            MA-polynomial
        """
        pr = self.maroots
        mainv = self.ma
        invertible = self.isinvertible
        if not invertible:
            pr[np.abs(pr) < 1] = 1. / pr[np.abs(pr) < 1]
            pnew = np.polynomial.Polynomial.fromroots(pr)
            mainv = pnew.coef / pnew.coef[0]

        if retnew:
            return self.__class__(self.ar, mainv, nobs=self.nobs)
            # TODO: dont do this multiple-return thing
        else:
            return (mainv, invertible)


class ARMAParams(object):

    @staticmethod
    def _unpack_params(params, order, k_trend, k_exog, reverse=False):
        (k_ar, k_ma) = order
        k = k_trend + k_exog
        maparams = params[k+k_ar:]
        arparams = params[k:k+k_ar]
        trend = params[:k_trend]
        exparams = params[k_trend:k]
        if reverse:
            arparams = arparams[::-1]
            maparams = maparams[::-1]
        return (trend, exparams, arparams, maparams)

    def _transparams(self, params):
        """Transforms params to induce stationarity/invertability.

        Reference
        ---------
        Jones(1980)
        """
        k_ar = self.k_ar
        k_ma = getattr(self, 'k_ma', 0)

        k = getattr(self, 'k_exog', 0) + self.k_trend

        # TODO: Use unpack_params?
        arparams = params[k:k+k_ar]  # TODO: Should we call this arcoefs?
        maparams = params[k+k_ar:]

        newparams = params.copy()

        if k != 0:
            # just copy exogenous parameters
            newparams[:k] = params[:k]

        if k_ar != 0:
            # AR Coeffs
            newparams[k:k+k_ar] = self._ar_transparams(arparams.copy())

        if k_ma != 0:
            # MA Coeffs
            newparams[k+k_ar:] = self._ma_transparams(maparams.copy())

        return newparams

    def _invtransparams(self, start_params):
        """
        Inverse of the Jones reparameterization
        """
        k_ar = self.k_ar
        k_ma = getattr(self, 'k_ma', 0)
        k = getattr(self, 'k_exog', 0) + self.k_trend

        arparams = start_params[k:k+k_ar]  # TODO: Should we call this arcoefs?
        maparams = start_params[k+k_ar:]

        newparams = start_params.copy()

        if k_ar != 0:
            # AR coeffs
            newparams[k:k+k_ar] = self._ar_invtransparams(arparams)

        if k_ma != 0:
            # MA coeffs
            newparams[k+k_ar:k+k_ar+k_ma] = self._ma_invtransparams(maparams)

        return newparams

    @staticmethod
    def _ar_transparams(params):
        """Transforms params to induce stationarity/invertability.

        Parameters
        ----------
        params : array
            The AR coefficients

        Reference
        ---------
        Jones(1980)
        """
        meparams = np.exp(-params)
        newparams = (1-meparams) / (1+meparams)
        tmp = (1-meparams) / (1+meparams)
        for j in range(1, len(params)):
            a = newparams[j]
            for kiter in range(j):
                tmp[kiter] -= a * newparams[j-kiter-1]
            newparams[:j] = tmp[:j]
        return newparams

    @staticmethod
    def _ar_invtransparams(params):
        """Inverse of the Jones reparameterization

        Parameters
        ----------
        params : array
            The transformed AR coefficients
        """
        # AR coeffs
        tmp = params.copy()
        for j in range(len(params)-1, 0, -1):
            a = params[j]
            for kiter in range(j):
                tmp[kiter] = (params[kiter] + a * params[j-kiter-1])/(1-a**2)
            params[:j] = tmp[:j]

        invarcoefs = -np.log((1-params)/(1+params))
        return invarcoefs

    @staticmethod
    def _ma_invtransparams(macoefs):
        """Inverse of the Jones reparameterization

        Parameters
        ----------
        params : array
            The transformed MA coefficients
        """
        tmp = macoefs.copy()
        for j in range(len(macoefs)-1, 0, -1):
            b = macoefs[j]
            for kiter in range(j):
                tmp[kiter] = (macoefs[kiter]-b * macoefs[j-kiter-1])/(1-b**2)
            macoefs[:j] = tmp[:j]

        invmacoefs = -np.log((1-macoefs)/(1+macoefs))
        return invmacoefs

    @staticmethod
    def _ma_transparams(params):
        """Transforms params to induce stationarity/invertability.

        Parameters
        ----------
        params : array
            The ma coeffecients of an (AR)MA model.

        Reference
        ---------
        Jones(1980)
        """
        meparams = np.exp(-params)
        newparams = (1-meparams) / (1+meparams)
        tmp = (1-meparams) / (1+meparams)

        # levinson-durbin to get macf
        for j in range(1, len(params)):
            b = newparams[j]
            for kiter in range(j):
                tmp[kiter] += b * newparams[j-kiter-1]
            newparams[:j] = tmp[:j]
        return newparams


# -----------------------------------------------------------------------

class VARRepresentation(_DimBase):
    """Class represents a known VAR(p) process, *without* any information
    about the distribution of error terms.

    Parameters
    ----------
    arcoefs : ndarray (p x k x k)
    macoefs : ndarray (q x k x k), optional
    intercept : ndarray (length k), optional

    Returns
    -------
    **Attributes**:
    """

    # TODO:
    # def orth_ma_rep
    # def svar_ma_rep
    # def isindependent --> property from varma_process
    # def isstructured --> property from varma_process

    # k_trend? # FIXME: fix docstring signature --> macoefs
    # TODO: clarify coefs vs params in arguments
    def __init__(self, arcoefs, macoefs=None, intercept=None):
        """

        Parameters
        ----------
        arcoefs : ndarray (p x k x k)

        intercept : ndarray (k x 1), optional

        """
        self.coefs = arcoefs  # for the VAR classes that inherit from this

        arcoefs = _shape_params(arcoefs)
        macoefs = _shape_params(macoefs)

        self.arcoefs = arcoefs
        self.macoefs = macoefs
        # TODO: i dont think we actually do anything useful with macoefs

        (k_ar, k_ma, neqs, intercept) = _unpack_lags_and_neqs(arcoefs,
                                                              macoefs,
                                                              intercept)

        self.intercept = intercept

    # TODO: maybe call this "arroots"?
    @cache_readonly
    def roots(self):
        arcoefs = self.arcoefs
        neqs = self.neqs
        k_ar = self.k_ar

        p = neqs * k_ar
        arr = np.zeros((p, p))
        arr[:neqs, :] = np.column_stack(arcoefs)
        arr[neqs:, :-neqs] = np.eye(p-neqs)
        roots = np.linalg.eig(arr)[0]**-1
        idx = np.argsort(np.abs(roots))[::-1]  # sort by reverse modulus
        return roots[idx]

    @property
    def _char_mat(self):
        arcoefs = self.arcoefs
        neqs = self.neqs
        return np.eye(neqs) - arcoefs.sum(axis=0)

    # TODO: ARIMA
    def long_run_effects(self):
        r"""Compute long-run effect of unit impulse

        .. math::

            \Psi_\infty = \sum_{i=0}^\infty \Phi_i
        """
        try:
            ret = scipy.linalg.inv(self._char_mat)
        except np.linalg.LinAlgError:  # TODO: warn?
            # Singular matrix
            ret = np.nan
        return ret

    def mean(self):
        r"""Mean of stable process

        Lutkepohl eq. 2.1.23

        .. math:: \mu = (I - A_1 - \dots - A_p)^{-1} \alpha
        """
        try:
            ret = np.linalg.solve(self._char_mat, self.intercept)
        except np.linalg.LinAlgError:  # TODO: warn?
            # Singular matrix
            ret = np.nan
        return ret

    def is_stable(self, verbose=False):
        """Determine stability of VAR(p) system by examining the eigenvalues
        of the VAR(1) representation

        Parameters
        ----------
        verbose : bool
            Print eigenvalues of the VAR(1) companion

        Returns
        -------
        is_stable : bool

        Notes
        -----
        Checks if det(I - Az) = 0 for any mod(z) <= 1, so all the
        eigenvalues of the companion matrix must lie outside the unit circle.
        """
        coefs = self.arcoefs

        from statsmodels.tsa.vector_ar.util import comp_matrix
        A_var1 = comp_matrix(coefs)
        eigs = np.linalg.eigvals(A_var1)  # TODO: cache?

        if verbose:
            print('Eigenvalues of VAR(1) rep')
            for val in np.abs(eigs):
                print(val)

        return (np.abs(eigs) <= 1).all()

    # TODO: scipy.signal.lfilter is very similar to this in the 1-dimensional
    # case.  Does it have an implementation we should be using for the
    # general case?
    # FIXME: If there are any ma coefficients already attached, they
    # are ignored
    def ma_rep(self, maxn=10):
        r"""MA(\infty) representation of VAR(p) process

        Parameters
        ----------
        coefs : ndarray (p x k x k)
        maxn : int
            Number of MA matrices to compute

        Notes
        -----
        VAR(p) process as

        .. math:: y_t = A_1 y_{t-1} + \ldots + A_p y_{t-p} + u_t

        can be equivalently represented as

        .. math:: y_t = \mu + \sum_{i=0}^\infty \Phi_i u_{t-i}

        e.g. can recursively compute the \Phi_i matrices with \Phi_0 = I_k

        Returns
        -------
        phis : ndarray (maxn + 1 x k x k)
        """
        arcoefs = self.arcoefs
        (p, k, k) = arcoefs.shape
        # macoefs = self.macoefs  # TODO: Wait, do we want macoefs or maparams?
        # if macoefs is not None and macoefs.shape[0] != 1:
        #    raise NotImplementedError

        phis = np.zeros((maxn+1, k, k))
        phis[0] = np.eye(k)

        # recursively compute Phi matrices
        for i in range(1, maxn+1):
            for j in range(1, i+1):
                if j > p:
                    break

                phis[i] += np.dot(phis[i-j], arcoefs[j-1])

        return phis
        # TODO: there's an analytical solution for this isnt there?
        # TODO: once we have the initial conditions, can we use an analytic
        # solution for the tail?


class ARMARepresentation(_DimBase, RootsMixin):

    # TODO: (optional)
    # quantecon.arma.set_params docstring is nice
    # def spectral_density(self) # See quantecon.arma
    # def autocorrelation(self)

    # TODO: I *really* don't think `nobs` should be an attribute here.
    @classmethod
    def from_coeffs(cls, arcoefs=None, macoefs=None, nobs=100):
        """Create instance from coefficients of the lag-polynomials

        Parameters
        ----------
        arcoefs : array-like
            Coefficient for autoregressive lag polynomial, not including zero
            lag. The sign is inverted to conform to the usual time series
            representation of an ARMA process in statistics. See the class
            docstring for more information.
        macoefs : array-like
            Coefficient for moving-average lag polynomial, including zero lag
        nobs : int, optional
            Length of simulated time series. Used, for example, if a sample
            is generated.

        Examples
        --------
        >>> arcoefs = [.75, -.25]
        >>> macoefs = [.65, .35]
        >>> arma_process = sm.tsa.ArmaProcess.from_coeffs(arcoefs, macoefs)
        >>> arma_process.isstationary
        True
        >>> arma_process.isinvertible
        True
        """
        if macoefs is None:
            macoefs = []
        if arcoefs is None:
            arcoefs = []
        arcoefs = np.asarray(arcoefs)
        macoefs = np.asarray(macoefs)
        inst = cls(np.r_[1, -arcoefs], np.r_[1, macoefs], nobs=nobs)
        return inst

    def __eq__(self, other):
        # Easier to check polynomials than coefficients
        # TODO: I'd rather `nobs` not be an attribute at this level.
        return (self.arpoly == other.arpoly and
                self.mapoly == other.mapoly and
                getattr(self, 'nobs', None) == getattr(other, 'nobs', None))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __mul__(self, other):
        if issubclass(other.__class__, ARMARepresentation):
            ar = (self.arpoly * other.arpoly).coef
            ma = (self.mapoly * other.mapoly).coef
        elif not isinstance(other, (list, tuple)) or len(other) != 2:
            raise TypeError(other)
        else:
            (aroth, maoth) = other
            arpolyoth = np.polynomial.Polynomial(aroth)
            mapolyoth = np.polynomial.Polynomial(maoth)
            ar = (self.arpoly * arpolyoth).coef
            ma = (self.mapoly * mapolyoth).coef

        product = self.__class__(ar, ma, nobs=self.nobs)
        return product

    def __repr__(self):
        arlist = self.ar.tolist()
        malist = self.ma.tolist()
        cname = self.__class__.__name__
        nobs = getattr(self, 'nobs', None)
        if nobs is not None:
            msg = '{0}(AR: {1}, MA: {2}, nobs={3}) at {4}'
            return msg.format(cname, arlist, malist, nobs, hex(id(self)))
        else:
            msg = '%{0}(AR: {1}, MA: {2}) at {3}'
            return msg.format(cname, arlist, malist, hex(id(self)))

    def __str__(self):
        arlist = self.ar.tolist()
        malist = self.ma.tolist()
        cname = self.__class__.__name__
        return '{0}\nAR: {1}\nMA: {2}'.format(cname, arlist, malist)

    # TODO: Make the inputs fooparams or foocoefs; don't pile on another
    # naming scheme.
    def __init__(self, ar, ma, intercept=0, nobs=None):
        # The inputs ar and ma are array-like in the form of lag
        # polynomials, i.e. always start with a [1]
        self.ar = np.asarray(ar)
        self.ma = np.asarray(ma)
        _check_is_poly(self.ar)
        _check_is_poly(self.ma)

        self.arcoefs = -self.ar[1:]
        self.macoefs = self.ma[1:]

        (k_ar, k_ma, neqs, intercept) = _unpack_lags_and_neqs(self.ar,
                                                              self.ma,
                                                              intercept)
        self.intercept = intercept

        if neqs != 1:
            msg = 'Lag Polynomials for the vector case not implemented.'
            raise NotImplementedError(msg)

        self.arpoly = np.polynomial.Polynomial(self.ar)
        self.mapoly = np.polynomial.Polynomial(self.ma)

        # TOOD: I'd rather this not be an attribute at this juncture.
        self.nobs = nobs

    # TODO: Should this return a ARMARepresentation object instead of
    # just an array?
    @deprecate_kwarg(old_arg_name='nobs', new_arg_name='lags')
    def arma2ma(self, lags=None):
        """get the impulse response function (MA representation) for
        ARMA process

        Parameters
        ----------
        ma : array_like, 1d
            moving average lag polynomial
        ar : array_like, 1d
            auto regressive lag polynomial
        lags : int
            number of observations to calculate

        Returns
        -------
        ir : array, 1d
            impulse response function with nobs elements

        Notes
        -----
        This is the same as finding the MA representation of an ARMA(p,q).
        By reversing the role of ar and ma in the function arguments, the
        returned result is the AR representation of an ARMA(p,q), i.e

        ma_representation = arma_impulse_response(ar, ma, lags=100)
        ar_representation = arma_impulse_response(ma, ar, lags=100)

        fully tested against matlab

        Examples
        --------
        AR(1)

        >>> arma_impulse_response([1.0, -0.8], [1.], lags=10)
        array([ 1.        ,  0.8       ,  0.64      ,  0.512     ,  0.4096    ,
                0.32768   ,  0.262144  ,  0.2097152 ,  0.16777216,  0.13421773])

        this is the same as

        >>> 0.8**np.arange(10)
        array([ 1.        ,  0.8       ,  0.64      ,  0.512     ,  0.4096    ,
                0.32768   ,  0.262144  ,  0.2097152 ,  0.16777216,  0.13421773])

        MA(2)

        >>> arma_impulse_response([1.0], [1., 0.5, 0.2], lags=10)
        array([ 1. ,  0.5,  0.2,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ])

        ARMA(1,2)

        >>> arma_impulse_response([1.0, -0.8], [1., 0.5, 0.2], lags=10)
        array([ 1.        ,  1.3       ,  1.24      ,  0.992     ,  0.7936    ,
                0.63488   ,  0.507904  ,  0.4063232 ,  0.32505856,  0.26004685])
        """
        ar = self.ar
        ma = self.ma
        _check_is_poly(np.array(ar))  # TODO: get rid of redundant checking
        _check_is_poly(np.array(ma))
        impulse = np.zeros(lags)
        impulse[0] = 1.0
        return scipy.signal.lfilter(ma, ar, impulse)

    # TODO: Should this return a ARMARepresentation object instead
    # of just an array?
    @deprecate_kwarg(old_arg_name='nobs', new_arg_name='lags')
    def arma2ar(self, lags):
        """get the AR representation of an ARMA process

        Parameters
        ----------
        ar : array_like, 1d
            auto regressive lag polynomial
        ma : array_like, 1d
            moving average lag polynomial
        lags : int
            number of observations to calculate

        Returns
        -------
        ar : array, 1d
            coefficients of AR lag polynomial with nobs elements

        Notes
        -----
        This is an alias for
        ``ar_representation = arma_impulse_response(ma, ar, lags=100)``
        which has been fully tested against MATLAB.

        Examples
        --------
        """
        # Switches the order of the arguments
        other = ARMARepresentation(self.ma, self.ar)
        return other.arma2ma(lags)
        # TODO: is it really that easy?

    # TODO: clarify worN arg?
    def arma_periodogram(self, worN=None, whole=0):
        """

        This might be more accurately referred to as a Spectral Density.
        Spectral Density refers to the population function, while Periodiogram
        refers to the sample estimate of the Spectral Density [citation needed]

        Periodogram for ARMA process given by lag-polynomials ar and ma

        Let $\phi$ and $\theta$ be lag polynomials so that we can write
        our ARMA process as:

        \phi(L)x_t = \theta(L)\epsilon_t

        The spectral density function $f(\omega)$ of this process is then:

        f(\omega) = \sigma^2_{\epsilon} \frac{|\theta(e^{-2\pi i \omega})|^2}{|\phi(e^{-2\pi i \omega})|^2}

        # TODO: Reference for the claim above

        Parameters
        ----------
        ar : array_like
            autoregressive lag-polynomial with leading 1 and lhs sign
        ma : array_like
            moving average lag-polynomial with leading 1
        worN : {None, int}, optional
            option for scipy.signal.freqz   (read "w or N")
            If None, then compute at 512 frequencies around the unit circle.
            If a single integer, the compute at that many frequencies.
            Otherwise, compute the response at frequencies given in worN
        whole : {0,1}, optional
            options for scipy.signal.freqz
            Normally, frequencies are computed from 0 to pi (upper-half of
            unit-circle.  If whole is non-zero compute frequencies from 0
            to 2*pi.

        Returns
        -------
        w : array
            frequencies
        sd : array
            periodogram, spectral density

        Notes
        -----
        Normalization ?

        This uses signal.freqz, which does not use fft. There is a fft
        version somewhere.
        """
        ar = self.ar
        ma = self.ma
        _check_is_poly(np.array(ar))  # TODO: get rid of redundant checking
        _check_is_poly(np.array(ma))
        (w, h) = scipy.signal.freqz(ma, ar, worN=worN, whole=whole)
        sd = np.abs(h)**2/np.sqrt(2*np.pi)
        # TODO: is this normalization standard in the literature?
        # Fourier Transforms are like armpits...
        if np.isnan(h).any():
            # This happens with unit root or seasonal unit root
            msg = 'nan in frequency response h, may be a unit root'
            warnings.warn(msg)
        return (w, sd)


def arma_periodogram(ar, ma, worN=None, whole=0):
    arma = ARMARepresentation(ar, ma)
    return arma.arma_periodogram(worN=worN, whole=whole)


# TODO: not hit in tests
@deprecate_kwarg(old_arg_name='nobs', new_arg_name='lags')
def arma2ar(ar, ma, lags):
    arma = ARMARepresentation(ar, ma)
    return arma.arma2ar(lags)


def arma2ma(ar, ma, lags):
    arma = ARMARepresentation(ar, ma)
    return arma.arma2ma(lags)


@deprecate_kwarg(old_arg_name='nobs', new_arg_name='leads')
def arma_impulse_response(ar, ma, leads=100):
    """alias for arma2ma"""
    return arma2ma(ar, ma, lags=leads)
