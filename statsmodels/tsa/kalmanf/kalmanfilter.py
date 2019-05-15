"""
State Space Analysis using the Kalman Filter

References
-----------
Durbin., J and Koopman, S.J.  `Time Series Analysis by State Space Methods`.
    Oxford, 2001.

Hamilton, J.D.  `Time Series Analysis`.  Princeton, 1994.

Harvey, A.C. `Forecasting, Structural Time Series Models and the Kalman Filter`.
    Cambridge, 1989.

Notes
-----
This file follows Hamilton's notation pretty closely.
The ARMA Model class follows Durbin and Koopman notation.
Harvey uses Durbin and Koopman notation.
"""  # noqa:E501
# Anderson and Moore `Optimal Filtering` provides a more efficient algorithm
# namely the information filter
# if the number of series is much greater than the number of states
# e.g., with a DSGE model.  See Also
# http://www.federalreserve.gov/pubs/oss/oss4/aimindex.html
# Harvey notes that the square root filter will keep P_t pos. def. but
# is not strictly needed outside of the engineering (long series)
from __future__ import print_function
import numpy as np

from . import kalman_loglike

# Fast filtering and smoothing for multivariate state space models
# and The Riksbank -- Strid and Walentin (2008)
# Block Kalman filtering for large-scale DSGE models
# but this is obviously macro model specific


class KalmanFilter(object):
    r"""
    Kalman Filter code intended for use with the ARMA model.

    Notes
    -----
    The notation for the state-space form follows Durbin and Koopman (2001).

    The observation equations is

    .. math:: y_{t} = Z_{t}\alpha_{t} + \epsilon_{t}

    The state equation is

    .. math:: \alpha_{t+1} = T_{t}\alpha_{t} + R_{t}\eta_{t}

    For the present purposed \epsilon_{t} is assumed to always be zero.
    """

    @classmethod
    def T(cls, params, r, k, p):  # F in Hamilton
        """
        The coefficient matrix for the state vector in the state equation.

        Its dimension is r+k x r+k.

        Parameters
        ----------
        r : int
            In the context of the ARMA model r is max(p,q+1) where p is the
            AR order and q is the MA order.
        k : int
            The number of exogenous variables in the ARMA model, including
            the constant if appropriate.
        p : int
            The AR coefficient in an ARMA model.

        References
        ----------
        Durbin and Koopman Section 3.7.
        """
        arr = np.zeros((r, r), dtype=params.dtype, order="F")
        # allows for complex-step derivative
        params_padded = np.zeros(r, dtype=params.dtype,
                                 order="F")
        # handle zero coefficients if necessary
        # NOTE: squeeze added for cg optimizer
        params_padded[:p] = params[k:p + k]
        arr[:, 0] = params_padded
        # first p params are AR coeffs w/ short params
        arr[:-1, 1:] = np.eye(r - 1)
        return arr

    @classmethod
    def R(cls, params, r, k, q, p):  # R is H in Hamilton
        """
        The coefficient matrix for the state vector in the observation
        equation.

        Its dimension is r+k x 1.

        Parameters
        ----------
        r : int
            In the context of the ARMA model r is max(p,q+1) where p is the
            AR order and q is the MA order.
        k : int
            The number of exogenous variables in the ARMA model, including
            the constant if appropriate.
        q : int
            The MA order in an ARMA model.
        p : int
            The AR order in an ARMA model.

        References
        ----------
        Durbin and Koopman Section 3.7.
        """
        arr = np.zeros((r, 1), dtype=params.dtype, order="F")
        # this allows zero coefficients
        # dtype allows for compl. der.
        arr[1:q + 1, :] = params[p + k:p + k + q][:, None]
        arr[0] = 1.0
        return arr

    @classmethod
    def Z(cls, r):
        """
        Returns the Z selector matrix in the observation equation.

        Parameters
        ----------
        r : int
            In the context of the ARMA model r is max(p,q+1) where p is the
            AR order and q is the MA order.

        Notes
        -----
        Currently only returns a 1 x r vector [1,0,0,...0].  Will need to
        be generalized when the Kalman Filter becomes more flexible.
        """
        arr = np.zeros((1, r), order="F")
        arr[:, 0] = 1.
        return arr

    @classmethod
    def geterrors(cls, y, k, k_ar, k_ma, k_lags, nobs, Z_mat, m, R_mat, T_mat,
                  paramsdtype):
        """
        Returns just the errors of the Kalman Filter
        """
        if np.issubdtype(paramsdtype, np.float64):
            return kalman_loglike.kalman_filter_double(
                y, k, k_ar, k_ma, k_lags, int(nobs), Z_mat, R_mat, T_mat)[0]
        elif np.issubdtype(paramsdtype, np.complex128):
            return kalman_loglike.kalman_filter_complex(
                y, k, k_ar, k_ma, k_lags, int(nobs), Z_mat, R_mat, T_mat)[0]
        else:
            raise TypeError("dtype %s is not supported "
                            "Please file a bug report" % paramsdtype)

    @classmethod
    def _init_kalman_state(cls, params, arma_model):
        """
        Returns the system matrices and other info needed for the
        Kalman Filter recursions
        """
        paramsdtype = params.dtype
        y = arma_model.endog.copy().astype(paramsdtype)
        k = arma_model.k_exog + arma_model.k_trend
        nobs = arma_model.nobs
        k_ar = arma_model.k_ar
        k_ma = arma_model.k_ma
        k_lags = arma_model.k_lags

        if arma_model.transparams:
            newparams = arma_model._transparams(params)
        else:
            newparams = params  # don't need a copy if not modified.

        if k > 0:
            y -= np.dot(arma_model.exog, newparams[:k])

        # system matrices
        Z_mat = cls.Z(k_lags)
        m = Z_mat.shape[1]  # r
        R_mat = cls.R(newparams, k_lags, k, k_ma, k_ar)
        T_mat = cls.T(newparams, k_lags, k, k_ar)
        return (y, k, nobs, k_ar, k_ma, k_lags,
                newparams, Z_mat, m, R_mat, T_mat, paramsdtype)

    @classmethod
    def loglike(cls, params, arma_model, set_sigma2=True):
        """
        The loglikelihood for an ARMA model using the Kalman Filter recursions.

        Parameters
        ----------
        params : array
            The coefficients of the ARMA model, assumed to be in the order of
            trend variables and `k` exogenous coefficients, the `p` AR
            coefficients, then the `q` MA coefficients.
        arma_model : `statsmodels.tsa.arima.ARMA` instance
            A reference to the ARMA model instance.
        set_sigma2 : bool, optional
            True if arma_model.sigma2 should be set.
            Note that sigma2 will be computed in any case,
            but it will be discarded if set_sigma2 is False.

        Notes
        -----
        This works for both real valued and complex valued parameters. The
        complex values being used to compute the numerical derivative. If
        available will use a Cython version of the Kalman Filter.
        """
        # TODO: see section 3.4.6 in Harvey for computing the derivatives in
        # the recursion itself.
        # TODO: this won't work for time-varying parameters
        (y, k, nobs, k_ar, k_ma, k_lags, newparams, Z_mat, m, R_mat, T_mat,
         paramsdtype) = cls._init_kalman_state(params, arma_model)
        if np.issubdtype(paramsdtype, np.float64):
            loglike, sigma2 = kalman_loglike.kalman_loglike_double(
                y, k, k_ar, k_ma, k_lags, int(nobs),
                Z_mat, R_mat, T_mat)
        elif np.issubdtype(paramsdtype, np.complex128):
            loglike, sigma2 = kalman_loglike.kalman_loglike_complex(
                y, k, k_ar, k_ma, k_lags, int(nobs),
                Z_mat.astype(complex), R_mat, T_mat)
        else:
            raise TypeError("This dtype %s is not supported "
                            " Please files a bug report." % paramsdtype)
        if set_sigma2:
            arma_model.sigma2 = sigma2

        return loglike
