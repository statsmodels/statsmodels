"""
Vector Autoregression (VAR) processes

References
----------
Lütkepohl (2005) New Introduction to Multiple Time Series Analysis
"""

from statsmodels.compat.pandas import deprecate_kwarg

import warnings

import numpy as np
import numpy.linalg as npl
from numpy.linalg import slogdet

from statsmodels.tools.numdiff import approx_hess
from statsmodels.tools.rng_qrng import check_random_state
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tools.validation import string_like
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.vector_ar import util
from statsmodels.tsa.vector_ar.hypothesis_test_results import ErrorBand
from statsmodels.tsa.vector_ar.irf import IRAnalysis
from statsmodels.tsa.vector_ar.var_model import (
    FEVD,
    VAR,
    VARProcess,
    VARResults,
    forecast,
)


def svar_ckerr(svar_type, A, B):
    if A is None and svar_type in {"A", "AB"}:
        raise ValueError("SVAR of type A or AB but A array not given.")
    if B is None and svar_type in {"B", "AB"}:

        raise ValueError("SVAR of type B or AB but B array not given.")


class SVAR(tsbase.TimeSeriesModel):
    r"""
    Fit VAR and then estimate structural components of A and B, defined:

    .. math:: Ay_t = A_1 y_{t-1} + \ldots + A_p y_{t-p} + B \varepsilon_t

    Parameters
    ----------
    endog : array_like
        2-d endogenous response variable, nobs x neqs.
    svar_type : {"A", "B", "AB"}
        "A" - estimate structural parameters of A matrix, B assumed = I
        "B" - estimate structural parameters of B matrix, A assumed = I
        "AB" - estimate structural parameters indicated in both A and B matrix

        .. versionchanged:: 0.15.1

           "E" entries in the matrix that svar_type does not estimate raise a
           ValueError. For "B" with "E" in A, fit raised an error or
           estimated another model, in which the entries of A were estimated
           too and those of B were tied to one value; for "A" with "E" in B,
           fit raised an error.
    dates : array_like, optional
        must match number of rows of endog
    freq : str, optional
        The frequency of the time-series. A Pandas offset or 'B', 'D', 'W',
        'M', 'A', or 'Q'. This is optional if dates are given.
    A : ndarray, optional
        neqs x neqs array with the unknown parameters marked "E" (or "e") for
        estimate and numbers (or strings of numbers) for the fixed entries.
        Required if `svar_type` is "A" or "AB". The identity if not given.
    B : ndarray, optional
        neqs x neqs array with the unknown parameters marked "E" (or "e") for
        estimate and numbers (or strings of numbers) for the fixed entries.
        Required if `svar_type` is "B" or "AB". The identity if not given.
    missing : {"none", "drop", "raise"}, optional
        Available options are 'none', 'drop', and 'raise'. If 'none', no nan
        checking is done. If 'drop', any observations with nans are dropped.
        If 'raise', an error is raised. Default is 'none'.

    References
    ----------
    Hamilton (1994) Time Series Analysis
    """

    def __init__(
        self, endog, svar_type, dates=None, freq=None, A=None, B=None, missing="none"
    ):
        super().__init__(endog, None, dates, freq, missing=missing)
        # (self.endog, self.names,
        # self.dates) = data_util.interpret_data(endog, names, dates)

        self.neqs = self.endog.shape[1]

        svar_type = string_like(
            svar_type, "svar_type", options=("A", "B", "AB"), lower=False
        )
        self.svar_type = svar_type

        svar_ckerr(svar_type, A, B)

        self.A_original = A
        self.B_original = B

        # Set by fit()/_estimate_svar(), once the lag order is known
        self.nobs = None
        self.sigma_u = None

        # initialize A, B as I if not given
        # Initialize SVAR masks
        if A is None:
            A = np.identity(self.neqs)
            self.A_mask = A_mask = np.zeros(A.shape, dtype=bool)
        else:
            A = A.astype("U")
            A_mask = np.logical_or(A == "E", A == "e")
            self.A_mask = A_mask
        if B is None:
            B = np.identity(self.neqs)
            self.B_mask = B_mask = np.zeros(B.shape, dtype=bool)
        else:
            B = B.astype("U")
            B_mask = np.logical_or(B == "E", B == "e")
            self.B_mask = B_mask
        # Only the matrices that svar_type names are estimated
        for name, mask in (("A", A_mask), ("B", B_mask)):
            if name not in svar_type and mask.any():
                raise ValueError(
                    f'svar_type="{svar_type}" does not estimate {name}, but '
                    f'{name} has entries "E" to estimate. Use svar_type="AB" to '
                    "estimate A and B."
                )

        # convert A and B to numeric
        # TODO: change this when masked support is better or with formula
        # integration
        Anum = np.zeros(A.shape, dtype=float)
        Anum[~A_mask] = A[~A_mask]
        Anum[A_mask] = np.nan
        self.A = Anum

        Bnum = np.zeros(B.shape, dtype=float)
        Bnum[~B_mask] = B[~B_mask]
        Bnum[B_mask] = np.nan
        self.B = Bnum

        # LikelihoodModel.__init__(self, endog)

        # super().__init__(endog)

    def fit(
        self,
        A_guess=None,
        B_guess=None,
        maxlags=None,
        method="ols",
        ic=None,
        trend="c",
        verbose=False,
        s_method="mle",
        solver="bfgs",
        override=False,
        maxiter=500,
        maxfun=None,
    ):
        """
        Fit the SVAR model and solve for structural parameters

        Parameters
        ----------
        A_guess : array_like, optional
            A vector of starting values for all parameters to be estimated
            in A, in the row-major order of the "E" entries of A. Defaults
            to 0.1 for every parameter.
        B_guess : array_like, optional
            A vector of starting values for all parameters to be estimated
            in B, in the row-major order of the "E" entries of B. Defaults
            to 0.1 for every parameter.
        maxlags : int, optional
            The number of lags of the VAR, 1 if not given. If `ic` is given,
            the maximum number of lags to check for order selection.
        method : {'ols'}, optional
            Estimation method to use

            .. deprecated:: 0.15.1

               Other values were ignored and the VAR was estimated by OLS.
               They issue a FutureWarning and will raise a ValueError after
               statsmodels 0.17 is released.
        ic : {'aic', 'fpe', 'hqic', 'bic', None}, optional
            Information criterion to use for VAR order selection.
            aic : Akaike
            fpe : Final prediction error
            hqic : Hannan-Quinn
            bic : Bayesian a.k.a. Schwarz

            .. versionchanged:: 0.15.1

               The order is selected by VAR.select_order with `maxlags` and
               `trend`. It raised an AttributeError.
        verbose : bool, optional
            Print order selection output to the screen
        trend : {"c", "ct", "ctt", "n"}, optional
            "c" - add constant
            "ct" - constant and trend
            "ctt" - constant, linear and quadratic trend
            "n" - no constant, no trend
            Note that these are prepended to the columns of the dataset.
        s_method : {'mle'}, optional
            Estimation method for structural parameters

            .. deprecated:: 0.15.1

               Other values were ignored and the structural parameters were
               estimated by maximum likelihood. They issue a FutureWarning
               and will raise a ValueError after statsmodels 0.17 is
               released.
        solver : {'bfgs', 'newton', 'nm', 'lbfgs', 'powell', 'cg', 'ncg', 'basinhopping', 'minimize'}, optional
            The optimizer that maximizes the log-likelihood of the structural
            parameters, see the method argument of
            statsmodels.base.model.LikelihoodModel.fit. The default is
            'bfgs'.
        override : bool, optional
            If True, returns estimates of A and B without checking
            order or rank condition

            .. versionchanged:: 0.15.1

               The rank condition uses the derivatives of the covariance of
               the model, A^-1 B B' A^-1', at the starting values. The
               derivatives with respect to B and the covariance were wrong,
               so some identified models raised and some models that are
               not identified, with free entries at the same position in A
               and B, were estimated.
        maxiter : int, optional
            Number of iterations to perform in solution method
        maxfun : int, optional
            Unused.

            .. deprecated:: 0.15.1

               maxfun was ignored. Passing it issues a FutureWarning, and it
               will be removed after statsmodels 0.17 is released. Use
               maxiter.

        Returns
        -------
        est : SVARResults

        Raises
        ------
        ValueError
            If the length of `A_guess` or `B_guess` is not the number of
            parameters to estimate in A or B, or, unless `override` is True,
            if the order or rank condition for identification is not met.

        Notes
        -----
        Lütkepohl pp. 146-153
        Hamilton pp. 324-336

        .. versionchanged:: 0.15.1

           The results hold their own A and B. They were the arrays of the
           model, so a later fit, or a call of loglike, score or hessian,
           changed the A and B, and the structural responses, of earlier
           results.
        """
        if method is not None and not (
            isinstance(method, str) and method.lower() == "ols"
        ):
            warnings.warn(
                f"method={method!r} is ignored and the VAR is estimated by OLS. "
                'Values other than "ols" are deprecated and will raise a '
                "ValueError after statsmodels 0.17 is released.",
                FutureWarning,
                stacklevel=2,
            )
        if s_method is not None and not (
            isinstance(s_method, str) and s_method.lower() == "mle"
        ):
            warnings.warn(
                f"s_method={s_method!r} is ignored and the structural "
                "parameters are estimated by maximum likelihood. Values other "
                'than "mle" are deprecated and will raise a ValueError after '
                "statsmodels 0.17 is released.",
                FutureWarning,
                stacklevel=2,
            )
        if maxfun is not None:
            warnings.warn(
                "maxfun is ignored and deprecated, and will be removed after "
                "statsmodels 0.17 is released. Use maxiter.",
                FutureWarning,
                stacklevel=2,
            )
        lags = maxlags
        if ic is not None:
            selections = VAR(self.endog).select_order(maxlags=maxlags, trend=trend)
            if ic not in selections.selected_orders:
                raise ValueError(
                    f"{ic} not recognized, must be among "
                    f"{sorted(selections.selected_orders)}"
                )
            lags = selections.selected_orders[ic]
            if verbose:
                print(selections)
                print(f"Using {lags:d} based on {ic} criterion")
        elif lags is None:
            lags = 1

        self.nobs = len(self.endog) - lags

        # initialize starting parameters
        start_params = self._get_init_params(A_guess, B_guess)

        return self._estimate_svar(
            start_params,
            lags,
            trend=trend,
            solver=solver,
            override=override,
            maxiter=maxiter,
            maxfun=maxfun,
        )

    def _get_init_params(self, A_guess, B_guess):
        """Returns either the given starting values or .1 if none are given"""

        var_type = self.svar_type.lower()

        n_masked_a = self.A_mask.sum()
        if var_type in ["ab", "a"]:
            if A_guess is None:
                A_guess = np.array([0.1] * n_masked_a)
            elif len(A_guess) != n_masked_a:
                msg = "len(A_guess) = %s, there are %s parameters in A"
                raise ValueError(msg % (len(A_guess), n_masked_a))
        else:
            A_guess = []

        n_masked_b = self.B_mask.sum()
        if var_type in ["ab", "b"]:
            if B_guess is None:
                B_guess = np.array([0.1] * n_masked_b)
            elif len(B_guess) != n_masked_b:
                msg = "len(B_guess) = %s, there are %s parameters in B"
                raise ValueError(msg % (len(B_guess), n_masked_b))
        else:
            B_guess = []

        return np.r_[A_guess, B_guess]

    def _estimate_svar(
        self,
        start_params,
        lags,
        maxiter,
        maxfun,
        trend="c",
        solver="nm",
        override=False,
    ):
        """
        Estimate the reduced-form VAR and then solve for structural params

        Parameters
        ----------
        start_params : array_like
            Starting values for the parameters to be estimated.
        lags : int
            Number of lags in the VAR.
        maxiter : int
            Number of iterations to perform in solution method.
        maxfun : int
            Number of function evaluations to perform.
        trend : {"n", "c", "ct", "ctt"}, optional
            As per above
        solver : str, optional
            Solution method
        override : bool, optional
            If True, returns estimates of A and B without checking
            order or rank condition
        """
        k_trend = util.get_trendorder(trend)
        y = self.endog
        z = util.get_var_endog(y, lags, trend=trend, has_constant="raise")
        y_sample = y[lags:]

        # Lutkepohl p75, about 5x faster than stated formula
        var_params = np.linalg.lstsq(z, y_sample, rcond=-1)[0]
        resid = y_sample - np.dot(z, var_params)

        # Unbiased estimate of covariance matrix $\Sigma_u$ of the white noise
        # process $u$
        # equivalent definition
        # .. math:: \frac{1}{T - Kp - 1} Y^\prime (I_T - Z (Z^\prime Z)^{-1}
        # Z^\prime) Y
        # Ref: Lutkepohl p.75
        # df_resid right now is T - Kp - 1, which is a suggested correction

        avobs = len(y_sample)

        df_resid = avobs - (self.neqs * lags + k_trend)

        sse = np.dot(resid.T, resid)
        # TODO: should give users the option to use a dof correction or not
        omega = sse / df_resid
        self.sigma_u = omega

        A, B = self._solve_AB(
            start_params, override=override, solver=solver, maxiter=maxiter
        )
        A_mask = self.A_mask
        B_mask = self.B_mask

        return SVARResults(
            y,
            z,
            var_params,
            omega,
            lags,
            names=self.endog_names,
            trend=trend,
            dates=self.data.dates,
            model=self,
            A=A,
            B=B,
            A_mask=A_mask,
            B_mask=B_mask,
        )

    def loglike(self, params):
        """
        Loglikelihood for SVAR model

        Parameters
        ----------
        params : array_like
            The concatenated unknown parameters of the A and B matrices, the
            "E" entries of A and then those of B, each in row-major order.

        Returns
        -------
        float
            The log-likelihood of the structural parameters, concentrated
            with respect to the autoregressive parameters.

        Notes
        -----
        This method assumes that the autoregressive parameters are
        first estimated, then likelihood with structural parameters
        is estimated. It uses the residual covariance and the number of
        observations of the last call to `fit`.

        .. versionchanged:: 0.15.1

           The term of B is log(det(B) ** 2). It was the log of the
           determinant of the elementwise square of B, which differs when B
           is not triangular, so the log-likelihood and the estimates of
           such models were wrong.
        """

        # TODO: this does not look robust if A or B is None
        A = self.A.copy()
        B = self.B.copy()
        A_mask = self.A_mask
        B_mask = self.B_mask
        A_len = len(A[A_mask])
        B_len = len(B[B_mask])

        A[A_mask] = params[:A_len]
        B[B_mask] = params[A_len : A_len + B_len]

        nobs = self.nobs
        neqs = self.neqs
        sigma_u = self.sigma_u

        W = np.dot(npl.inv(B), A)
        trc_in = np.dot(np.dot(W.T, W), sigma_u)
        # log(det(A) ** 2) and log(det(B) ** 2)
        a_logdet = 2 * slogdet(A)[1]
        b_logdet = 2 * slogdet(B)[1]

        likl = (
            -nobs
            / 2.0
            * (neqs * np.log(2 * np.pi) - a_logdet + b_logdet + np.trace(trc_in))
        )

        return likl

    def score(self, AB_mask):
        r"""
        Return the gradient of the loglike at AB_mask

        Parameters
        ----------
        AB_mask : ndarray
            Concatenated unknown values of the A and B matrices, in the order
            of `loglike`.

        Returns
        -------
        ndarray
            The gradient, 1-d with one entry per unknown value.

        Notes
        -----
        The gradient of the log-likelihood with respect to A and B is

        .. math::

           n (A^{-T} - B^{-T} W \Sigma_u), \quad
           n B^{-T} (W \Sigma_u W^T - I), \quad W = B^{-1} A,

        of which the entries of the unknown values are returned.

        .. versionchanged:: 0.15.1

           The gradient is computed analytically. It was computed with
           forward differences with a step of 1e-8, whose error (about 1e-5
           relative for the entries of a B of order 0.01) exceeded the
           tolerance of the solvers of `fit` at the maximum: the default
           "bfgs" solver could warn that the optimization did not converge,
           depending on rounding, and "newton" stopped about 3e-5 (relative)
           from the maximum. The estimates of `fit` change accordingly.
        """
        params = np.asarray(AB_mask, dtype=float).ravel()
        A = np.array(self.A, dtype=float)
        B = np.array(self.B, dtype=float)
        A_len = int(self.A_mask.sum())
        A[self.A_mask] = params[:A_len]
        B[self.B_mask] = params[A_len:]

        inv_A = npl.inv(A)
        inv_B = npl.inv(B)
        W = inv_B @ A
        sigma_u = self.sigma_u
        nobs = self.nobs
        grad_A = nobs * (inv_A.T - inv_B.T @ W @ sigma_u)
        grad_B = nobs * inv_B.T @ (W @ sigma_u @ W.T - np.eye(self.neqs))
        return np.r_[grad_A[self.A_mask], grad_B[self.B_mask]]

    def hessian(self, AB_mask):
        """
        Returns numerical hessian

        Parameters
        ----------
        AB_mask : ndarray
            Concatenated unknown values of the A and B matrices, in the order
            of `loglike`.

        Returns
        -------
        ndarray
            The Hessian of `loglike`, square with one row per unknown value.
        """
        loglike = self.loglike
        if AB_mask.ndim > 1:
            AB_mask = AB_mask.ravel()
        return approx_hess(AB_mask, loglike)

    def _solve_AB(self, start_params, maxiter, override=False, solver="bfgs"):
        """
        Solves for MLE estimate of structural parameters

        Parameters
        ----------
        start_params : array_like
            Starting values for the parameters to be estimated.
        maxiter : int
            The maximum number of iterations.
        override : bool, optional
            If True, returns estimates of A and B without checking
            order or rank condition
        solver : str, optional
            Solver to be used. The default is 'bfgs'. Other
            choices are 'nm' (Nelder-Mead), 'newton' (Newton-Raphson), 'cg'
            conjugate, 'ncg' (non-conjugate gradient), and 'powell'.

        Returns
        -------
        A_solve : ndarray
            ML solution for the A matrix.
        B_solve : ndarray
            ML solution for the B matrix.
        """
        # TODO: this could stand a refactor
        A_mask = self.A_mask
        B_mask = self.B_mask
        A = self.A.copy()
        B = self.B.copy()
        A_len = len(A[A_mask])

        A[A_mask] = start_params[:A_len]
        B[B_mask] = start_params[A_len:]

        if not override:
            J = self._compute_J(A, B)
            self.check_order(J)
            self.check_rank(J)
        else:  # TODO: change to a warning?
            print("Order/rank conditions have not been checked")

        mlefit = super().fit(
            start_params=start_params,
            method=solver,
            maxiter=maxiter,
            disp=False,
        )
        # used by sirf_errband_mc to count the fits that did not converge
        self.mle_retvals = mlefit.mle_retvals
        retvals = mlefit.params
        A[A_mask] = retvals[:A_len]
        B[B_mask] = retvals[A_len:]

        return A, B

    def _compute_J(self, A_solve, B_solve):

        # first compute appropriate duplication matrix
        # taken from Magnus and Neudecker (1980),
        # "The Elimination Matrix: Some Lemmas and Applications
        # the creation of the D_n matrix follows MN (1980) directly,
        # while the rest follows Hamilton (1994)

        neqs = self.neqs
        A_mask = self.A_mask
        B_mask = self.B_mask

        # first generate duplication matrix, see MN (1980) for notation

        D_nT = np.zeros([int((1.0 / 2) * (neqs) * (neqs + 1)), neqs**2])

        for j in range(neqs):
            i = j
            while j <= i < neqs:
                u = np.zeros([int((1.0 / 2) * neqs * (neqs + 1)), 1])
                u[int(j * neqs + (i + 1) - (1.0 / 2) * (j + 1) * j - 1)] = 1
                Tij = np.zeros([neqs, neqs])
                Tij[i, j] = 1
                Tij[j, i] = 1
                D_nT = D_nT + np.dot(u, (Tij.ravel("F")[:, None]).T)
                i = i + 1

        D_n = D_nT.T
        D_pl = npl.pinv(D_n)

        # generate S_B
        S_B = np.zeros((neqs**2, len(A_solve[A_mask])))
        S_D = np.zeros((neqs**2, len(B_solve[B_mask])))

        j = 0
        j_d = 0
        if len(A_solve[A_mask]) != 0:
            A_vec = np.ravel(A_mask, order="F")
            for k in range(neqs**2):
                if A_vec[k]:
                    S_B[k, j] = -1
                    j += 1
        if len(B_solve[B_mask]) != 0:
            B_vec = np.ravel(B_mask, order="F")
            for k in range(neqs**2):
                if B_vec[k]:
                    S_D[k, j_d] = 1
                    j_d += 1

        # now compute J, the derivative of vech(sigma) with respect to the
        # free entries of A and B, at the covariance of the model,
        # sigma = A^-1 B B' A^-1'
        invA = npl.inv(A_solve)
        invA_B = np.dot(invA, B_solve)
        sigma = np.dot(invA_B, invA_B.T)
        J_p1i = np.dot(np.dot(D_pl, np.kron(sigma, invA)), S_B)
        J_p1 = -2.0 * J_p1i
        J_p2 = 2.0 * np.dot(np.dot(D_pl, np.kron(invA_B, invA)), S_D)

        J = np.append(J_p1, J_p2, axis=1)

        return J

    def check_order(self, J):
        """
        Check the order condition for identification

        Parameters
        ----------
        J : ndarray
            The Jacobian of the distinct entries of the residual covariance
            with respect to the unknown parameters, with neqs * (neqs + 1) / 2
            rows and one column per unknown parameter.

        Raises
        ------
        ValueError
            If there are more unknown parameters than rows of `J`.
        """
        if np.size(J, axis=0) < np.size(J, axis=1):
            raise ValueError("Order condition not met: solution may not be unique")

    def check_rank(self, J):
        """
        Check the rank condition for identification

        Parameters
        ----------
        J : ndarray
            The Jacobian of the distinct entries of the residual covariance
            with respect to the unknown parameters, with neqs * (neqs + 1) / 2
            rows and one column per unknown parameter.

        Raises
        ------
        ValueError
            If the rank of `J` is less than the number of unknown parameters.
        """
        rank = np.linalg.matrix_rank(J)
        if rank < np.size(J, axis=1):
            raise ValueError("Rank condition not met: solution may not be unique.")


class SVARProcess(VARProcess):
    """
    Class represents a known SVAR(p) process

    Parameters
    ----------
    coefs : ndarray (p x k x k)
        The lag coefficient matrices A_1, ..., A_p.
    intercept : ndarray (length k)
        The intercept.
    sigma_u : ndarray (k x k)
        The covariance of the reduced-form residuals.
    A_solve : ndarray
        neqs x neqs array of estimated structural parameters for the A
        matrix.
    B_solve : ndarray
        neqs x neqs array of estimated structural parameters for the B
        matrix.
    names : sequence of str, optional
        length k
    """

    def __init__(self, coefs, intercept, sigma_u, A_solve, B_solve, names=None):
        self.k_ar = len(coefs)
        self.neqs = coefs.shape[1]
        self.coefs = coefs
        self.intercept = intercept
        self.sigma_u = sigma_u
        self.A_solve = A_solve
        self.B_solve = B_solve
        self.names = names
        # The deterministic terms that the methods of VARProcess use: the
        # intercept and no exog. SVARResults sets those of its trend.
        self.coefs_exog = intercept
        self.k_trend = 1
        self.k_exog_user = 0

    def forecast(self, y, steps, exog_future=None):
        """
        Produce linear minimum MSE forecasts for desired number of steps
        ahead, using prior values y

        Parameters
        ----------
        y : ndarray (p x k)
            The values of the process before the forecast period, at least
            the last p.
        steps : int
            The number of steps to forecast.
        exog_future : None, optional
            Must be None, since an SVAR has no exog.

        Returns
        -------
        forecasts : ndarray (steps x neqs)

        Notes
        -----
        Lütkepohl pp 37-38

        The deterministic terms of SVARResults continue those of the
        estimation sample.

        .. versionchanged:: 0.15.1

           Forecasts are available. They raised an AttributeError.
        """
        if exog_future is not None:
            raise ValueError("An SVAR has no exog, so exog_future must be None.")
        trend_coefs = self.coefs_exog.T if self.k_trend else None
        exog = None
        if self.k_trend > 1:
            # The time trend continues that of the estimation sample
            time = self.endog_lagged[-1, 1] + np.arange(1.0, steps + 1)
            exog = np.vander(time, self.k_trend, increasing=True)
        return forecast(y, self.coefs, trend_coefs, steps, exog)

    def orth_ma_rep(self, maxn=10, P=None):
        """
        Unavailable for SVAR, use `svar_ma_rep`

        Parameters
        ----------
        maxn : int, optional
            Unused.
        P : ndarray, optional
            Unused.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError

    def svar_ma_rep(self, maxn=10, P=None):
        """
        Compute Structural MA coefficient matrices using MLE of A, B

        Parameters
        ----------
        maxn : int, optional
            Number of MA matrices to compute.
        P : array_like, optional
            Solved A^{-1}B coefficient matrix used to orthogonalize the MA
            representation. If not provided, it is computed from the
            estimated A and B matrices.

        Returns
        -------
        ndarray
            The structural MA coefficient matrices Phi_h P, h = 0, ..., maxn,
            with shape (maxn + 1, k, k), where Phi_h are the coefficient
            matrices of `ma_rep`.
        """
        if P is None:
            A_solve = self.A_solve
            B_solve = self.B_solve
            P = np.dot(npl.inv(A_solve), B_solve)

        ma_mats = self.ma_rep(maxn=maxn)
        return np.array([np.dot(coefs, P) for coefs in ma_mats])


class SVARResults(SVARProcess, VARResults):
    """
    Results of the estimation of an SVAR(p) process

    Parameters
    ----------
    endog : ndarray
        The endogenous variables, nobs x neqs.
    endog_lagged : ndarray
        The regressors of the VAR: the deterministic terms and the lags.
    params : ndarray
        The estimated VAR coefficients, one column per equation.
    sigma_u : ndarray
        The estimated covariance of the reduced-form residuals.
    lag_order : int
        The number of lags of the VAR.
    A : ndarray, optional
        neqs x neqs array of estimated structural parameters for the A
        matrix.
    B : ndarray, optional
        neqs x neqs array of estimated structural parameters for the B
        matrix.
    A_mask : ndarray of bool, optional
        neqs x neqs array, True where the corresponding entry of `A` is an
        estimated (unknown) parameter.
    B_mask : ndarray of bool, optional
        neqs x neqs array, True where the corresponding entry of `B` is an
        estimated (unknown) parameter.
    model : SVAR, optional
        The SVAR model instance used to produce the results.
    trend : {"n", "c", "ct", "ctt"}, optional
        The deterministic terms of the VAR.
    names : sequence of str, optional
        List of names of the endogenous variables in order of appearance in `endog`.
    dates : array_like, optional
        The date index of `endog`.

    Attributes
    ----------
    A : ndarray
        The estimated A matrix, neqs x neqs.
    A_mask : ndarray of bool
        True where the corresponding entry of `A` is estimated.
    aic : float
    B : ndarray
        The estimated B matrix, neqs x neqs.
    B_mask : ndarray of bool
        True where the corresponding entry of `B` is estimated.
    bic : float
    bse : ndarray
    coefs : ndarray (p x K x K)
        Estimated A_i matrices, A_i = coefs[i-1]
    dates : pandas.Index or None
        The date index of `endog`, None if it has none.
    detomega : float
    df_model : int
    df_resid : int
    endog : ndarray
    endog_lagged : ndarray
    fittedvalues : ndarray
    fpe : float
    intercept : ndarray
    info_criteria : dict of str to float
    k_ar : int
        Order of VAR process
    k_trend : int
    llf : float
    model : SVAR
    names : list of str
        variable names
    neqs : int
        Number of variables (equations)
    nobs : int
    n_totobs : int
    params : ndarray ((k_trend + Kp) x K)
        The coefficients of the deterministic terms and the transposed A_i
        matrices in stacked form [deterministic terms, A_1', ..., A_p'], one
        column per equation.
    pvalues : ndarray
    resid : ndarray
    sigma_u : ndarray (K x K)
        Estimate of white noise process variance Var[u_t]
    sigma_u_mle : ndarray
    stderr : ndarray
    trendorder : int or None
    tvalues : ndarray
    """

    _model_type = "SVAR"

    def __init__(
        self,
        endog,
        endog_lagged,
        params,
        sigma_u,
        lag_order,
        A=None,
        B=None,
        A_mask=None,
        B_mask=None,
        model=None,
        trend="c",
        names=None,
        dates=None,
    ):

        self.model = model
        self.endog = endog
        self.endog_lagged = endog_lagged
        self.dates = dates

        self.n_totobs, self.neqs = self.endog.shape
        self.nobs = self.n_totobs - lag_order
        k_trend = util.get_trendorder(trend)
        if k_trend > 0:  # make this the polynomial trend order
            trendorder = k_trend - 1
        else:
            trendorder = None
        self.trendorder = trendorder

        self.exog_names = util.make_lag_names(names, lag_order, k_trend)
        self.params = params
        self.sigma_u = sigma_u

        # Each matrix needs to be transposed
        reshaped = self.params[k_trend:]
        reshaped = reshaped.reshape((lag_order, self.neqs, self.neqs))

        # Need to transpose each coefficient matrix
        intercept = self.params[0]
        coefs = reshaped.swapaxes(1, 2).copy()

        # SVAR components
        # TODO: if you define these here, you do not also have to define
        # them in SVAR process, but I left them for now -ss
        self.A = A
        self.B = B
        self.A_mask = A_mask
        self.B_mask = B_mask

        super().__init__(coefs, intercept, sigma_u, A, B, names=names)

        # The deterministic terms that the methods of VARResults use, which
        # replace those of a known process set by SVARProcess (no exog)
        self.trend = trend
        self.exog = None
        self.coefs_exog = self.params[:k_trend].T
        self.k_trend = k_trend
        self.k_exog = k_trend  # now (0.9) required by VARProcess

    def irf(self, periods=10, var_order=None):
        """
        Analyze structural impulse responses to shocks in system

        Parameters
        ----------
        periods : int, optional
            The number of periods after the shock, 10 by default.
        var_order : sequence, optional
            Alternate variable order for the structural impulse matrix. Not
            implemented, and so must be None.

        Returns
        -------
        irf : IRAnalysis
            The structural responses Phi_h A^{-1} B are in its `svar_irfs`.

        Raises
        ------
        NotImplementedError
            If `var_order` is not None.
        """
        if var_order is not None:
            raise NotImplementedError(
                "alternate variable order not implemented (yet)"
            )

        A = self.A
        B = self.B
        P = np.dot(npl.inv(A), B)

        return IRAnalysis(self, P=P, periods=periods, svar=True)

    def fevd(self, periods=10):
        """
        Compute the forecast error variance decomposition of the structural shocks

        Parameters
        ----------
        periods : int, optional
            The number of forecast horizons, 10 by default.

        Returns
        -------
        FEVD
            The shares of the structural shocks in the forecast error
            variances, computed from the structural responses
            Phi_h A^{-1} B. The variances are those of the structural model,
            the sums of the squared structural responses, as in R's vars, so
            that the shares sum to one also for an over-identified model.

        Notes
        -----
        .. versionchanged:: 0.15.1

           The decomposition is available. It raised a TypeError.
        """
        return _SVARFEVD(self, periods)

    @deprecate_kwarg("seed", "rng")
    def sirf_errband_mc(
        self,
        orth=False,
        repl=1000,
        steps=10,
        signif=0.05,
        rng=None,
        burn=100,
        cum=False,
    ):
        """
        Compute Monte Carlo integrated error bands assuming normally
        distributed for impulse response functions

        Parameters
        ----------
        orth : bool, optional
            Compute orthogonalized impulse response error bands

            .. deprecated:: 0.15.1

               True was ignored and gave the bands of the structural
               responses. It issues a FutureWarning and will raise a
               ValueError after statsmodels 0.17 is released.
        repl : int, optional
            number of Monte Carlo replications to perform
        steps : int, optional
            number of impulse response periods
        signif : float, optional
            Significance level for error bars (0 < signif < 1), defaults to
            95% CI
        rng : int, array_like of int, numpy.random.Generator, or numpy.random.RandomState, optional
            Source of random numbers used for the Monte Carlo
            replications. If `rng` is None, a new ``Generator`` is
            created using fresh entropy from the operating system. If
            `rng` is an int, a new ``RandomState`` instance is created,
            seeded with `rng`; this integer-seeding behavior is
            deprecated and will change to creating a ``Generator`` in a
            future release. If `rng` is already a ``Generator`` or
            ``RandomState`` instance, that instance is used.
        seed : int, array_like of int, numpy.random.Generator, or numpy.random.RandomState, optional
            .. deprecated:: 0.15

               seed has been deprecated. In-line with SPEC-007, use
               rng for passing a random number generator or seed.
        burn : int, optional
            number of initial observations to discard for simulation
        cum : bool, optional
            produce cumulative irf error bands

        Returns
        -------
        ErrorBand
            A result object with fields ``lower`` and ``upper``, the
            ``signif / 2`` and ``1 - signif / 2`` quantiles of the structural
            MA coefficient matrices (see `svar_ma_rep`, cumulated if `cum`) of
            the replications, each with shape (steps + 1, neqs, neqs).

        Warns
        -----
        ConvergenceWarning
            One warning reports the number of replications for which the
            maximum likelihood fit of the structural parameters did not
            converge. The error bands include these replications.

        Notes
        -----
        Lütkepohl (2005) Appendix D
        """
        if orth:
            warnings.warn(
                "orth=True is ignored and the error bands are those of the "
                "structural responses. It is deprecated and will raise a "
                "ValueError after statsmodels 0.17 is released.",
                FutureWarning,
                stacklevel=3,
            )
        neqs = self.neqs
        k_ar = self.k_ar
        coefs = self.coefs
        sigma_u = self.sigma_u
        intercept = self.intercept
        nobs = self.nobs

        ma_coll = np.zeros((repl, steps + 1, neqs, neqs))
        A = self.A
        B = self.B
        A_mask = self.A_mask
        B_mask = self.B_mask
        A_pass = self.model.A_original
        B_pass = self.model.B_original
        s_type = self.model.svar_type

        g_list = []

        def agg(impulses):
            if cum:
                return impulses.cumsum(axis=0)
            return impulses

        # Normalize once so that the same generator instance is advanced
        # across replications instead of being re-seeded from the same
        # raw seed on every iteration, which would otherwise make every
        # replication identical.
        rng = check_random_state(rng, deprecated=True)

        opt_A = A[A_mask]
        opt_B = B[B_mask]
        n_not_converged = 0
        for i in range(repl):
            # discard first hundred to correct for starting bias
            sim = util.varsim(coefs, intercept, sigma_u, rng=rng, steps=nobs + burn)
            sim = sim[burn:]

            smod = SVAR(sim, svar_type=s_type, A=A_pass, B=B_pass)
            if i == 10:
                # Use first 10 to update starting val for remainder of fits
                mean_AB = np.mean(g_list, axis=0)
                split = len(A[A_mask])
                opt_A = mean_AB[:split]
                opt_B = mean_AB[split:]

            # one warning below reports all the fits that did not converge
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Maximum Likelihood optimization failed",
                    category=ConvergenceWarning,
                )
                sres = smod.fit(maxlags=k_ar, A_guess=opt_A, B_guess=opt_B)
            n_not_converged += not smod.mle_retvals["converged"]

            if i < 10:
                # save estimates for starting val if in first 10
                g_list.append(
                    np.append(sres.A[A_mask].tolist(), sres.B[B_mask].tolist())
                )
            ma_coll[i] = agg(sres.svar_ma_rep(maxn=steps))

        if n_not_converged:
            warnings.warn(
                "Maximum Likelihood optimization failed to converge in "
                f"{n_not_converged} of {repl} Monte Carlo replications. The "
                "error bands include these replications.",
                ConvergenceWarning,
                stacklevel=3,
            )

        ma_sort = np.sort(ma_coll, axis=0)  # sort to get quantiles
        index = (
            int(round(signif / 2 * repl) - 1),
            int(round((1 - signif / 2) * repl) - 1),
        )
        lower = ma_sort[index[0], :, :, :]
        upper = ma_sort[index[1], :, :, :]
        return ErrorBand(lower, upper)


class _SVARFEVD(FEVD):
    """
    Forecast error variance decomposition of the structural shocks of an SVAR

    Parameters
    ----------
    model : SVARResults
        The SVAR results.
    periods : int
        The number of forecast horizons.
    """

    def __init__(self, model, periods):
        super().__init__(model, periods=periods)

    def _decompose(self, P):
        # The structural responses are those of A and B, so P (None) is unused
        periods = self.periods
        self.irfobj = self.model.irf(periods=periods)
        # squared structural responses, cumulated over the horizons, of shape
        # (periods, neqs, neqs): horizon x response x shock
        cum = (self.irfobj.svar_irfs[:periods] ** 2).cumsum(axis=0)
        # their sums over the shocks are the forecast error variances
        fevd = cum / cum.sum(axis=2, keepdims=True)
        # switch to response x horizon x shock
        return fevd.swapaxes(0, 1)
