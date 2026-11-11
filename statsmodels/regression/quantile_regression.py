"""
Quantile regression model

Model parameters are estimated using iterated reweighted least squares or the
Frisch-Newton interior point method. The asymptotic covariance matrix
estimated using kernel density estimation.

Author: Vincent Arel-Bundock
License: BSD-3
Created: 2013-03-19

The original IRLS function was written for Matlab by Shapour Mohammadi,
University of Tehran, 2008 (shmohammadi@gmail.com), with some lines based on
code written by James P. Lesage in Applied Econometrics Using MATLAB(1999).PP.
73-4.  Translated to python with permission from original author by Christian
Prinoth (christian at prinoth dot name).
"""

import warnings

import numpy as np
from numpy.linalg import pinv
from scipy import linalg, stats
from scipy.stats import norm

from statsmodels.regression.linear_model import (
    RegressionModel,
    RegressionResults,
    RegressionResultsWrapper,
)
from statsmodels.tools._decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import ConvergenceWarning, IterationLimitWarning
from statsmodels.tools.validation import string_like


class QuantReg(RegressionModel):
    """
    Quantile Regression

    Estimate a quantile regression model using iterative reweighted least
    squares or the Frisch-Newton interior point method.

    Parameters
    ----------
    endog : array_like
        Endogenous/response variable.
    exog : array_like
        Exogenous/explanatory variable(s).

    Notes
    -----
    The Least Absolute Deviation (LAD) estimator is a special case where
    quantile is set to 0.5 (q argument of the fit method).

    The asymptotic covariance matrix is estimated following the procedure in
    Greene (2008, p.407-408), using either the logistic or gaussian kernels
    (kernel argument of the fit method).

    References
    ----------
    General:

    * Birkes, D. and Y. Dodge (1993). Alternative Methods of Regression, John Wiley and Sons.
    * Green, W. H. (2008). Econometric Analysis. Sixth Edition. International Student Edition.
    * Koenker, R. (2005). Quantile Regression. New York: Cambridge University Press.
    * LeSage, J. P. (1999). Applied Econometrics Using MATLAB,
    * Portnoy, S. and R. Koenker (1997). The Gaussian hare and the Laplacian tortoise: computability of squared-error versus absolute-error estimators. Statistical Science 12(4): 279-300.

    Kernels (used by the fit method):

    * Green (2008) Table 14.2

    Bandwidth selection (used by the fit method):

    * Bofinger, E. (1975). Estimation of a density function using order statistics. Australian Journal of Statistics 17: 1-17.
    * Chamberlain, G. (1994). Quantile regression, censoring, and the structure of wages. In Advances in Econometrics, Vol. 1: Sixth World Congress, ed. C. A. Sims, 171-209. Cambridge: Cambridge University Press.
    * Hall, P., and S. Sheather. (1988). On the distribution of the Studentized quantile. Journal of the Royal Statistical Society, Series B 50: 381-391.

    Keywords: Least Absolute Deviation(LAD) Regression, Quantile Regression,
    Regression, Robust Estimation.

    """

    def __init__(self, endog, exog, **kwargs):
        self._check_kwargs(kwargs)
        super().__init__(endog, exog, **kwargs)

    def whiten(self, data):
        """QuantReg model whitener does nothing: returns data"""
        return data

    def fit(
        self,
        q=0.5,
        vcov="robust",
        kernel="epa",
        bandwidth="hsheather",
        max_iter=1000,
        p_tol=1e-6,
        method="irls",
        **kwargs,
    ):
        """
        Solve by Iterative Weighted Least Squares or Interior Point

        Parameters
        ----------
        q : float, optional
            Quantile must be strictly between 0 and 1.
        vcov : {'robust', 'iid'}, optional
            Method used to calculate the variance-covariance matrix
            of the parameters. Default is ``robust``:

            - robust : heteroskedasticity robust standard errors (as suggested
              in Greene 6th edition)
            - iid : iid errors (as in Stata 12)

        kernel : {'biw', 'cos', 'epa', 'gau', 'par'}, optional
            Kernel to use in the kernel density estimation for the
            asymptotic covariance matrix:

            - biw: Biweight
            - epa: Epanechnikov
            - cos: Cosine
            - gau: Gaussian
            - par: Parzen

        bandwidth : {'hsheather', 'bofinger', 'chamberlain'}, optional
            Bandwidth selection method in kernel density
            estimation for asymptotic covariance estimate (full
            references in QuantReg docstring):

            - hsheather: Hall-Sheather (1988)
            - bofinger: Bofinger (1975)
            - chamberlain: Chamberlain (1994)
        max_iter : int, optional
            Maximum number of iterations.
        p_tol : float, optional
            Convergence tolerance. For ``method="irls"`` it is the tolerance
            for the largest absolute change of the parameter estimates. For
            ``method="fn"`` it is the tolerance for the duality gap, which
            is in the units of the objective function, the sum of the check
            function of the residuals. The default 1e-6 is the default of
            ``rq.fit.fnb`` in the R package quantreg.
        method : {"irls", "fn"}, optional
            The algorithm used to estimate the parameters:

            - irls : Iterative reweighted least squares (default).
            - fn : Frisch-Newton interior point method of Portnoy and
              Koenker (1997), as ``rq(method="fn")`` in the R package
              quantreg. It is usually much faster than ``irls`` and
              solves the linear program to high precision, see Notes.

            .. versionadded:: 0.15.1
        **kwargs
            Additional keyword arguments, accepted for API compatibility.

        Returns
        -------
        RegressionResultsWrapper
            Results instance for the fitted quantile regression, with
            additional ``q``, ``iterations``, ``sparsity``, ``bandwidth``,
            and ``history`` attributes. For ``method="irls"`` the
            ``history`` dict holds the ``params`` and ``mse`` of each
            iteration, for ``method="fn"`` it holds the ``params`` and the
            duality ``gap`` of each iteration.

        Notes
        -----
        The Frisch-Newton method solves the dual of the linear program of
        quantile regression by a primal-dual path following interior point
        algorithm with Mehrotra's predictor-corrector steps. It needs few
        iterations, usually fewer than 50 also for large samples, each of
        which costs one weighted cross product of ``exog``.

        If the solution is not unique, for example the median of an even
        number of observations, the interior point method stops at a point
        inside the set of solutions, close to its center, and not at a
        vertex as a simplex method would. All points in the set have the
        same value of the objective function.

        If ``exog`` does not have full column rank, both methods return the
        solution of minimum norm that lies in the row space of ``exog``,
        i.e. the parameters of the collinear columns are not identified and
        only the fitted values are unique. ``irls`` obtains this by the
        pseudoinverse, ``fn`` by solving the problem in the basis of the
        right singular vectors of ``exog`` with nonzero singular values.
        """
        if q <= 0 or q >= 1:
            raise ValueError("q must be strictly between 0 and 1")

        method = string_like(method, "method", options=("irls", "fn"), lower=False)

        kernel = string_like(
            kernel, "kernel", options=("biw", "cos", "epa", "gau", "par"),
            lower=False,
        )
        kernel = kernels[kernel]

        bandwidth = string_like(
            bandwidth, "bandwidth",
            options=("hsheather", "bofinger", "chamberlain"), lower=False,
        )
        if bandwidth == "hsheather":
            bandwidth = hall_sheather
        elif bandwidth == "bofinger":
            bandwidth = bofinger
        else:  # bandwidth == "chamberlain"
            bandwidth = chamberlain

        endog = self.endog
        exog = self.exog
        nobs = self.nobs
        exog_rank = np.linalg.matrix_rank(self.exog)
        self.rank = exog_rank
        self.df_model = float(self.rank - self.k_constant)
        self.df_resid = self.nobs - self.rank
        if method == "fn":
            if exog_rank < exog.shape[1]:
                # Solve in the basis of the row space of exog, which gives the
                # minimum norm solution like the pseudoinverse in IRLS.
                basis = np.linalg.svd(exog, full_matrices=False)[2][:exog_rank].T
                gamma, n_iter, history, converged = _fit_frisch_newton(
                    endog, exog @ basis, q, max_iter, p_tol
                )
                beta = basis @ gamma
                history["params"] = [basis @ g for g in history["params"]]
            else:
                beta, n_iter, history, converged = _fit_frisch_newton(
                    endog, exog, q, max_iter, p_tol
                )
            if not converged:
                warnings.warn(
                    f"Maximum number of iterations ({max_iter}) reached.",
                    IterationLimitWarning,
                    stacklevel=2,
                )
        else:  # method == "irls"
            n_iter = 0
            xstar = exog

            beta = np.ones(exog.shape[1])
            # TODO: better start, initial beta is used only for convergence check

            # Note the following does not work yet,
            # the iteration loop always starts with OLS as initial beta
            # if start_params is not None:
            #    if len(start_params) != rank:
            #       raise ValueError('start_params has wrong length')
            #       beta = start_params
            #    else:
            #       # start with OLS
            #       beta = np.dot(np.linalg.pinv(exog), endog)

            diff = 10
            cycle = False

            history = dict(params=[], mse=[])
            while n_iter < max_iter and diff > p_tol and not cycle:
                n_iter += 1
                beta0 = beta
                xtx = np.dot(xstar.T, exog)
                xty = np.dot(xstar.T, endog)
                beta = np.dot(pinv(xtx), xty)
                resid = endog - np.dot(exog, beta)

                mask = np.abs(resid) < 0.000001
                resid[mask] = ((resid[mask] >= 0) * 2 - 1) * 0.000001
                resid = np.where(resid < 0, q * resid, (1 - q) * resid)
                resid = np.abs(resid)
                xstar = exog / resid[:, np.newaxis]
                diff = np.max(np.abs(beta - beta0))
                history["params"].append(beta)
                history["mse"].append(np.mean(resid * resid))

                if (n_iter >= 300) and (n_iter % 100 == 0):
                    # check for convergence circle, should not happen
                    for ii in range(2, 10):
                        if np.all(beta == history["params"][-ii]):
                            cycle = True
                            warnings.warn(
                                "Convergence cycle detected",
                                ConvergenceWarning,
                                stacklevel=2,
                            )
                            break

            if n_iter == max_iter:
                warnings.warn(
                    "Maximum number of iterations (" + str(max_iter) + ") reached.",
                    IterationLimitWarning,
                    stacklevel=2,
                )

        e = endog - np.dot(exog, beta)
        # Greene (2008, p.407) writes that Stata 6 uses this bandwidth:
        # h = 0.9 * np.std(e) / (nobs**0.2)
        # Instead, we calculate bandwidth as in Stata 12
        iqre = stats.scoreatpercentile(e, 75) - stats.scoreatpercentile(e, 25)
        h = bandwidth(nobs, q)
        h = min(np.std(endog), iqre / 1.34) * (norm.ppf(q + h) - norm.ppf(q - h))

        fhat0 = 1.0 / (nobs * h) * np.sum(kernel(e / h))

        vcov = string_like(vcov, "vcov", options=("robust", "iid"), lower=False)
        if vcov == "robust":
            d = np.where(e > 0, (q / fhat0) ** 2, ((1 - q) / fhat0) ** 2)
            xtxi = pinv(np.dot(exog.T, exog))
            xtdx = np.dot(exog.T * d[np.newaxis, :], exog)
            vcov = xtxi @ xtdx @ xtxi
        else:  # vcov == "iid"
            vcov = (1.0 / fhat0) ** 2 * q * (1 - q) * pinv(np.dot(exog.T, exog))

        lfit = QuantRegResults(self, beta, normalized_cov_params=vcov)

        lfit.q = q
        lfit.iterations = n_iter
        lfit.sparsity = 1.0 / fhat0
        lfit.bandwidth = h
        lfit.history = history

        return RegressionResultsWrapper(lfit)


def _max_step(v, dv):
    """Largest step t with v + t * dv >= 0 for v >= 0, inf if dv >= 0."""
    # v + t * dv >= 0 is 1 / t >= -dv / v for the elements with dv < 0. An
    # element with v = 0 and dv < 0 gives inf, i.e. a step of zero, and
    # v = dv = 0 gives nan, which fmax ignores.
    with np.errstate(divide="ignore", invalid="ignore"):
        largest = np.fmax.reduce(-dv / v)
    return 1.0 / largest if largest > 0 else np.inf


def _normal_solver(xq, exog):
    """
    Factor exog' diag(q) exog once, with xq = diag(q) exog.

    Returns a function that solves (exog' diag(q) exog) d = xq' rhs.
    """
    mat = xq.T @ exog
    try:
        factor = linalg.cho_factor(mat)
    except linalg.LinAlgError:
        # Near the solution the weights q span many orders of magnitude and
        # the matrix can lose positive definiteness in floating point.
        return lambda rhs: np.linalg.lstsq(mat, xq.T @ rhs, rcond=None)[0]
    return lambda rhs: linalg.cho_solve(factor, xq.T @ rhs)


def _fit_frisch_newton(endog, exog, q, max_iter, tol, step=0.99995):
    """
    Quantile regression by the Frisch-Newton interior point method.

    Parameters
    ----------
    endog : ndarray
        The response, shape (nobs,).
    exog : ndarray
        The design matrix with full column rank, shape (nobs, k).
    q : float
        The quantile, strictly between 0 and 1.
    max_iter : int
        The maximum number of interior point iterations.
    tol : float
        The iterations stop when the duality gap is below ``tol``.
    step : float, optional
        Fraction of the step to the boundary of the feasible region.

    Returns
    -------
    params : ndarray
        The estimated coefficients.
    n_iter : int
        The number of iterations.
    history : dict
        The ``params`` and the duality ``gap`` after each iteration.
    converged : bool
        True if the duality gap is below ``tol``.

    Notes
    -----
    The primal-dual path following algorithm with Mehrotra's predictor
    corrector steps of Portnoy and Koenker (1997) is applied to the dual of
    the quantile regression linear program,

        max_a endog' a  subject to  exog' a = (1 - q) exog' 1,  0 <= a <= 1.

    The coefficients are the Lagrange multipliers of the equality
    constraints. In the notation below the problem is written as
    min c'x subject to A x = b, 0 <= x <= 1 with c = -endog, A = exog' and
    b = (1 - q) exog' 1, the slack is s = 1 - x, the dual variables are y
    (with params = -y), z >= 0 for x >= 0 and w >= 0 for s >= 0.

    The structure of the iterations (starting point, step lengths and the
    centering parameter) follows Portnoy and Koenker (1997) and the lp_fnm
    code of Morillo, Koenker and Eilers, as released under the BSD-3 license
    with the permission of the authors in QuantileRegressions.jl, see also
    pyqreg (BSD-3). The Newton directions are derived from the perturbed
    KKT conditions of the problem above.

    References
    ----------
    .. [*] Portnoy, S. and R. Koenker (1997). The Gaussian hare and the
       Laplacian tortoise: computability of squared-error versus
       absolute-error estimators. Statistical Science 12(4): 279-300.
    """
    nobs = exog.shape[0]
    # Feasible primal start: a = 1 - q satisfies the equality constraints.
    x = np.full(nobs, 1.0 - q)
    s = np.full(nobs, q)
    # Dual start from least squares, the dual slacks split the residuals
    # r = c - A'y = -(endog - exog @ beta_ols) into positive and negative
    # parts, so that A'y + z - w = c holds exactly.
    y = -np.linalg.lstsq(exog, endog, rcond=None)[0]
    r = -endog - exog @ y
    z = np.maximum(r, 0)
    w = z - r
    # Zero residuals would give z = w = 0 on the boundary. Moving both
    # slacks into the interior keeps z - w = r, so the start stays dual
    # feasible.
    zero = r == 0
    z[zero] = 0.001
    w[zero] = 0.001

    xq = np.empty_like(exog)
    history = dict(params=[], gap=[])
    gap = x @ z + s @ w
    n_iter = 0
    while gap > tol and n_iter < max_iter:
        n_iter += 1
        # Affine scaling (predictor) direction. Dual feasibility is kept,
        # so r = z - w = c - A'y.
        qw = 1.0 / (z / x + w / s)
        r = z - w
        np.multiply(exog, qw[:, None], out=xq)
        solve = _normal_solver(xq, exog)
        dy = solve(r)
        dx = qw * (exog @ dy - r)
        ds = -dx
        dz = -z * (1 + dx / x)
        dw = -w * (1 + ds / s)

        fp = min(step * min(_max_step(x, dx), _max_step(s, ds)), 1.0)
        fd = min(step * min(_max_step(z, dz), _max_step(w, dw)), 1.0)

        if min(fp, fd) < 1:
            # Centering and corrector step. The target mu shrinks with the
            # cube of the complementarity reached by the affine step.
            mu = x @ z + s @ w
            g = (x + fp * dx) @ (z + fd * dz) + (s + fp * ds) @ (w + fd * dw)
            mu = mu * (g / mu) ** 3 / (2 * nobs)

            # Newton step for x z = mu and s w = mu including the second
            # order terms dx * dz and ds * dw of the affine step.
            dxdz = dx * dz
            dsdw = ds * dw
            xinv = 1.0 / x
            sinv = 1.0 / s
            xi = mu * (xinv - sinv)
            rhs = r - xi + dxdz * xinv - dsdw * sinv
            dy = solve(rhs)
            dx = qw * (exog @ dy - rhs)
            ds = -dx
            dz = mu * xinv - z - xinv * (z * dx + dxdz)
            dw = mu * sinv - w - sinv * (w * ds + dsdw)

            fp = min(step * min(_max_step(x, dx), _max_step(s, ds)), 1.0)
            fd = min(step * min(_max_step(z, dz), _max_step(w, dw)), 1.0)

        x += fp * dx
        s += fp * ds
        y += fd * dy
        z += fd * dz
        w += fd * dw
        gap = x @ z + s @ w
        history["params"].append(-y)
        history["gap"].append(gap)

    return -y, n_iter, history, gap <= tol


def _parzen(u):
    z = np.where(
        np.abs(u) <= 0.5,
        4.0 / 3 - 8.0 * u**2 + 8.0 * np.abs(u) ** 3,
        8.0 * (1 - np.abs(u)) ** 3 / 3.0,
    )
    z[np.abs(u) > 1] = 0
    return z


kernels = {}
kernels["biw"] = lambda u: 15.0 / 16 * (1 - u**2) ** 2 * np.where(np.abs(u) <= 1, 1, 0)
kernels["cos"] = lambda u: np.where(np.abs(u) <= 0.5, 1 + np.cos(2 * np.pi * u), 0)
kernels["epa"] = lambda u: 3.0 / 4 * (1 - u**2) * np.where(np.abs(u) <= 1, 1, 0)
kernels["gau"] = norm.pdf
kernels["par"] = _parzen
# kernels['bet'] = lambda u: np.where(np.abs(u) <= 1, .75 * (1 - u) * (1 + u), 0)
# kernels['log'] = lambda u: logistic.pdf(u) * (1 - logistic.pdf(u))
# kernels['tri'] = lambda u: np.where(np.abs(u) <= 1, 1 - np.abs(u), 0)
# kernels['trw'] = lambda u: 35. / 32 * (1 - u**2)**3 * np.where(np.abs(u) <= 1, 1, 0)
# kernels['uni'] = lambda u: 1. / 2 * np.where(np.abs(u) <= 1, 1, 0)


def hall_sheather(n, q, alpha=0.05):
    z = norm.ppf(q)
    num = 1.5 * norm.pdf(z) ** 2.0
    den = 2.0 * z**2.0 + 1.0
    h = (
        n ** (-1.0 / 3)
        * norm.ppf(1.0 - alpha / 2.0) ** (2.0 / 3)
        * (num / den) ** (1.0 / 3)
    )
    return h


def bofinger(n, q):
    num = 9.0 / 2 * norm.pdf(2 * norm.ppf(q)) ** 4
    den = (2 * norm.ppf(q) ** 2 + 1) ** 2
    h = n ** (-1.0 / 5) * (num / den) ** (1.0 / 5)
    return h


def chamberlain(n, q, alpha=0.05):
    return norm.ppf(1 - alpha / 2) * np.sqrt(q * (1 - q) / n)


class QuantRegResults(RegressionResults):
    """Results instance for the QuantReg model"""

    @cache_readonly
    def prsquared(self):
        q = self.q
        endog = self.model.endog
        e = self.resid
        e = np.where(e < 0, (1 - q) * e, q * e)
        e = np.abs(e)
        ered = endog - stats.scoreatpercentile(endog, q * 100)
        ered = np.where(ered < 0, (1 - q) * ered, q * ered)
        ered = np.abs(ered)
        return 1 - np.sum(e) / np.sum(ered)

    @cache_writable()
    def scale(self):
        return 1.0

    @cache_readonly
    def bic(self):
        return np.nan

    @cache_readonly
    def aic(self):
        return np.nan

    @cache_readonly
    def llf(self):
        return np.nan

    @cache_readonly
    def rsquared(self):
        return np.nan

    @cache_readonly
    def rsquared_adj(self):
        return np.nan

    @cache_readonly
    def mse(self):
        return np.nan

    @cache_readonly
    def mse_model(self):
        return np.nan

    @cache_readonly
    def mse_total(self):
        return np.nan

    @cache_readonly
    def centered_tss(self):
        return np.nan

    @cache_readonly
    def uncentered_tss(self):
        return np.nan

    @cache_readonly
    def HC0_se(self):
        raise NotImplementedError

    @cache_readonly
    def HC1_se(self):
        raise NotImplementedError

    @cache_readonly
    def HC2_se(self):
        raise NotImplementedError

    @cache_readonly
    def HC3_se(self):
        raise NotImplementedError

    def summary(
            self,
            yname=None,
            xname=None,
            title=None,
            alpha=0.05
    ):
        """
        Summarize the Regression Results

        Parameters
        ----------
        yname : str, optional
            Default is `y`.
        xname : list[str], optional
            Names for the exogenous variables. Default is `var_##` where
            `##` is the 0-based index of the regressor. Must match the
            number of parameters in the model.
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title.
        alpha : float, optional
            Significance level for the confidence intervals.

        Returns
        -------
        smry : Summary instance
            This holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary results
        """
        # Cache the data-dependent scalars computed below as a plain dict
        # (not cache_readonly) so that summary() keeps working after
        # remove_data() has cleared resid/model.exog and the like.
        cache = self.__dict__.setdefault("_summary_cache", {})

        def cached(key, compute):
            if key not in cache:
                cache[key] = compute()
            return cache[key]

        eigvals = cached("eigvals", lambda: self.eigenvals)
        condno = cached("condno", lambda: self.condition_number)
        prsquared = cached("prsquared", lambda: self.prsquared)

        top_left = [
            ("Dep. Variable:", None),
            ("Model:", None),
            ("Method:", ["Least Squares"]),
            ("Date:", None),
            ("Time:", None),
        ]

        top_right = [
            ("Pseudo R-squared:", [f"{prsquared:#8.4g}"]),
            ("Bandwidth:", [f"{self.bandwidth:#8.4g}"]),
            ("Sparsity:", [f"{self.sparsity:#8.4g}"]),
            ("No. Observations:", None),
            ("Df Residuals:", None),
            ("Df Model:", None),
        ]

        if title is None:
            title = self.model.__class__.__name__ + " " + "Regression Results"

        # create summary table instance
        from statsmodels.iolib.summary import Summary

        smry = Summary()
        smry.add_table_2cols(
            self,
            gleft=top_left,
            gright=top_right,
            yname=yname,
            xname=xname,
            title=title,
        )
        smry.add_table_params(
            self, yname=yname, xname=xname, alpha=alpha, use_t=self.use_t
        )

        # add warnings/notes, added to text format only
        etext = []
        if eigvals[-1] < 1e-10:
            wstr = "The smallest eigenvalue is %6.3g. This might indicate "
            wstr += "that there are\n"
            wstr += "strong multicollinearity problems or that the design "
            wstr += "matrix is singular."
            wstr = wstr % eigvals[-1]
            etext.append(wstr)
        elif condno > 1000:  # TODO: what is recommended
            wstr = "The condition number is large, %6.3g. This might "
            wstr += "indicate that there are\n"
            wstr += "strong multicollinearity or other numerical "
            wstr += "problems."
            wstr = wstr % condno
            etext.append(wstr)

        if etext:
            smry.add_extra_txt(etext)

        return smry
