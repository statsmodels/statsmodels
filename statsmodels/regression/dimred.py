import numpy as np
from statsmodels.base import model
import statsmodels.base.wrapper as wrap
import pandas as pd
import warnings


class _DimReductionRegression(model.Model):
    """
    A base class for dimension reduction regression methods.
    """

    def __init__(self, endog, exog, **kwargs):
        super(_DimReductionRegression, self).__init__(endog, exog, **kwargs)

    def _prep(self, n_slice):

        # Sort the data by endog
        ii = np.argsort(self.endog)
        x = self.exog[ii, :]

        # Whiten the data
        x -= x.mean(0)
        covx = np.cov(x.T)
        covxr = np.linalg.cholesky(covx)
        x = np.linalg.solve(covxr, x.T).T
        self.wexog = x
        self._covxr = covxr

        # Split the data into slices
        self._split_wexog = np.array_split(x, n_slice)


class SlicedInverseReg(_DimReductionRegression):
    """
    Sliced Inverse Regression (SIR)

    Parameters
    ----------
    endog : array-like (1d)
        The dependent variable
    exog : array-like (2d)
        The covariates

    References
    ----------
    KC Li (1991).  Sliced inverse regression for dimension reduction.
    JASA 86, 316-342.
    """

    def fit(self, **kwargs):
        """
        Estimate the EDR space.

        Parameters
        ----------
        slice_n : int, optional
            Number of observations per slice
        """

        # Sample size per slice
        slice_n = kwargs.get("slice_n", 20)

        # Number of slices
        n_slice = self.exog.shape[0] // slice_n

        self._prep(n_slice)

        mn = [z.mean(0) for z in self._split_wexog]
        n = [z.shape[0] for z in self._split_wexog]
        mn = np.asarray(mn)
        n = np.asarray(n)
        mnc = np.cov(mn.T, fweights=n)

        a, b = np.linalg.eigh(mnc)
        jj = np.argsort(-a)
        a = a[jj]
        b = b[:, jj]
        params = np.linalg.solve(self._covxr.T, b)

        results = DimReductionResults(self, params, eigs=a)
        return DimReductionResultsWrapper(results)


class PrincipalHessianDirections(_DimReductionRegression):
    """
    Principal Hessian Directions

    Parameters
    ----------
    endog : array-like (1d)
        The dependent variable
    exog : array-like (2d)
        The covariates

    References
    ----------
    KC Li (1992).  On Principal Hessian Directions for Data
    Visualization and Dimension Reduction: Another application
    of Stein's lemma. JASA 87:420.
    """

    def fit(self, **kwargs):
        """
        Estimate the EDR space using PHD.

        Parameters
        ----------
        resid : bool, optional
            If True, use least squares regression to remove the
            linear relationship between each covariate and the
            response, before conducting PHD.
        """

        resid = kwargs.get("resid", False)

        y = self.endog - self.endog.mean()
        x = self.exog - self.exog.mean(0)

        if resid:
            from statsmodels.regression.linear_model import OLS
            r = OLS(y, x).fit()
            y = r.resid

        cm = np.einsum('i,ij,ik->jk', y, x, x)
        cm /= len(y)

        cx = np.cov(x.T)
        cb = np.linalg.solve(cx, cm)

        a, b = np.linalg.eig(cb)
        jj = np.argsort(-np.abs(a))
        a = a[jj]
        params = b[:, jj]

        results = DimReductionResults(self, params, eigs=a)
        return DimReductionResultsWrapper(results)


class SlicedAverageVarianceEstimation(_DimReductionRegression):
    """
    Sliced Average Variance Estimation (SAVE)

    Parameters
    ----------
    endog : array-like (1d)
        The dependent variable
    exog : array-like (2d)
        The covariates
    bc : bool, optional
        If True, use the bias-correctedCSAVE method of Li and Zhu.

    References
    ----------
    RD Cook.  SAVE: A method for dimension reduction and graphics
    in regression.
    http://www.stat.umn.edu/RegGraph/RecentDev/save.pdf

    Y Li, L-X Zhu (2007). Asymptotics for sliced average
    variance estimation.  The Annals of Statistics.
    https://arxiv.org/pdf/0708.0462.pdf
    """

    def __init__(self, endog, exog, **kwargs):
        super(SAVE, self).__init__(endog, exog, **kwargs)

        self.bc = False
        if "bc" in kwargs and kwargs["bc"] is True:
            self.bc = True

    def fit(self, **kwargs):
        """
        Estimate the EDR space.

        Parameters
        ----------
        slice_n : int
            Number of observations per slice
        """

        # Sample size per slice
        slice_n = kwargs.get("slice_n", 50)

        # Number of slices
        n_slice = self.exog.shape[0] // slice_n

        self._prep(n_slice)

        cv = [np.cov(z.T) for z in self._split_wexog]
        ns = [z.shape[0] for z in self._split_wexog]

        p = self.wexog.shape[1]

        if not self.bc:
            # Cook's original approach
            vm = 0
            for w, cvx in zip(ns, cv):
                icv = np.eye(p) - cvx
                vm += w * np.dot(icv, icv)
            vm /= len(cv)
        else:
            # The bias-corrected approach of Li and Zhu

            # \Lambda_n in Li, Zhu
            av = 0
            for c in cv:
                av += np.dot(c, c)
            av /= len(cv)

            # V_n in Li, Zhu
            vn = 0
            for x in self._split_wexog:
                r = x - x.mean(0)
                for i in range(r.shape[0]):
                    u = r[i, :]
                    m = np.outer(u, u)
                    vn += np.dot(m, m)
            vn /= self.exog.shape[0]

            c = np.mean(ns)
            k1 = c * (c - 1) / ((c - 1)**2 + 1)
            k2 = (c - 1) / ((c - 1)**2 + 1)
            av2 = k1 * av - k2 * vn

            vm = np.eye(p) - 2 * sum(cv) / len(cv) + av2

        a, b = np.linalg.eigh(vm)
        jj = np.argsort(-a)
        a = a[jj]
        b = b[:, jj]
        params = np.linalg.solve(self._covxr.T, b)

        results = DimReductionResults(self, params, eigs=a)
        return DimReductionResultsWrapper(results)


class DimReductionResults(model.Results):
    """
    Results class for a dimension reduction regression.
    """

    def __init__(self, model, params, eigs):
        super(DimReductionResults, self).__init__(
              model, params)
        self.eigs = eigs


class DimReductionResultsWrapper(wrap.ResultsWrapper):
    _attrs = {
        'params': 'columns',
    }
    _wrap_attrs = _attrs

wrap.populate_wrapper(DimReductionResultsWrapper,  # noqa:E305
                      DimReductionResults)

# aliases for expert users
SIR = SlicedInverseReg
PHD = PrincipalHessianDirections
SAVE = SlicedAverageVarianceEstimation


class CovReduce(_DimReductionRegression):
    """
    Dimension reduction for a collection of covariance matrices.

    Parameters
    ----------
    endog : array-like
        The dependent variable, treated as group labels
    exog : array-like
        The independent variables.
    dim : integer
        The dimension of the subspace onto which the covariance
        matrices are projected.

    Returns
    -------
    An orthogonal matrix P such that replacing each group's
    covariance matrix C with P'CP optimally preserves the
    differences among these matrices.

    Notes
    -----
    This is a likelihood-based dimension reduction procedure based
    on Wishart models for sample covariance matrices.  The goal
    is to find a projection matrix P so that C_i | P'C_iP and
    C_j | P'C_jP are equal in distribution for all i, j, where
    the C_i are the within-group covariance matrices.

    The model and methodology are as described in Cook and Forzani,
    but the optimization method follows Edelman et. al.

    References
    ----------
    DR Cook, L Forzani (2008).  Covariance reducing models: an alternative
    to spectral modeling of covariance matrices.  Biometrika 95:4.

    A Edelman, TA Arias, ST Smith (1998).  The geometry of algorithms with
    orthogonality constraints. SIAM J Matrix Anal Appl.
    http://math.mit.edu/~edelman/publications/geometry_of_algorithms.pdf
    """

    def __init__(self, endog, exog, dim):

        super(CovReduce, self).__init__(endog, exog)

        covs, ns = [], []
        df = pd.DataFrame(self.exog, index=self.endog)
        for _, v in df.groupby(df.index):
            covs.append(v.cov().values)
            ns.append(v.shape[0])

        self.nobs = len(endog)

        # The marginal covariance
        covm = 0
        for i, _ in enumerate(covs):
            covm += covs[i] * ns[i]
        covm /= self.nobs
        self.covm = covm

        self.covs = covs
        self.ns = ns
        self.dim = dim

    def loglike(self, params):
        """
        Evaluate the log-likelihood

        Parameters
        ----------
        params : array-like
            The projection matrix used to reduce the covariances, flattened
            to 1d.

        Returns the log-likelihood.
        """

        p = self.covm.shape[0]
        proj = params.reshape((p, self.dim))

        c = np.dot(proj.T, np.dot(self.covm, proj))
        _, ldet = np.linalg.slogdet(c)
        f = self.nobs * ldet / 2

        for j, c in enumerate(self.covs):
            c = np.dot(proj.T, np.dot(c, proj))
            _, ldet = np.linalg.slogdet(c)
            f -= self.ns[j] * ldet / 2

        return f

    def score(self, params):
        """
        Evaluate the score function.

        Parameters
        ----------
        params : array-like
            The projection matrix used to reduce the covariances,
            flattened to 1d.

        Returns the score function evaluated at 'params'.
        """

        p = self.covm.shape[0]
        proj = params.reshape((p, self.dim))

        c0 = np.dot(proj.T, np.dot(self.covm, proj))
        cP = np.dot(self.covm, proj)
        g = self.nobs * np.linalg.solve(c0, cP.T).T

        for j, c in enumerate(self.covs):
            c0 = np.dot(proj.T, np.dot(c, proj))
            cP = np.dot(c, proj)
            g -= self.ns[j] * np.linalg.solve(c0, cP.T).T

        return g.ravel()

    def fit(self, start_params=None, maxiter=100, gtol=1e-4):
        """
        Fit the covariance reduction model.

        Parameters
        ----------
        start_params : array-like
            Starting value for the projection matrix. May be
            rectangular, or flattened.
        maxiter : integer
            The maximum number of gradient steps to take.
        gtol : float
            Convergence criterion for the gradient norm.

        Returns
        -------
        An orthogonal p x d matrix P that optimizes the likelihood.
        """

        p = self.covm.shape[0]
        d = self.dim

        # Starting value for params
        if start_params is None:
            params = np.zeros((p, d))
            params[0:d, 0:d] = np.eye(d)
            params = params.ravel()
        else:
            params = start_params.ravel()

        llf = self.loglike(params)

        for _ in range(maxiter):

            g = self.score(params)
            g -= np.dot(g, params) * params / np.dot(params, params)

            if np.sqrt(np.sum(g * g)) < gtol:
                break

            gm = g.reshape((p, d))
            u, s, vt = np.linalg.svd(gm, 0)

            paramsm = params.reshape((p, d))
            pa0 = np.dot(paramsm, vt.T)

            def geo(t):
                # Parameterize the geodesic path in the direction
                # of the gradient as a function of t (real).
                pa = pa0 * np.cos(s * t) + u * np.sin(s * t)
                return np.dot(pa, vt).ravel()

            # Try to find an uphill step along the geodesic path.
            step = 2.
            while step > 1e-10:
                pa = geo(step)
                llf1 = self.loglike(pa)
                if llf1 > llf:
                    params = pa
                    llf = llf1
                    break
                step /= 2

            if step <= 1e-10:
                msg = "CovReduce optimization did not converge"
                warnings.warn(msg)
                break

        params = params.reshape((p, d))
        results = DimReductionResults(self, params, eigs=None)
        results.llf = llf
        return DimReductionResultsWrapper(results)
