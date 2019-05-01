import numpy as np
from statsmodels.base import model
import statsmodels.base.wrapper as wrap

class DimReductionRegression(model.Model):

    def __init__(self, endog, exog, **kwargs):
        super(DimReductionRegression, self).__init__(endog, exog, **kwargs)

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


class SIR(DimReductionRegression):
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

    def __init__(self, endog, exog, **kwargs):
        super(SIR, self).__init__(endog, exog, **kwargs)


    def fit(self, **kwargs):
        """
        Estimate the EDR space.

        Optional keyword parameters
        ---------------------------
        slice_n : int
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


class PHD(DimReductionRegression):
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

    def __init__(self, endog, exog, **kwargs):
        super(PHD, self).__init__(endog, exog, **kwargs)


    def fit(self, **kwargs):
        """
        Estimate the EDR space using PHD.

        Optional keyword parameters
        ---------------------------
        resid : bool
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


class SAVE(DimReductionRegression):
    """
    Sliced Average Variance Estimation (SAVE)

    Parameters
    ----------
    endog : array-like (1d)
        The dependent variable
    exog : array-like (2d)
        The covariates

    Keyword parameters
    ------------------
    bc : bool
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
        if "bc" in kwargs.keys() and kwargs["bc"] == True:
            self.bc = True

    def fit(self, **kwargs):
        """
        Estimate the EDR space.

        Optional keyword arguments
        --------------------------
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

    def __init__(self, model, params, eigs):
        super(DimReductionResults, self).__init__(
              model, params)
        self.eigs = eigs


class DimReductionResultsWrapper(wrap.ResultsWrapper):
    _attrs = {
        'params': 'columns',
    }
    _wrap_attrs = _attrs

wrap.populate_wrapper(DimReductionResultsWrapper,
                      DimReductionResults)
