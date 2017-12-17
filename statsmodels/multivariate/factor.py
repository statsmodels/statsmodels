# -*- coding: utf-8 -*-

from statsmodels.base.model import Model
from .factor_rotation import rotate_factors, promax
import numpy as np
from numpy.linalg import eigh, inv, norm, matrix_rank
import pandas as pd
from statsmodels.tools.decorators import cache_readonly
from scipy.optimize import minimize
from statsmodels.iolib import summary2
import warnings

try:
    import matplotlib.pyplot
    missing_matplotlib = False
except ImportError:
    missing_matplotlib = True

if not missing_matplotlib:
    from .plots import plot_scree, plot_loadings

# Factor analysis models can be hard to estimate, need stricter
# conditions.
_opt_defaults = {"gtol": 1e-8}


def _check_args_1(endog, n_factor, corr, nobs):

    msg = "Either endog or corr must be provided, but not both"
    if endog is not None and corr is not None:
        raise ValueError(msg)
    if endog is None and corr is None:
        raise ValueError(msg)

    if n_factor <= 0:
        raise ValueError('n_factor must be larger than 0! %d < 0' %
                         (n_factor))

    if nobs is not None and endog is not None:
        warnings.warn("nobs is ignored when endog is provided")


def _check_args_2(endog, n_factor, corr, nobs, k_endog):

    if n_factor > k_endog:
        raise ValueError('n_factor cannot be greater than the number'
                         ' of variables! %d > %d' %
                         (n_factor, k_endog))

    if np.max(np.abs(np.diag(corr) - 1)) > 1e-10:
        raise ValueError("corr must be a correlation matrix")

    if corr.shape[0] != corr.shape[1]:
        raise ValueError('Correlation matrix corr must be a square '
                         '(rows %d != cols %d)' % corr.shape)


class Factor(Model):
    """
    Factor analysis
    Status: experimental

    .. [1] Hofacker, C. (2004). Exploratory Factor Analysis, Mathematical
    Marketing. http://www.openaccesstexts.org/pdf/Quant_Chapter_11_efa.pdf

    Supported rotations:
        'varimax', 'quartimax', 'biquartimax', 'equamax', 'oblimin',
        'parsimax', 'parsimony', 'biquartimin', 'promax'

    Parameters
    ----------
    endog : array-like
        Variables in columns, observations in rows.  May be `None` if
        `corr` is not `None`.
    n_factor : int
        The number of factors to extract
    corr : array-like
        Directly specify the correlation matrix instead of estimating
        it from `endog`.  If provided, `endog` is not used.
    method : str
        The method to extract factors, currently must be either 'pa'
        for principal axis factor analysis or 'ml' for maximum
        likelihood estimation.
    smc : True or False
        Whether or not to apply squared multiple correlations (method='pa')
    endog_names: str
        Names of endogeous variables.  If specified, it will be used
        instead of the column names in endog
    nobs: int
        The number of observations, not used if endog is present.

    Notes
    -----
    If method='ml', the factors are rotated to satisfy condition IC3
    of Bai and Li (2012).  This means that the scores have covariance
    I, so the model for the covariance matrix is L * L' + diag(U),
    where L are the loadings and U are the uniquenesses.  In addition,
    L' * diag(U)^{-1} L must be diagonal.

    References
    ----------
    J Bai, K Li (2012).  Statistical analysis of factor models of high
    dimension.  Annals of Statistics. https://arxiv.org/pdf/1205.6617.pdf
    """
    def __init__(self, endog=None, n_factor=1, corr=None, method='pa',
                 smc=True, missing='drop', endog_names=None, nobs=None):

        _check_args_1(endog, n_factor, corr, nobs)

        if endog is not None:
            k_endog = endog.shape[1]
            nobs = endog.shape[0]
            corr = np.corrcoef(endog, rowvar=0)
        elif corr is not None:
            k_endog = corr.shape[0]
        else:
            msg = "Either endog or corr must be provided, but not both"
            raise ValueError(msg)

        _check_args_2(endog, n_factor, corr, nobs, k_endog)

        self.n_factor = n_factor
        self.loadings = None
        self.communality = None
        self.method = method
        self.smc = smc
        self.nobs = nobs
        self.method = method
        self.corr = corr
        self.k_endog = k_endog

        if endog_names is None:
            if hasattr(corr, 'index'):
                endog_names = corr.index
            if hasattr(corr, 'columns'):
                endog_names = corr.columns
        self.endog_names = endog_names

        if corr is not None:
            self.corr = np.asarray(corr)
        else:
            self.corr = None

        if endog is not None:
            # Do not preprocess endog if None
            super(Factor, self).__init__(endog, exog=None, missing=missing)
        else:
            self.endog = None

    @property
    def endog_names(self):
        """Names of endogenous variables"""
        if self._endog_names is not None:
            return self._endog_names
        else:
            if self.endog is not None:
                return self.data.ynames
            else:
                d = 0
                n = self.corr.shape[0] - 1
                while n > 0:
                    d += 1
                    n //= 10
                return [('var%0' + str(d) + 'd') % i
                        for i in range(self.corr.shape[0])]

    @endog_names.setter
    def endog_names(self, value):
        # Check validity of endog_names:
        if value is not None:
            if len(value) != self.corr.shape[0]:
                raise ValueError('The length of `endog_names` must '
                                 'equal the number of variables.')
            self._endog_names = np.asarray(value)
        else:
            self._endog_names = None

    def fit(self, maxiter=50, tol=1e-8, start=None, opt_method='bfgs',
            opt=None):
        """
        Estimate factor model parameters.

        Parameters
        ----------
        maxiter : int
            Maximum number of iterations for iterative estimation algorithms
        tol : float
            Stopping critera (error tolerance) for iterative estimation
            algorithms
        start : array-like
            Starting values, currently only used for ML estimation
        opt_method : string
            Optimization method for ML estimation
        opt : dict-like
            Keyword arguments passed to optimizer, only used for ML estimation

        Returns
        -------
        results: FactorResults

        """
        method = self.method.lower()
        if method == 'pa':
            return self._fit_pa(maxiter=maxiter, tol=tol)
        elif method == 'ml':
            return self._fit_ml(start, opt_method, opt)
        else:
            msg = "Unknown factor extraction approach '%s'" % self.method
            raise ValueError(msg)

    def _fit_pa(self, maxiter=50, tol=1e-8):
        """
        Extract factors using the iterative principal axis method

        Parameters
        ----------
        maxiter : int
            Maximum number of iterations for communality estimation
        tol : float
            If `norm(communality - last_communality)  < tolerance`,
            estimation stops

        """
        # Estimate correlation matrix
        if self.corr is not None:
            R = self.corr
        else:
            R = np.corrcoef(self.endog, rowvar=0)

        # Parameter validation
        self.n_comp = matrix_rank(R)
        if self.n_factor > self.n_comp:
            raise ValueError('n_factor must be smaller or equal to the rank'
                             ' of endog! %d > %d' %
                             (self.n_factor, self.n_comp))
        if maxiter <= 0:
            raise ValueError('n_max_iter must be larger than 0! %d < 0' %
                             (maxiter))
        if tol <= 0 or tol > 0.01:
            raise ValueError('tolerance must be larger than 0 and smaller than'
                             ' 0.01! Got %f instead' % (tol))

        #  Initial communality estimation
        if self.smc:
            c = 1 - 1 / np.diag(inv(R))
        else:
            c = np.ones(len(R))

        # Iterative communality estimation
        eigenvals = None
        for i in range(maxiter):
            # Get eigenvalues/eigenvectors of R with diag replaced by
            # communality
            for j in range(len(R)):
                R[j, j] = c[j]
            L, V = eigh(R, UPLO='U')
            c_last = np.array(c)
            ind = np.argsort(L)
            ind = ind[::-1]
            L = L[ind]
            n_pos = (L > 0).sum()
            V = V[:, ind]
            eigenvals = np.array(L)

            # Select eigenvectors with positive eigenvalues
            n = np.min([n_pos, self.n_factor])
            sL = np.diag(np.sqrt(L[:n]))
            V = V[:, :n]

            # Calculate new loadings and communality
            A = V.dot(sL)
            c = np.power(A, 2).sum(axis=1)
            if norm(c_last - c) < tol:
                break

        self.eigenvals = eigenvals
        self.communality = c
        self.uniqueness = 1 - c
        self.loadings = A
        return FactorResults(self)

    # Unpacks the model parameters from a flat vector, used for ML
    # estimation.  The first k_endog elements of par are the square
    # roots of the uniquenesses.  The remaining elements are the
    # factor loadings, packed one factor at a time.
    def _unpack(self, par):
        return (par[0:self.k_endog]**2,
                np.reshape(par[self.k_endog:], (-1, self.k_endog)).T)

    # Packs the model parameters into a flat parameter, used for ML
    # estimation.
    def _pack(self, gamma, sigma):
        return np.concatenate((np.sqrt(sigma), gamma.T.flat))

    # The log-likelihood function.  The input can be either a packed
    # representation of the model parameters or a 2-tuple containing a
    # `k_endog x n_factor` matrix of factor loadings and a `k_endog`
    # vector of uniquenesses.
    def loglike(self, par):

        if len(par) == 2:
            par = self._pack(par[0], par[1])

        sig2, gamma = self._unpack(par)
        sigam = gamma / sig2[:, None]
        gamtsigam = np.dot(gamma.T, sigam)

        # log|GG' + S|
        # Using matrix determinant lemma:
        # |GG' + S| = |I + G'S^{-1}G|*|S|
        gamtsigam.flat[::gamtsigam.shape[0]+1] += 1
        _, ld = np.linalg.slogdet(gamtsigam)
        v = np.sum(np.log(sig2)) + ld

        # tr((GG' + S)^{-1}C)
        # Using Sherman-Morrison-Woodbury
        w = np.sum(1 / sig2)
        b = np.dot(gamma.T, self.corr / sig2[:, None])
        b = np.linalg.solve(gamtsigam, b)
        b = np.dot(gamma, b) / sig2[:, None]
        w -= np.trace(b)

        # Scaled log-likelihood
        return -(v + w) / (2*self.k_endog)

    # Maximum likelihood factor analysis.
    def _fit_ml(self, start, opt_method, opt):

        corr = self.corr
        n_factor = self.n_factor

        # Dimension of the problem
        k_endog = corr.shape[0]

        # Starting values
        if start is None:
            # Use simple PCA procedure for starting values.
            u, s, _ = np.linalg.svd(corr, 0)
            u *= np.sqrt(s)
            u = u[:, 0:n_factor]
            f = 1 - s[0:n_factor].sum() / k_endog
            start1 = f * np.ones(k_endog)
            start = np.concatenate((start1, u.T.flat))
        elif len(start) == 2:
            if len(start[1]) != start[0].shape[0]:
                msg = "Starting values have incompatible dimensions"
                raise ValueError(msg)
            start = self._pack(start[0], start[1])
        else:
            raise ValueError("Invalid starting values")

        # Do the optimization
        def nloglike(par):
            return -self.loglike(par)
        if opt is not None:
            opt = dict(opt)
        else:
            opt = {}
        opt.update(_opt_defaults)
        r = minimize(nloglike, start, method=opt_method, options=opt)
        par = r.x
        uniq, load = self._unpack(par)

        if uniq.min() < 1e-10:
            warnings.warn("some uniquenesses are nearly zero")

        # Rotate solution to satisfy IC3 of Bai and Li
        load3, load0 = self._rotate(load)

        self.uniqueness = uniq
        self.communality = 1 - uniq
        self.loadings = load3
        self.loadings_unrotated = load0

        return FactorResults(self)

    def _rotate(self, load):
        # Rotations used in ML estimation.
        load0, s, _ = np.linalg.svd(load, 0)
        if self.nobs is not None:
            load3 = load0 * np.sqrt(self.nobs)
        else:
            load3 = load0
        load0 *= s
        return load3, load0


class FactorResults(object):
    """
    Factor results class (status experimental)

    For result summary, scree/loading plots and factor rotations

    Parameters
    ----------
    factor : Factor
        Fitted Factor class

    Attributes
    ----------
    uniqueness: ndarray
        The uniqueness (variance of uncorrelated errors unique to
        each variable)
    communality: ndarray
        1 - uniqueness
    loadings : ndarray
        Each column is the loading vector for one factor
    loadings_canonical : ndarray
        A rotation of the loadings, see notes
    n_comp : int
        Number of components (factors)
    nbs : int
        Number of observations
    fa_method : string
        The method used to obtain the decomposition, either 'pa' for
        'principal axes' or 'ml' for maximum likelihood.

    Notes
    -----
    Under ML estimation, the default rotation (used for `loadings`) is
    condition IC3 of Bai and Li (2012).  The standard errors are only
    applicable under this rotation.  An alternative rotation is the
    'canonical loadings' given by `loadings_canonical`.  Under the
    canonical loadings, the factor scores are iid and standardized.
    If `G` is the canonical loadings and `U` is the vector of
    uniquenesses, then the covariance matrix implied by the factor
    analysis is `GG' + diag(U)`.
    """
    def __init__(self, factor):
        self.endog_names = factor.endog_names
        self.loadings_no_rot = factor.loadings
        self.loadings = factor.loadings
        if hasattr(factor, "eigenvals"):
            self.eigenvals = factor.eigenvals
        if hasattr(factor, "loadings_unrotated"):
            self.loadings_unrotated = factor.loadings_unrotated
        self.communality = factor.communality
        self.uniqueness = factor.uniqueness
        self.rotation_method = None
        self.fa_method = factor.method
        self.n_comp = factor.loadings.shape[1]
        self.nobs = factor.nobs
        self._factor = factor

        p, k = self.loadings.shape
        self.df = ((p - k)**2 - (p + k)) // 2

    def __str__(self):
        return self.summary().__str__()

    def rotate(self, method):
        """
        Apply rotation

        Parameters
        ----------
        method : string
            rotation to be applied
        -------
        """
        self.rotation_method = method
        if method not in ['varimax', 'quartimax', 'biquartimax',
                          'equamax', 'oblimin', 'parsimax', 'parsimony',
                          'biquartimin', 'promax']:
            raise ValueError('Unknown rotation method %s' % (method))

        if method in ['varimax', 'quartimax', 'biquartimax', 'equamax',
                      'parsimax', 'parsimony', 'biquartimin']:
            self.loadings, T = rotate_factors(self.loadings_no_rot, method)
        elif method == 'oblimin':
            self.loadings, T = rotate_factors(self.loadings_no_rot,
                                              'quartimin')
        elif method == 'promax':
            self.loadings, T = promax(self.loadings_no_rot)

    def summary(self):
        summ = summary2.Summary()
        summ.add_title('Factor analysis results')
        loadings_no_rot = pd.DataFrame(
            self.loadings_no_rot,
            columns=["factor %d" % (i)
                     for i in range(self.loadings_no_rot.shape[1])],
            index=self.endog_names
        )
        if hasattr(self, "eigenvals"):
            # eigenvals not available for ML method
            eigenvals = pd.DataFrame(
                [self.eigenvals], columns=self.endog_names, index=[''])
            summ.add_dict({'': 'Eigenvalues'})
            summ.add_df(eigenvals)
        communality = pd.DataFrame([self.communality],
                                   columns=self.endog_names, index=[''])
        summ.add_dict({'': ''})
        summ.add_dict({'': 'Communality'})
        summ.add_df(communality)
        summ.add_dict({'': ''})
        summ.add_dict({'': 'Pre-rotated loadings'})
        summ.add_df(loadings_no_rot)
        summ.add_dict({'': ''})
        if self.rotation_method is not None:
            loadings = pd.DataFrame(
                self.loadings,
                columns=["factor %d" % (i)
                         for i in range(self.loadings.shape[1])],
                index=self.endog_names
            )
            summ.add_dict({'': '%s rotated loadings' % (self.rotation_method)})
            summ.add_df(loadings)
        return summ

    def plot_scree(self, ncomp=None):
        """
        Plot of the ordered eigenvalues and variance explained for the loadings

        Parameters
        ----------
        ncomp : int, optional
            Number of loadings to include in the plot.  If None, will
            included the same as the number of maximum possible loadings

        Returns
        -------
        fig : figure
            Handle to the figure
        """
        if missing_matplotlib:
            raise ImportError("Matplotlib missing")
        return plot_scree(self.eigenvals, self.n_comp, ncomp)

    def plot_loadings(self, loading_pairs=None, plot_prerotated=False):
        """
        Plot factor loadings in 2-d plots

        Parameters
        ----------
        loading_pairs : None or a list of tuples
            Specify plots. Each tuple (i, j) represent one figure, i and j is
            the loading number for x-axis and y-axis, respectively. If `None`,
            all combinations of the loadings will be plotted.
        plot_prerotated : True or False
            If True, the loadings before rotation applied will be plotted. If
            False, rotated loadings will be plotted.

        Returns
        -------
        figs : a list of figure handles

        """
        if missing_matplotlib:
            raise ImportError("Matplotlib missing")

        if self.rotation_method is None:
            plot_prerotated = True
        loadings = self.loadings_no_rot if plot_prerotated else self.loadings
        if plot_prerotated:
            title = 'Prerotated Factor Pattern'
        else:
            title = '%s Rotated Factor Pattern' % (self.rotation_method)
        var_explained = self.eigenvals / self.n_comp * 100

        return plot_loadings(loadings, loading_pairs=loading_pairs,
                             title=title, row_names=self.endog_names,
                             percent_variance=var_explained)

    @cache_readonly
    def fitted_cov(self):
        """
        Returns the fitted covariance matrix.
        """

        if hasattr(self, "loadings_unrotated"):
            c = np.dot(self.loadings, self.loadings.T)
            c.flat[::c.shape[0]+1] += self.uniqueness
            return c
        else:
            msg = "Cannot compute fitted covariance"
            raise ValueError(msg)

    @cache_readonly
    def uniq_stderr(self, kurt=0):
        """
        The standard errors of the uniquenesses.

        If excess kurtosis is known, provide as `kurt`.  Standard
        errors are only available if the model was fit using maximum
        likelihood.  If `endog` is not provided, `nobs`must be
        provided to obtain standard errors.

        These are asymptotic standard errors.  See Bai and Li (2012)
        for conditions under which the standard errors are valid.
        """

        if self.fa_method.lower() != "ml":
            msg = "Standard errors only available under ML estimation"
            raise ValueError(msg)

        if self.nobs is None:
            msg = "nobs is required to obtain standard errors."
            raise ValueError(msg)

        v = self.uniqueness**2 * (2 + kurt)
        return np.sqrt(v / self.nobs)

    @cache_readonly
    def load_stderr(self):
        """
        The standard errors of the loadings.

        Standard errors are only available if the model was fit using
        maximum likelihood.  If `endog` is not provided, `nobs`must be
        provided to obtain standard errors.

        These are asymptotic standard errors.  See Bai and Li (2012)
        for conditions under which the standard errors are valid.
        """

        if self.fa_method.lower() != "ml":
            msg = "Standard errors only available under ML estimation"
            raise ValueError(msg)

        if self.nobs is None:
            msg = "nobs is required to obtain standard errors."
            raise ValueError(msg)

        v = np.outer(self.uniqueness, np.ones(self.loadings.shape[1]))
        return np.sqrt(v / self.nobs)
