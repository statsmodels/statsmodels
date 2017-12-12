# -*- coding: utf-8 -*-

from statsmodels.base.model import Model
from .factor_rotation import rotate_factors, promax

try:
    import matplotlib.pyplot as plt
    missing_matplotlib = False
except ImportError:
    missing_matplotlib = True

if not missing_matplotlib:
    from .plots import plot_scree, plot_loadings

import numpy as np
from numpy.linalg import eigh, inv, norm, matrix_rank
import pandas as pd

from statsmodels.iolib import summary2


class Factor(Model):
    """
    Factor analysis
    Status: experimental

    .. [1] Hofacker, C. (2004). Exploratory Factor Analysis, Mathematical Marketing.
    http://www.openaccesstexts.org/pdf/Quant_Chapter_11_efa.pdf

    Supported rotations:
        'varimax', 'quartimax', 'biquartimax', 'equamax', 'oblimin',
        'parsimax', 'parsimony', 'biquartimin', 'promax'

    Parameters
    ----------
    endog : array-like
        Variables in columns, observations in rows
        Could be `None` if `corr` is not `None`
    n_factor : int
        The number of factors to extract
    corr : array-like
        Directly specify the correlation matrix instead of estimating from endog
        If not `None`, `endog` will not be used
    method : str
        Specify the method to extract factors
        'pa' - Principal axis factor analysis
    smc : True or False
        Whether or not to apply squared multiple correlations
    endog_names: str
        Names of endogeous variables.
        If specified, it will be used instead of the column names in endog
    nobs: int
        The number of observations. To be used together with `corr`
        Should be equals to the number of rows in `endog`.

    """
    def __init__(self, endog, n_factor, corr=None, method='pa', smc=True,
                 missing='drop', endog_names=None, nobs=None):
        if endog is not None:
            k_endog = endog.shape[1]
        elif corr is not None:
            k_endog = corr.shape[0]

        # Check validity of n_factor
        if n_factor <= 0:
            raise ValueError('n_factor must be larger than 0! %d < 0' %
                             (n_factor))
        if endog is not None and n_factor > k_endog:
            raise ValueError('n_factor must be smaller or equal to the number'
                             ' of columns of endog! %d > %d' %
                             (n_factor, k_endog))
        self.n_factor = n_factor

        if corr is None and endog is None:
            raise ValueError('Both endog and corr is None!')

        self.loadings = None
        self.communality = None
        self.eigenvals = None
        self.method = method
        self.smc = smc

        # Check validity of corr
        if corr is not None:
            if corr.shape[0] != corr.shape[1]:
                raise ValueError('Correlation matrix corr must be a square '
                                 '(rows %d != cols %d)' % corr.shape)
            if endog is not None and k_endog != corr.shape[0]:
                    raise ValueError('The number of columns in endog (=%d) must be '
                                     'equal to the number of columns and rows corr (=%d)'
                                     % (k_endog, corr.shape[0]))
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

        # Check validity of n_obs
        if nobs is not None:
            if endog is not None and endog.shape[0] != nobs:
                raise ValueError('n_obs must be equal to the number of rows in endog')

        # Do not preprocess endog if None
        if endog is not None:
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
                return [('var%0' + str(d) + 'd') % i for i in range(self.corr.shape[0])]

    @endog_names.setter
    def endog_names(self, value):
        # Check validity of endog_names:
        if value is not None:
            if self.corr is not None and len(value) != self.corr.shape[0]:
                raise ValueError('The number of elements in endog_names must '
                                 'be equal to the number of columns and rows in corr')
            if self.endog is not None and len(value) != self.endog.shape[1]:
                raise ValueError('The number of elements in endog_names must '
                                 'be equal to the number of columns in endog')
            self._endog_names = np.asarray(value)
        else:
            self._endog_names = None

    def fit(self, maxiter=50, tol=1e-8):
        """
        Extract factors

        Parameters
        ----------
        maxiter : int
            Maximum number of iterations for iterative estimation algorithms
        tol : float
            Stopping critera (error tolerance) for iterative estimation algorithms

        Returns
        -------
        results: FactorResults

        """
        if self.method == 'pa':
            return self._fit_pa(maxiter=maxiter, tol=tol)
        else:
            raise ValueError("Unknown factor extraction approach '%s'" % self.method)

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
        self.loadings = A
        return FactorResults(self)


class FactorResults(object):
    """
    Factor results class
    For result summary, scree/loading plots and factor rotations
    Status: experimental

    Parameters
    ----------
    factor : Factor
        Fitted Factor class

    """
    def __init__(self, factor):
        if not isinstance(factor, Factor):
            raise ValueError('Input must be a `Factor` class. Got %s instead'
                             % (factor.__str__))
        self.endog_names = factor.endog_names
        self.loadings_no_rot = factor.loadings
        self.loadings = factor.loadings
        self.eigenvals = factor.eigenvals
        self.communality = factor.communality
        self.rotation_method = None
        self.fa_method = factor.method
        self.n_comp = factor.n_comp

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
            self.loadings, T = rotate_factors(self.loadings_no_rot, 'quartimin')
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
        eigenvals = pd.DataFrame([self.eigenvals], columns=self.endog_names,
                                 index=[''])
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
        if self.rotation is not None:
            loadings = pd.DataFrame(
                self.loadings,
                columns=["factor %d" % (i)
                         for i in range(self.loadings.shape[1])],
                index=self.endog_names
            )
            summ.add_dict({'': '%s rotated loadings' % (self.rotation)})
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
