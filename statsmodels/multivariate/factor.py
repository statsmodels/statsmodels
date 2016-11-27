# -*- coding: utf-8 -*-

from statsmodels.base.model import Model
from .factor_rotation import rotate_factors, promax
import numpy as np
from numpy.linalg import eig, inv, norm, matrix_rank
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.iolib import summary2


class Factor(Model):
    def __init__(self, endog, n_factor, exog=None, **kwargs):
        self.n_factor = n_factor
        super(Factor, self).__init__(endog, exog)

    def fit(self, n_max_iter=50, tolerance=1e-6, rotation=None,
            SMC=True):
        """
        Fit the factor model

        Parameters
        ----------
        n_max_iter : int
            Maximum number of iterations for communality estimation
        tolerance : float
            If `norm(communality - last_communality)  < tolerance`,
            estimation stops
        rotation : string
            rotation to be applied
        SMC : True or False
            Whether or not to apply squared multiple correlations

        -------

        """
        R = pd.DataFrame(self.endog).corr().values
        self.n_comp = matrix_rank(R)

        #  Initial communality estimation
        if SMC:
            c = 1 - 1 / np.diag(inv(R))
            self.SMC = np.array(c)
        else:
            c = np.ones([1, len(R)])

        # Iterative communality estimation
        eigenvals = None
        for i in range(n_max_iter):
            # Get eigenvalues/eigenvectors of R with diag replaced by
            # communality
            for j in range(len(R)):
                R[j, j] = c[j]
            L, V = eig(R)
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
            if norm(c_last - c) < tolerance:
                break
        self.eigenvals = eigenvals

        # Perform rotation of the loadings
        self.loadings_no_rot = np.array(A)
        if rotation in ['varimax', 'quartimax', 'biquartimax', 'equamax',
                        'parsimax', 'parsimony', 'biquartimin']:
            A, T = rotate_factors(A, rotation)
        elif rotation == 'oblimin':
            A, T = rotate_factors(A, 'quartimin')
        elif rotation == 'promax':
            A, T = promax(A)
        if rotation is not None:  # Rotated
            c = np.power(A, 2).sum(axis=1)
        self.communality = c
        self.loadings = A
        self.rotation = rotation

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
        fig = plt.figure()
        ncomp = self.n_comp if ncomp is None else ncomp
        vals = np.asarray(self.eigenvals)
        vals = vals[:ncomp]
        #    vals = np.cumsum(vals)

        ax = fig.add_subplot(121)
        ax.plot(np.arange(ncomp), vals[: ncomp], 'b-o')
        ax.autoscale(tight=True)
        xlim = np.array(ax.get_xlim())
        sp = xlim[1] - xlim[0]
        xlim += 0.02 * np.array([-sp, sp])
        ax.set_xticks(np.arange(ncomp))
        ax.set_xlim(xlim)

        ylim = np.array(ax.get_ylim())
        scale = 0.02
        sp = ylim[1] - ylim[0]
        ylim += scale * np.array([-sp, sp])
        ax.set_ylim(ylim)
        ax.set_title('Scree Plot')
        ax.set_ylabel('Eigenvalue')
        ax.set_xlabel('Factor')

        per_variance = vals / self.n_comp
        cumper_variance = np.cumsum(per_variance)
        ax = fig.add_subplot(122)

        ax.plot(np.arange(ncomp), per_variance[: ncomp], 'b-o')
        ax.plot(np.arange(ncomp), cumper_variance[: ncomp], 'g--o')
        ax.autoscale(tight=True)
        xlim = np.array(ax.get_xlim())
        sp = xlim[1] - xlim[0]
        xlim += 0.02 * np.array([-sp, sp])
        ax.set_xticks(np.arange(ncomp))
        ax.set_xlim(xlim)

        ylim = np.array(ax.get_ylim())
        scale = 0.02
        sp = ylim[1] - ylim[0]
        ylim += scale * np.array([-sp, sp])
        ax.set_ylim(ylim)
        ax.set_title('Variance Explained')
        ax.set_ylabel('Proportion')
        ax.set_xlabel('Factor')
        ax.legend(['Proportion', 'Cumulative'], loc=5)
        fig.tight_layout()
        return fig

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
        if self.rotation is None:
            plot_prerotated = True
        loadings = self.loadings_no_rot if plot_prerotated else self.loadings
        if loading_pairs is None:
            loading_pairs = []
            for i in range(self.n_factor):
                for j in range(i+1, self.n_factor):
                    loading_pairs.append([i, j])
        figs = []
        for item in loading_pairs:
            i = item[0]
            j = item[1]
            fig = plt.figure(figsize=(7, 7))
            figs.append(fig)
            ax = fig.add_subplot(111)
            for k in range(loadings.shape[0]):
                plt.text(loadings[k, i], loadings[k, j],
                         self.endog_names[k], fontsize=12)
            ax.plot(loadings[:, i], loadings[:, j], 'bo')
            if plot_prerotated:
                ax.set_title('Prerotated Factor Pattern')
            else:
                ax.set_title('%s Rotated Factor Pattern' % (self.rotation))
            ax.set_xlabel('Factor %d (%.1f%%)' %
                          (i, self.eigenvals[i]/self.n_comp * 100))
            ax.set_ylabel('Factor %d (%.1f%%)' %
                          (j, self.eigenvals[j]/self.n_comp * 100))
            v = 1.05
            xlim = np.array([-v, v])
            ylim = np.array([-v, v])
            ax.plot(xlim, [0, 0], 'k--')
            ax.plot([0, 0], ylim, 'k--')
            ax.set_aspect('equal', 'datalim')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            fig.tight_layout()
        return figs
