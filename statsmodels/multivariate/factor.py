# -*- coding: utf-8 -*-

from statsmodels.base.model import Model
from .factor_rotation import rotate_factors, promax
import numpy as np
from numpy.linalg import eig, inv, norm
import pandas as pd
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

        #  Initial communality estimation
        if SMC:
            c = 1 - 1 / np.diag(inv(R))
        else:
            c = np.ones([1, len(R)])
        # Iterative communality estimation
        for i in range(n_max_iter):
            for j in range(len(R)):
                R[j, j] = c[j]
            c_last = np.array(c)
            L, V = eig(R)
            ind = np.argsort(L)
            ind = ind[::-1]
            L = L[ind]
            n_pos = (L > 0).sum()
            n = np.min([n_pos, self.n_factor])
            V = V[:, ind]
            sL = np.diag(np.sqrt(L[:n]))
            V = V[:, :n]
            A = V.dot(sL)
            c = np.power(A, 2).sum(axis=1)
            if norm(c_last - c) < tolerance:
                break
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
