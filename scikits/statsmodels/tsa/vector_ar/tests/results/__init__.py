"""
Impulse reponse-related code
"""

from __future__ import division

import numpy as np
import numpy.linalg as npl
import scipy.linalg as L

from scikits.statsmodels.decorators import cache_readonly
from scikits.statsmodels.tools.tools import chain_dot

import scikits.statsmodels.tsa.tsatools as tsa
import scikits.statsmodels.tsa.var.plotting as plotting
import scikits.statsmodels.tsa.var.util as util

mat = np.array

class BaseIRAnalysis(object):
    """
    Base class for plotting and computing IRF-related statistics, want to be
    able to handle known and estimated processes
    """

    def __init__(self, model, P=None, periods=10):
        self.model = model
        self.periods = periods
        self.k, self.lags, self.T  = model.k, model.p, model.T

        if P is None:
            P = model._chol_sigma_u
        self.P = P

        self.irfs = model.ma_rep(periods)
        self.orth_irfs = model.orth_ma_rep(periods)

        self.cum_effects = self.irfs.cumsum(axis=0)
        self.orth_cum_effects = self.orth_irfs.cumsum(axis=0)

        self.lr_effects = model.long_run_effects()
        self.orth_lr_effects = np.dot(model.long_run_effects(), P)

        # auxiliary stuff
        self._A = util.comp_matrix(model.coefs)

    def cov(self, *args, **kwargs):
        raise NotImplementedError

    def cum_effect_cov(self, *args, **kwargs):
        raise NotImplementedError

    def plot(self, orth=False, impcol=None, rescol=None, signif=0.05,
             plot_params=None, subplot_params=None):
        """
        Plot impulse responses

        Parameters
        ----------
        orth : bool, default False
            Compute orthogonalized impulse responses
        impcol : string or int
            variable providing the impulse
        rescol : string or int
            variable affected by the impulse
        signif : float (0 < signif < 1)
            Significance level for error bars, defaults to 95% CI
        subplot_params : dict
            To pass to subplot plotting funcions. Example: if fonts are too big,
            pass {'fontsize' : 8} or some number to your taste.
        plot_params : dict
        """
        if orth:
            title = 'Impulse responses (orthogonalized)'
            irfs = self.orth_irfs
        else:
            title = 'Impulse responses'
            irfs = self.irfs

        try:
            stderr = self.cov(orth=orth)
        except NotImplementedError:
            stderr = None

        plotting.irf_grid_plot(irfs, stderr, impcol, rescol, self.model.names,
                               title, signif=signif,
                               subplot_params=subplot_params,
                               plot_params=plot_params)

    def plot_cum_effects(self, orth=False, impcol=None, rescol=None,
                         signif=0.05, plot_params=None,
                         subplot_params=None):
        """

        """

        if orth:
            title = 'Cumulative responses responses (orthogonalized)'
            cum_effects = self.orth_cum_effects
            lr_effects = self.orth_lr_effects
        else:
            title = 'Cumulative responses'
            cum_effects = self.cum_effects
            lr_effects = self.lr_effects

        try:
            stderr = self.cum_effect_cov(orth=orth)
        except NotImplementedError:
            stderr = None

        plotting.irf_grid_plot(cum_effects, stderr, impcol, rescol,
                               self.model.names, title, signif=signif,
                               hlines=lr_effects, subplot_params=subplot_params,
                               plot_params=plot_params)

class IRAnalysis(BaseIRAnalysis):
    """
    Impulse response analysis class. Computes impulse responses, asymptotic
    standard errors, and produces relevant plots

    Parameters
    ----------
    model : VAR instance

    Notes
    -----
    Using Lutkepohl (2005) notation
    """
    def __init__(self, model, P=None, periods=10):
        BaseIRAnalysis.__init__(self, model, P=P, periods=periods)

        self.cov_a = model._cov_alpha
        self.cov_sig = model._cov_sigma

        # memoize dict for G matrix function
        self._g_memo = {}

    def cov(self, orth=False):
        """

        Notes
        -----
        Lutkepohl eq 3.7.5

        Returns
        -------
        """
        if orth:
            return self._orth_cov()

        covs = self._empty_covm(self.periods + 1)
        covs[0] = np.zeros((self.k ** 2, self.k ** 2))
        for i in range(1, self.periods + 1):
            Gi = self.G[i - 1]
            covs[i] = chain_dot(Gi, self.cov_a, Gi.T)

        return covs

    def _orth_cov(self):
        """

        Notes
        -----
        Lutkepohl 3.7.8

        Returns
        -------

        """
        Ik = np.eye(self.k)
        PIk = np.kron(self.P.T, Ik)
        H = self.H

        covs = self._empty_covm(self.periods + 1)
        for i in range(self.periods + 1):
            if i == 0:
                apiece = 0
            else:
                Ci = np.dot(PIk, self.G[i-1])
                apiece = chain_dot(Ci, self.cov_a, Ci.T)

            Cibar = np.dot(np.kron(Ik, self.irfs[i]), H)
            bpiece = chain_dot(Cibar, self.cov_sig, Cibar.T) / self.T

            # Lutkepohl typo, cov_sig correct
            covs[i] = apiece + bpiece

        return covs

    def cum_effect_cov(self, orth=False):
        """

        Parameters
        ----------
        orth : boolean

        Notes
        -----
        eq. 3.7.7 (non-orth), 3.7.10 (orth)

        Returns
        -------

        """
        Ik = np.eye(self.k)
        PIk = np.kron(self.P.T, Ik)

        F = 0.
        covs = self._empty_covm(self.periods + 1)
        for i in range(self.periods + 1):
            if i > 0:
                F = F + self.G[i - 1]

            if orth:
                if i == 0:
                    apiece = 0
                else:
                    Bn = np.dot(PIk, F)
                    apiece = chain_dot(Bn, self.cov_a, Bn.T)

                Bnbar = np.dot(np.kron(Ik, self.cum_effects[i]), self.H)
                bpiece = chain_dot(Bnbar, self.cov_sig, Bnbar.T) / self.T

                covs[i] = apiece + bpiece
            else:
                if i == 0:
                    covs[i] = np.zeros((self.k**2, self.k**2))
                    continue

                covs[i] = chain_dot(F, self.cov_a, F.T)

        return covs

    def lr_effect_cov(self, orth=False):
        """

        Returns
        -------

        """
        lre = self.lr_effects
        Finfty = np.kron(np.tile(lre.T, self.lags), lre)

        if orth:
            Binf = np.dot(np.kron(self.P.T, np.eye(self.k)), Finfty)
            Binfbar = np.dot(np.kron(np.eye(self.k), self.lr_effects), self.H)

            return (chain_dot(Binf, self.cov_a, Binf.T) +
                    chain_dot(Binfbar, self.cov_a, Binfbar.T))
        else:
            return chain_dot(Finfty, self.cov_a, Finfty.T)

    def _empty_covm(self, periods):
        return np.zeros((periods, self.k ** 2, self.k ** 2),
                        dtype=float)

    @cache_readonly
    def G(self):
        def _make_g(i):
            # p. 111 Lutkepohl
            G = 0.
            for m in range(i):
                # be a bit cute to go faster
                idx = i - 1 - m
                if idx in self._g_memo:
                    apow = self._g_memo[idx]
                else:
                    apow = npl.matrix_power(self._A.T, idx)[:self.k]

                    self._g_memo[idx] = apow

                # take first K rows
                piece = np.kron(apow, self.irfs[m])
                G = G + piece

            return G

        return [_make_g(i) for i in range(1, self.periods + 1)]

    @cache_readonly
    def H(self):
        k = self.k
        Lk = tsa.elimination_matrix(k)
        Kkk = tsa.commutation_matrix(k, k)
        Ik = np.eye(k)

        # B = chain_dot(Lk, np.eye(k**2) + commutation_matrix(k, k),
        #               np.kron(self.P, np.eye(k)), Lk.T)

        # return np.dot(Lk.T, L.inv(B))

        B = chain_dot(Lk,
                      np.dot(np.kron(Ik, self.P), Kkk) + np.kron(self.P, Ik),
                      Lk.T)

        return np.dot(Lk.T, L.inv(B))

    def fevd_table(self):
        pass

if __name__ == '__main__':
    pass
