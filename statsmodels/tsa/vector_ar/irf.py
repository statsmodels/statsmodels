"""Impulse response-related code"""

from statsmodels.compat.pandas import deprecate_kwarg

import warnings

import numpy as np
import numpy.linalg as la
import scipy.linalg as L

from statsmodels.tools._decorators import cache_readonly
from statsmodels.tools.validation import string_like
import statsmodels.tsa.tsatools as tsa
from statsmodels.tsa.vector_ar import plotting, util
from statsmodels.tsa.vector_ar.hypothesis_test_results import ErrorBand

mat = np.array


def _check_component(component, n_components):
    # The indices of the eigenvalues used by the Sims-Zha bands must be in
    # [0, n_components); negative indices count from the end and are deprecated
    component = np.asarray(component)
    if np.any(component >= n_components) or np.any(component < -n_components):
        raise ValueError(
            "At least one of the components does not exist. Components must be "
            f"between 0 and {n_components - 1}."
        )
    if np.any(component < 0):
        warnings.warn(
            "Negative values in component select eigenvalues from the end. They "
            "are deprecated and will raise a ValueError after statsmodels 0.17 "
            "is released.",
            FutureWarning,
            stacklevel=4,
        )
    return component


class BaseIRAnalysis:
    """
    Base class for plotting and computing IRF-related statistics, designed
    to handle both known and estimated processes

    Parameters
    ----------
    model : VARResults, SVARResults, or VECMResults
        The fitted model results used to compute impulse responses.
    P : ndarray, optional
        The matrix used for orthogonalization, satisfying sigma_u = P P'.
        If None, computed as the Cholesky decomposition of the model's
        residual covariance matrix.
    periods : int, optional
        Number of periods to compute the impulse responses for. The
        default is 10.
    order : sequence, optional
        Alternate variable order for the Cholesky decomposition. Not
        currently implemented.
    svar : bool, optional
        Flag indicating whether the model is a structural VAR. The
        default is False.
    vecm : bool, optional
        Flag indicating whether the model is a VECM. The default is
        False.
    """

    def __init__(self, model, P=None, periods=10, order=None, svar=False, vecm=False):
        self.model = model
        self.periods = periods
        self.neqs, self.lags, self.T = model.neqs, model.k_ar, model.nobs

        self.order = order

        if P is None:
            sigma = model.sigma_u

            # TODO, may be difficult at the moment
            # if order is not None:
            #     indexer = [model.get_eq_index(name) for name in order]
            #     sigma = sigma[:, indexer][indexer, :]

            #     if sigma.shape != model.sigma_u.shape:
            #         raise ValueError('variable order is wrong length')

            P = la.cholesky(sigma)

        self.P = P

        self.svar = svar

        self.irfs = model.ma_rep(periods)
        if svar:
            self.svar_irfs = model.svar_ma_rep(periods, P=P)
        else:
            self.orth_irfs = model.orth_ma_rep(periods, P=P)

        self.cum_effects = self.irfs.cumsum(axis=0)
        if svar:
            self.svar_cum_effects = self.svar_irfs.cumsum(axis=0)
        else:
            self.orth_cum_effects = self.orth_irfs.cumsum(axis=0)

        # long-run effects may be infinite for VECMs.
        if not vecm:
            self.lr_effects = model.long_run_effects()
            if svar:
                self.svar_lr_effects = np.dot(model.long_run_effects(), P)
            else:
                self.orth_lr_effects = np.dot(model.long_run_effects(), P)

        # auxiliary stuff
        if vecm:
            self._A = util.comp_matrix(model.var_rep)
        else:
            self._A = util.comp_matrix(model.coefs)

    def _choose_irfs(self, orth=False, svar=False):
        if orth:
            return self.orth_irfs
        elif svar:
            return self.svar_irfs
        else:
            return self.irfs

    def cov(self, *args, **kwargs):
        """Not implemented in the base class."""
        raise NotImplementedError

    def cum_effect_cov(self, *args, **kwargs):
        """Not implemented in the base class."""
        raise NotImplementedError

    @deprecate_kwarg("seed", "rng")
    def plot(
        self,
        orth=False,
        *,
        impulse=None,
        response=None,
        signif=0.05,
        plot_params=None,
        figsize=(10, 10),
        subplot_params=None,
        plot_stderr=True,
        stderr_type="asym",
        repl=1000,
        rng=None,
        component=None,
        err_bands=None,
    ):
        """
        Plot impulse responses

        Parameters
        ----------
        orth : bool, optional
            Compute orthogonalized impulse responses. The default is False.
        impulse : str or int, optional
            Name or index of the variable providing the impulse. If None,
            plots impulses from all variables.
        response : str or int, optional
            Name or index of the variable affected by the impulse. If None,
            plots responses of all variables.
        signif : float, optional
            Significance level for the confidence interval, between 0 and
            1. The default is 0.05, giving a 95% confidence interval.
        subplot_params : dict, optional
            Only the key ``"fontsize"`` is used, as the font size of the
            subplot titles. The default font size is 12.
        plot_params : dict, optional
            Properties of :class:`matplotlib.lines.Line2D` for the response
            curves, such as ``color`` and ``linewidth``. A key that is not a
            ``Line2D`` property raises an ``AttributeError``. Does not affect
            error bands or horizontal reference lines.

            .. versionchanged:: 0.15.1

               ``plot_params`` was ignored before.
        figsize : tuple of float, optional
            Figure size (width, height in inches). The default is (10, 10).
        plot_stderr : bool, optional
            Plot standard impulse response error bands. The default is
            True.

            .. versionchanged:: 0.15.1

               Any false value, such as ``numpy.False_``, turns the error
               bands off. Only ``False`` did.
        stderr_type : {"asym", "mc", "sz1", "sz2", "sz3"}, optional
            The method used to compute the error bands. "asym" computes
            asymptotic standard errors and is the default. "mc" computes
            Monte Carlo standard errors using ``repl`` replications.
            "sz1", "sz2", and "sz3" compute Sims-Zha error bands using
            replications and, optionally, ``component``. The "sz1" bands
            (``err_band_sz1``) are symmetric around the impulse responses.
            The "sz2" and "sz3" bands (``err_band_sz2``, ``err_band_sz3``)
            add to the impulse responses the quantiles of the simulated
            responses weighted elementwise by a row of the eigenvector
            matrix of their covariance, of each response ("sz2") or of the
            responses of all variables to a shock ("sz3"), so they need not
            contain the impulse responses.
        repl : int, optional
            Number of replications for Monte Carlo and Sims-Zha standard
            errors. The default is 1000.
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
        component : ndarray of int, optional
            Index of the principal component to use for the Sims-Zha
            "sz1"/"sz2" (shape (neqs, neqs)) or "sz3" (length neqs) error
            bands. If None, the component with the largest eigenvalue is
            used for each element. Ignored unless ``stderr_type`` is
            "sz1", "sz2", or "sz3".
        err_bands : ndarray, ErrorBand, or tuple of ndarray, optional
            Pre-computed error bands, an array of shape
            (2, periods + 1, neqs, neqs), an ``ErrorBand``, such as the one
            returned by :meth:`errband_mc`, or a tuple of the lower and upper
            bands. The first dimension contains the lower and upper bounds of
            the confidence interval, respectively. If provided, the internal
            calculation of standard errors is bypassed and ``stderr_type`` is
            used only for plot formatting.
        """
        svar = self.svar

        if orth and svar:
            raise ValueError("For SVAR system, set orth=False")

        irfs = self._choose_irfs(orth, svar)
        if orth:
            title = "Impulse responses (orthogonalized)"
        elif svar:
            title = "Impulse responses (structural)"
        else:
            title = "Impulse responses"

        stderr_type = string_like(
            stderr_type,
            "stderr_type",
            options=("asym", "mc", "sz1", "sz2", "sz3"),
            lower=False,
        )

        if not plot_stderr:
            stderr = None
        elif err_bands is not None:
            if isinstance(err_bands, ErrorBand):
                err_bands = (err_bands.lower, err_bands.upper)
            expected_shape = (2, self.periods + 1, self.neqs, self.neqs)
            if np.asarray(err_bands).shape != expected_shape:
                raise ValueError(
                    f"err_bands has shape {np.asarray(err_bands).shape}, expected "
                    f"{expected_shape} (2, periods+1, neqs, neqs)."
                )
            stderr = err_bands
            if stderr_type == "asym":
                # the bands are plotted as given
                stderr_type = "mc"
        elif stderr_type == "asym":
            stderr = self.cov(orth=orth)
        elif stderr_type == "mc":
            _err_band = self.errband_mc(
                orth=orth, svar=svar, repl=repl, signif=signif, rng=rng
            )
            stderr = (_err_band.lower, _err_band.upper)
        elif stderr_type == "sz1":
            _err_band = self.err_band_sz1(
                orth=orth,
                svar=svar,
                repl=repl,
                signif=signif,
                rng=rng,
                component=component,
            )
            stderr = (_err_band.lower, _err_band.upper)
        elif stderr_type == "sz2":
            _err_band = self.err_band_sz2(
                orth=orth,
                svar=svar,
                repl=repl,
                signif=signif,
                rng=rng,
                component=component,
            )
            stderr = (_err_band.lower, _err_band.upper)
        else:  # stderr_type == "sz3":
            _err_band = self.err_band_sz3(
                orth=orth,
                svar=svar,
                repl=repl,
                signif=signif,
                rng=rng,
                component=component,
            )
            stderr = (_err_band.lower, _err_band.upper)

        fig = plotting.irf_grid_plot(
            irfs,
            stderr,
            impulse,
            response,
            self.model.names,
            title,
            signif=signif,
            subplot_params=subplot_params,
            plot_params=plot_params,
            figsize=figsize,
            stderr_type=stderr_type,
        )
        return fig

    @deprecate_kwarg("seed", "rng")
    def plot_cum_effects(
        self,
        orth=False,
        *,
        impulse=None,
        response=None,
        signif=0.05,
        plot_params=None,
        figsize=(10, 10),
        subplot_params=None,
        plot_stderr=True,
        stderr_type="asym",
        repl=1000,
        rng=None,
        err_bands=None,
    ):
        """
        Plot cumulative impulse response functions

        Parameters
        ----------
        orth : bool, optional
            Compute orthogonalized impulse responses. The default is False.
        impulse : str or int, optional
            Name or index of the variable providing the impulse. If None,
            plots impulses from all variables.
        response : str or int, optional
            Name or index of the variable affected by the impulse. If None,
            plots responses of all variables.
        signif : float, optional
            Significance level for the confidence interval, between 0 and
            1. The default is 0.05, giving a 95% confidence interval.
        subplot_params : dict, optional
            Only the key ``"fontsize"`` is used, as the font size of the
            subplot titles. The default font size is 12.
        plot_params : dict, optional
            Properties of :class:`matplotlib.lines.Line2D` for the response
            curves, such as ``color`` and ``linewidth``. A key that is not a
            ``Line2D`` property raises an ``AttributeError``. Does not affect
            error bands or horizontal reference lines.

            .. versionchanged:: 0.15.1

               ``plot_params`` was ignored before.
        figsize : tuple of float, optional
            Figure size (width, height in inches). The default is (10, 10).
        plot_stderr : bool, optional
            Plot standard impulse response error bands. The default is
            True.
        stderr_type : {"asym", "mc"}, optional
            The method used to compute the error bands. "asym" computes
            asymptotic standard errors and is the default. "mc" computes
            Monte Carlo standard errors using ``repl`` replications.
        repl : int, optional
            Number of replications for Monte Carlo standard errors. The
            default is 1000.
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
        err_bands : ndarray, ErrorBand, or tuple of ndarray, optional
            Pre-computed error bands, an array of shape
            (2, periods + 1, neqs, neqs), an ``ErrorBand``, such as the one
            returned by :meth:`cum_errband_mc`, or a tuple of the lower and
            upper bands. The first dimension contains the lower and upper
            bounds of the confidence interval, respectively. If provided, the
            internal calculation of standard errors is bypassed and
            ``stderr_type`` is used only for plot formatting.
        """

        if orth:
            title = "Cumulative responses (orthogonalized)"
            cum_effects = self.orth_cum_effects
            lr_effects = self.orth_lr_effects
        else:
            title = "Cumulative responses"
            cum_effects = self.cum_effects
            lr_effects = self.lr_effects

        if not plot_stderr:
            stderr = None
        elif err_bands is not None:
            if isinstance(err_bands, ErrorBand):
                err_bands = (err_bands.lower, err_bands.upper)
            expected_shape = (2, self.periods + 1, self.neqs, self.neqs)
            if np.asarray(err_bands).shape != expected_shape:
                raise ValueError(
                    f"err_bands has shape {np.asarray(err_bands).shape}, expected "
                    f"{expected_shape} (2, periods+1, neqs, neqs)."
                )
            stderr = err_bands
            if stderr_type == "asym":
                # the bands are plotted as given
                stderr_type = "mc"
        else:
            stderr_type = string_like(
                stderr_type, "stderr_type", options=("asym", "mc"), lower=False
            )
            if stderr_type == "asym":
                stderr = self.cum_effect_cov(orth=orth)
            else:  # stderr_type == "mc"
                _err_band = self.cum_errband_mc(
                    orth=orth, repl=repl, signif=signif, rng=rng
                )
                stderr = (_err_band.lower, _err_band.upper)

        fig = plotting.irf_grid_plot(
            cum_effects,
            stderr,
            impulse,
            response,
            self.model.names,
            title,
            signif=signif,
            hlines=lr_effects,
            subplot_params=subplot_params,
            plot_params=plot_params,
            figsize=figsize,
            stderr_type=stderr_type,
        )
        return fig


class IRAnalysis(BaseIRAnalysis):
    """
    Impulse response analysis class. Computes impulse responses, asymptotic
    standard errors, and produces relevant plots

    Parameters
    ----------
    model : VARResults, SVARResults, or VECMResults
        The fitted model results used to compute impulse responses.
    P : ndarray, optional
        The matrix used for orthogonalization, satisfying sigma_u = P P'.
        If None, computed as the Cholesky decomposition of the model's
        residual covariance matrix.
    periods : int, optional
        Number of periods to compute the impulse responses for. The
        default is 10.
    order : sequence, optional
        Alternate variable order for the Cholesky decomposition. Not
        currently implemented.
    svar : bool, optional
        Flag indicating whether the model is a structural VAR. The
        default is False.
    vecm : bool, optional
        Flag indicating whether the model is a VECM. The default is
        False.

    Attributes
    ----------
    irfs : ndarray
        The impulse responses, the MA coefficient matrices, of shape
        (periods + 1, neqs, neqs). ``irfs[h, i, j]`` is the response of
        variable ``i`` at horizon ``h`` to a unit shock in variable ``j``.
    orth_irfs : ndarray
        The orthogonalized impulse responses ``irfs[h] @ P``, of shape
        (periods + 1, neqs, neqs). Not available if `svar` is True.
    svar_irfs : ndarray
        The structural impulse responses ``irfs[h] @ P``, of shape
        (periods + 1, neqs, neqs). Only available if `svar` is True.
    cum_effects : ndarray
        The cumulative impulse responses, the sums of ``irfs`` over the
        horizons 0 to h, of shape (periods + 1, neqs, neqs).
    orth_cum_effects : ndarray
        The cumulative orthogonalized impulse responses. Not available if
        `svar` is True.
    svar_cum_effects : ndarray
        The cumulative structural impulse responses. Only available if
        `svar` is True.
    lr_effects : ndarray
        The long-run effects ``inv(I - A_1 - ... - A_p)``, the limit of
        `cum_effects`, of shape (neqs, neqs). Not available if `vecm` is
        True.
    orth_lr_effects : ndarray
        The orthogonalized long-run effects ``lr_effects @ P``. Not
        available if `svar` or `vecm` is True.
    svar_lr_effects : ndarray
        The structural long-run effects ``lr_effects @ P``. Only available
        if `svar` is True and `vecm` is False.
    P : ndarray
        The (neqs, neqs) matrix used for orthogonalization. For a
        structural VAR, ``inv(A) @ B``.
    periods : int
        The number of periods after the impact period.
    neqs : int
        The number of variables.

    Notes
    -----
    Using Lütkepohl (2005) notation
    """

    def __init__(self, model, P=None, periods=10, order=None, svar=False, vecm=False):
        BaseIRAnalysis.__init__(
            self, model, P=P, periods=periods, order=order, svar=svar, vecm=vecm
        )

        if vecm:
            self.cov_a = model.cov_var_repr
        else:
            self.cov_a = model._cov_alpha
        self.cov_sig = model._cov_sigma

        # memoize dict for G matrix function
        self._g_memo = {}

    def cov(self, orth=False):
        """
        Compute asymptotic standard errors for impulse response coefficients

        Parameters
        ----------
        orth : bool, optional
            Compute orthogonalized impulse responses. The default is False.

        Returns
        -------
        ndarray
            Array of shape (periods + 1, neqs ** 2, neqs ** 2) containing
            the covariance matrix of the impulse response coefficients for
            each period.

        Notes
        -----
        Lütkepohl eq 3.7.5
        """
        if orth:
            return self._orth_cov()

        covs = self._empty_covm(self.periods + 1)
        covs[0] = np.zeros((self.neqs**2, self.neqs**2))
        for i in range(1, self.periods + 1):
            Gi = self.G[i - 1]
            covs[i] = Gi @ self.cov_a @ Gi.T

        return covs

    @deprecate_kwarg("seed", "rng")
    def errband_mc(
        self, orth=False, svar=False, repl=1000, signif=0.05, rng=None, burn=100
    ):
        """
        IRF Monte Carlo integrated error bands

        Parameters
        ----------
        orth : bool, optional
            Compute orthogonalized impulse responses. The default is False.
        svar : bool, optional
            Compute structural impulse responses. The default is False.
        repl : int, optional
            Number of MC replications. The default is 1000.
        signif : float, optional
            Significance level for the confidence interval, between 0 and
            1. The default is 0.05, giving a 95% confidence interval.
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
            Number of initial simulated obs to discard. The default is 100.

        Returns
        -------
        ErrorBand
            A result object with fields ``lower`` and ``upper``.
        """
        model = self.model
        periods = self.periods
        if svar:
            return model.sirf_errband_mc(
                orth=orth,
                repl=repl,
                steps=periods,
                signif=signif,
                rng=rng,
                burn=burn,
                cum=False,
            )
        else:
            return model.irf_errband_mc(
                orth=orth,
                repl=repl,
                steps=periods,
                signif=signif,
                rng=rng,
                burn=burn,
                cum=False,
            )

    @deprecate_kwarg("seed", "rng")
    def err_band_sz1(
        self,
        orth=False,
        svar=False,
        repl=1000,
        signif=0.05,
        rng=None,
        burn=100,
        component=None,
    ):
        """
        IRF Sims-Zha error band method 1. Assumes symmetric error bands around
        mean.

        Parameters
        ----------
        orth : bool, optional
            Compute orthogonalized impulse responses. The default is False.
        svar : bool, optional
            Use structural IRFs. The default is False.
        repl : int, optional
            Number of MC replications. The default is 1000.
        signif : float, optional
            Significance level for the confidence interval, between 0 and
            1. The default is 0.05, giving a 95% confidence interval.
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
            Number of initial simulated obs to discard. The default is 100.
        component : array_like of int, optional
            Array of shape (neqs, neqs) giving the index of the
            column of eigenvector/value to use for each error band. Note:
            the period of impulse (t=0) is not included when computing the
            principal component. If None, the column of the largest
            eigenvalue is used for each element. The eigenvalues are in the
            order returned by :func:`scipy.linalg.eig`, which is not sorted.

            .. versionchanged:: 0.15.1

               Each value must be between 0 and periods - 1. A valid
               component could be rejected and a value of periods or more
               could pass the check. Negative values select eigenvalues
               from the end; they are deprecated and issue a FutureWarning.

        Returns
        -------
        ErrorBand
            A result object with fields ``lower`` and ``upper``.

        Notes
        -----
        For the response of variable i to a shock in variable j, ``V`` is
        the covariance matrix of the `repl` simulated responses (from
        ``irf_resim``) in periods 1 to `periods`, and ``W[:, k]`` is the
        eigenvector of its largest eigenvalue ``lam[k]`` (or of the one
        that `component` selects). The bands are the point estimate plus
        (``lower``) and minus (``upper``) ``z * sqrt(lam[k]) * W[:, k]``,
        with ``z`` the ``1 - signif / 2`` quantile of the standard normal
        distribution, and both equal the point estimate in period 0. They
        are symmetric around the point estimate. The sign of an eigenvector
        is arbitrary, so ``lower`` is above ``upper`` in the periods where
        the element of ``W[:, k]`` is positive.

        References
        ----------
        Sims, Christopher A., and Tao Zha. 1999. "Error Bands for Impulse
        Response". Econometrica 67: 1113-1155.
        """

        model = self.model
        periods = self.periods
        irfs = self._choose_irfs(orth, svar)
        neqs = self.neqs
        irf_resim = model.irf_resim(
            orth=orth, repl=repl, steps=periods, rng=rng, burn=burn
        )
        q = util.norm_signif_level(signif)

        W, eigva, k = self._eigval_decomp_SZ(irf_resim)

        if component is not None:
            if np.shape(component) != (neqs, neqs):
                raise ValueError(
                    "Component array must be " + str(neqs) + " x " + str(neqs)
                )
            k = _check_component(component, periods)

        # here take the kth column of W, which we determine by finding the largest eigenvalue of the covaraince matrix
        lower = np.copy(irfs)
        upper = np.copy(irfs)
        for i in range(neqs):
            for j in range(neqs):
                lower[1:, i, j] = irfs[1:, i, j] + W[i, j, :, k[i, j]] * q * np.sqrt(
                    eigva[i, j, k[i, j]]
                )
                upper[1:, i, j] = irfs[1:, i, j] - W[i, j, :, k[i, j]] * q * np.sqrt(
                    eigva[i, j, k[i, j]]
                )

        return ErrorBand(lower, upper)

    @deprecate_kwarg("seed", "rng")
    def err_band_sz2(
        self,
        orth=False,
        svar=False,
        repl=1000,
        signif=0.05,
        rng=None,
        burn=100,
        component=None,
    ):
        """
        IRF Sims-Zha error band method 2

        This method does not assume symmetric error bands around mean.

        Parameters
        ----------
        orth : bool, optional
            Compute orthogonalized impulse responses. The default is False.
        svar : bool, optional
            Use structural IRFs. The default is False.
        repl : int, optional
            Number of MC replications. The default is 1000.
        signif : float, optional
            Significance level for the confidence interval, between 0 and
            1. The default is 0.05, giving a 95% confidence interval.
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
            Number of initial simulated obs to discard. The default is 100.
        component : array_like of int, optional
            Array of shape (neqs, neqs) giving the index of the
            column of eigenvector/value to use for each error band. Note:
            the period of impulse (t=0) is not included when computing the
            principal component. If None, the column of the largest
            eigenvalue is used for each element. The eigenvalues are in the
            order returned by :func:`scipy.linalg.eig`, which is not sorted.

            .. versionchanged:: 0.15.1

               Each value must be between 0 and periods - 1. A valid
               component could be rejected and a value of periods or more
               could pass the check. Negative values select eigenvalues
               from the end; they are deprecated and issue a FutureWarning.

        Returns
        -------
        ErrorBand
            A result object with fields ``lower`` and ``upper``.

        Notes
        -----
        For the response of variable i to a shock in variable j, ``V`` is
        the covariance matrix of the `repl` simulated responses ``c`` (from
        ``irf_resim``) in periods 1 to `periods`, and the columns of ``W``
        are its eigenvectors, in the unsorted order of
        :func:`scipy.linalg.eig`. With ``k`` the index of the largest
        eigenvalue (or the one that `component` selects), the ``k``-th row
        of ``W``, not the ``k``-th eigenvector (its ``k``-th column),
        multiplies each simulated response elementwise,
        ``g = W[k, :] * c``. In each period, ``lower`` and ``upper`` are the
        point estimate plus the ``round(signif / 2 * repl)``-th and the
        ``round((1 - signif / 2) * repl)``-th smallest of the `repl` values
        of ``g``; in period 0 both equal the point estimate.

        So in period t the band is the point estimate plus the quantiles of
        the simulated responses scaled by ``W[k, t]``, the ``k``-th element
        of the t-th eigenvector, whose sign is arbitrary. Eq. (9) of Sims and
        Zha (1999) instead adds to the mean ``m`` of the simulated responses
        ``W[:, k]`` times the quantiles of the component
        ``W[:, k]' (c - m)`` of their deviations from the mean. As the
        simulated responses here are not centered and are weighted by a row
        of ``W``, the band need not contain the point estimate.

        References
        ----------
        Sims, Christopher A., and Tao Zha. 1999. "Error Bands for Impulse
        Response". Econometrica 67: 1113-1155.
        """
        model = self.model
        periods = self.periods
        irfs = self._choose_irfs(orth, svar)
        neqs = self.neqs
        irf_resim = model.irf_resim(
            orth=orth, repl=repl, steps=periods, rng=rng, burn=burn
        )

        W, eigva, k = self._eigval_decomp_SZ(irf_resim)

        if component is not None:
            if np.shape(component) != (neqs, neqs):
                raise ValueError(
                    "Component array must be " + str(neqs) + " x " + str(neqs)
                )
            k = _check_component(component, periods)

        gamma = np.zeros((repl, periods + 1, neqs, neqs))
        for p in range(repl):
            for i in range(neqs):
                for j in range(neqs):
                    gamma[p, 1:, i, j] = W[i, j, k[i, j], :] * irf_resim[p, 1:, i, j]

        gamma_sort = np.sort(gamma, axis=0)  # sort to get quantiles
        indx = round(signif / 2 * repl) - 1, round((1 - signif / 2) * repl) - 1

        lower = np.copy(irfs)
        upper = np.copy(irfs)
        for i in range(neqs):
            for j in range(neqs):
                lower[:, i, j] = irfs[:, i, j] + gamma_sort[indx[0], :, i, j]
                upper[:, i, j] = irfs[:, i, j] + gamma_sort[indx[1], :, i, j]

        return ErrorBand(lower, upper)

    @deprecate_kwarg("seed", "rng")
    def err_band_sz3(
        self,
        orth=False,
        svar=False,
        repl=1000,
        signif=0.05,
        rng=None,
        burn=100,
        component=None,
    ):
        """
        IRF Sims-Zha error band method 3. Does not assume symmetric error
        bands around mean.

        Parameters
        ----------
        orth : bool, optional
            Compute orthogonalized impulse responses. The default is False.
        svar : bool, optional
            Use structural IRFs. The default is False.
        repl : int, optional
            Number of MC replications. The default is 1000.
        signif : float, optional
            Significance level for the confidence interval, between 0 and
            1. The default is 0.05, giving a 95% confidence interval.
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
            Number of initial simulated obs to discard. The default is 100.
        component : array_like of int, optional
            Sequence of length neqs giving the index of the column of
            eigenvector/value to use for each error band. Note: the period
            of impulse (t=0) is not included when computing the principal
            component. If None, the column of the largest eigenvalue is
            used for each element.

            .. versionchanged:: 0.15.1

               Each value must be between 0 and neqs * periods - 1. Values
               of neqs * periods or more were not checked. Negative values
               select eigenvalues from the end; they are deprecated and
               issue a FutureWarning.

               `component` is used and no longer overwritten. It was
               ignored: the largest eigenvalue was always used, and the
               indices of the largest eigenvalues were written into the
               array passed.

        Returns
        -------
        ErrorBand
            A result object with fields ``lower`` and ``upper``.

        Notes
        -----
        For a shock in variable j, the simulated responses (from
        ``irf_resim``) of all variables in periods 1 to `periods` are stacked
        variable by variable into vectors of length ``neqs * periods``.
        ``V`` is their covariance matrix over the `repl` replications and the
        columns of ``W`` are its eigenvectors, in the unsorted order of
        :func:`scipy.linalg.eig`. With ``k`` the index of the largest
        eigenvalue (or the one that `component` selects for the shock), the
        ``k``-th row of ``W`` is split into ``neqs`` blocks of length
        `periods`, and block i multiplies each simulated response ``c`` of
        variable i elementwise,
        ``g = W[k, i * periods:(i + 1) * periods] * c``. In each period,
        ``lower`` and ``upper`` are the point estimate plus the
        ``round(signif / 2 * repl)``-th and the
        ``round((1 - signif / 2) * repl)``-th smallest of the `repl` values
        of ``g``; in period 0 both equal the point estimate. As for
        ``err_band_sz2``, the simulated responses are not centered and are
        weighted by a row of ``W``, not by an eigenvector as in eq. (9) of
        Sims and Zha (1999), so the band need not contain the point estimate.

        References
        ----------
        Sims, Christopher A., and Tao Zha. 1999. "Error Bands for Impulse
        Response". Econometrica 67: 1113-1155.
        """

        model = self.model
        periods = self.periods
        irfs = self._choose_irfs(orth, svar)
        neqs = self.neqs
        irf_resim = model.irf_resim(
            orth=orth, repl=repl, steps=periods, rng=rng, burn=burn
        )
        stack = np.zeros((neqs, repl, periods * neqs))

        # stack left to right, up and down

        for p in range(repl):
            for i in range(neqs):
                stack[i, p, :] = np.ravel(irf_resim[p, 1:, :, i].T)

        stack_cov = np.zeros((neqs, periods * neqs, periods * neqs))
        W = np.zeros((neqs, periods * neqs, periods * neqs))
        eigva = np.zeros((neqs, periods * neqs))
        k = np.zeros(neqs, dtype=int)

        if component is not None:
            if np.size(component) != (neqs):
                raise ValueError("Component array must be of length " + str(neqs))
            k = _check_component(component, neqs * periods)

        # compute for eigen decomp for each stack
        for i in range(neqs):
            stack_cov[i] = np.cov(stack[i], rowvar=0)
            W[i], eigva[i], k_max = util.eigval_decomp(stack_cov[i])
            if component is None:
                k[i] = k_max

        gamma = np.zeros((repl, periods + 1, neqs, neqs))
        for p in range(repl):
            for j in range(neqs):
                for i in range(neqs):
                    gamma[p, 1:, i, j] = (
                        W[j, k[j], i * periods : (i + 1) * periods]
                        * irf_resim[p, 1:, i, j]
                    )
                    if i == neqs - 1:
                        gamma[p, 1:, i, j] = (
                            W[j, k[j], i * periods :] * irf_resim[p, 1:, i, j]
                        )

        gamma_sort = np.sort(gamma, axis=0)  # sort to get quantiles
        indx = round(signif / 2 * repl) - 1, round((1 - signif / 2) * repl) - 1

        lower = np.copy(irfs)
        upper = np.copy(irfs)
        for i in range(neqs):
            for j in range(neqs):
                lower[:, i, j] = irfs[:, i, j] + gamma_sort[indx[0], :, i, j]
                upper[:, i, j] = irfs[:, i, j] + gamma_sort[indx[1], :, i, j]

        return ErrorBand(lower, upper)

    def _eigval_decomp_SZ(self, irf_resim):
        """
        Eigenvalue decomposition of the covariance matrix of the resimulated
        impulse responses

        Parameters
        ----------
        irf_resim : ndarray
            Monte Carlo resimulated impulse responses.

        Returns
        -------
        W : ndarray
            Array of eigenvectors.
        eigva : ndarray
            Array of eigenvalues.
        k : ndarray
            Matrix indicating the column number of the largest eigenvalue
            for each c_i,j.
        """
        neqs = self.neqs
        periods = self.periods

        cov_hold = np.zeros((neqs, neqs, periods, periods))
        for i in range(neqs):
            for j in range(neqs):
                cov_hold[i, j, :, :] = np.cov(irf_resim[:, 1:, i, j], rowvar=0)

        W = np.zeros((neqs, neqs, periods, periods))
        eigva = np.zeros((neqs, neqs, periods, 1))
        k = np.zeros((neqs, neqs), dtype=int)

        for i in range(neqs):
            for j in range(neqs):
                W[i, j, :, :], eigva[i, j, :, 0], k[i, j] = util.eigval_decomp(
                    cov_hold[i, j, :, :]
                )
        return W, eigva, k

    @cache_readonly
    def G(self):
        """
        Derivatives of the impulse responses with respect to the VAR
        coefficients

        A list of length ``periods`` whose element ``i - 1`` is the
        (neqs ** 2, neqs ** 2 * lags) matrix
        :math:`G_i = \\partial vec(\\Phi_i) / \\partial vec(A_1, \\dots, A_p)'`
        of Lütkepohl (2005, p. 111).
        """
        # Gi matrices as defined on p. 111

        K = self.neqs

        # nlags = self.model.p
        # J = np.hstack((np.eye(K),) + (np.zeros((K, K)),) * (nlags - 1))

        def _make_g(i):
            # p. 111 Lutkepohl
            G = 0.0
            for m in range(i):
                # be a bit cute to go faster
                idx = i - 1 - m
                if idx in self._g_memo:
                    apow = self._g_memo[idx]
                else:
                    apow = la.matrix_power(self._A.T, idx)
                    # apow = np.dot(J, apow)
                    apow = apow[:K]
                    self._g_memo[idx] = apow

                # take first K rows
                piece = np.kron(apow, self.irfs[m])
                G = G + piece

            return G

        return [_make_g(i) for i in range(1, self.periods + 1)]

    def _orth_cov(self):
        # Lutkepohl 3.7.8

        Ik = np.eye(self.neqs)
        PIk = np.kron(self.P.T, Ik)
        H = self.H

        covs = self._empty_covm(self.periods + 1)
        for i in range(self.periods + 1):
            if i == 0:
                apiece = 0
            else:
                Ci = np.dot(PIk, self.G[i - 1])
                apiece = Ci @ self.cov_a @ Ci.T

            Cibar = np.dot(np.kron(Ik, self.irfs[i]), H)
            bpiece = (Cibar @ self.cov_sig @ Cibar.T) / self.T

            # Lutkepohl typo, cov_sig correct
            covs[i] = apiece + bpiece

        return covs

    def cum_effect_cov(self, orth=False):
        """
        Compute asymptotic standard errors for cumulative impulse response
        coefficients

        Parameters
        ----------
        orth : bool, optional
            Compute orthogonalized impulse responses. The default is False.

        Returns
        -------
        ndarray
            Array of shape (periods + 1, neqs ** 2, neqs ** 2) containing
            the covariance matrix of the cumulative impulse response
            coefficients for each period.

        Notes
        -----
        eq. 3.7.7 (non-orth), 3.7.10 (orth)
        """
        Ik = np.eye(self.neqs)
        PIk = np.kron(self.P.T, Ik)

        F = 0.0
        covs = self._empty_covm(self.periods + 1)
        for i in range(self.periods + 1):
            if i > 0:
                F = F + self.G[i - 1]

            if orth:
                if i == 0:
                    apiece = 0
                else:
                    Bn = np.dot(PIk, F)
                    apiece = Bn @ self.cov_a @ Bn.T

                Bnbar = np.dot(np.kron(Ik, self.cum_effects[i]), self.H)
                bpiece = (Bnbar @ self.cov_sig @ Bnbar.T) / self.T

                covs[i] = apiece + bpiece
            else:
                if i == 0:
                    covs[i] = np.zeros((self.neqs**2, self.neqs**2))
                    continue

                covs[i] = F @ self.cov_a @ F.T

        return covs

    @deprecate_kwarg("seed", "rng")
    def cum_errband_mc(self, orth=False, repl=1000, signif=0.05, rng=None, burn=100):
        """
        IRF Monte Carlo integrated error bands of cumulative effect

        Parameters
        ----------
        orth : bool, optional
            Compute orthogonalized impulse responses. The default is False.
        repl : int, optional
            Number of MC replications. The default is 1000.
        signif : float, optional
            Significance level for the confidence interval, between 0 and
            1. The default is 0.05, giving a 95% confidence interval.
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
            Number of initial simulated obs to discard. The default is 100.

        Returns
        -------
        ErrorBand
            A result object with fields ``lower`` and ``upper``.
        """
        model = self.model
        periods = self.periods
        return model.irf_errband_mc(
            orth=orth,
            repl=repl,
            steps=periods,
            signif=signif,
            rng=rng,
            burn=burn,
            cum=True,
        )

    def lr_effect_cov(self, orth=False):
        """
        Compute asymptotic standard errors for long-run effects

        Parameters
        ----------
        orth : bool, optional
            Compute orthogonalized impulse responses. The default is False.

        Returns
        -------
        ndarray
            The (neqs ** 2, neqs ** 2) covariance matrix of the vectorized
            long-run effects.
        """
        lre = self.lr_effects
        Finfty = np.kron(np.tile(lre.T, self.lags), lre)
        Ik = np.eye(self.neqs)

        if orth:
            Binf = np.dot(np.kron(self.P.T, np.eye(self.neqs)), Finfty)
            Binfbar = np.dot(np.kron(Ik, lre), self.H)

            return Binf @ self.cov_a @ Binf.T + Binfbar @ self.cov_sig @ Binfbar.T
        else:
            return Finfty @ self.cov_a @ Finfty.T

    def stderr(self, orth=False):
        """
        Asymptotic standard errors of the impulse responses

        Parameters
        ----------
        orth : bool, optional
            Compute orthogonalized impulse responses. The default is False.

        Returns
        -------
        ndarray
            Array of shape (periods + 1, neqs, neqs). Element ``[h, i, j]``
            is the standard error of the response of variable ``i`` at
            horizon ``h`` to a shock in variable ``j``.
        """
        return np.array([tsa.unvec(np.sqrt(np.diag(c))) for c in self.cov(orth=orth)])

    def cum_effect_stderr(self, orth=False):
        """
        Asymptotic standard errors of the cumulative impulse responses

        Parameters
        ----------
        orth : bool, optional
            Compute orthogonalized impulse responses. The default is False.

        Returns
        -------
        ndarray
            Array of shape (periods + 1, neqs, neqs), arranged as
            ``cum_effects``.
        """
        return np.array(
            [tsa.unvec(np.sqrt(np.diag(c))) for c in self.cum_effect_cov(orth=orth)]
        )

    def lr_effect_stderr(self, orth=False):
        """
        Asymptotic standard errors of the long-run effects

        Parameters
        ----------
        orth : bool, optional
            Compute orthogonalized impulse responses. The default is False.

        Returns
        -------
        ndarray
            Array of shape (neqs, neqs), arranged as ``lr_effects``.
        """
        cov = self.lr_effect_cov(orth=orth)
        return tsa.unvec(np.sqrt(np.diag(cov)))

    def _empty_covm(self, periods):
        return np.zeros((periods, self.neqs**2, self.neqs**2), dtype=float)

    @cache_readonly
    def H(self):
        """
        Derivative of the orthogonalization matrix with respect to sigma_u

        The (neqs ** 2, neqs * (neqs + 1) / 2) matrix
        :math:`H = \\partial vec(P) / \\partial vech(\\Sigma_u)'` of
        Lütkepohl (2005, Section 3.7), where ``P`` is the Cholesky factor of
        sigma_u.
        """
        k = self.neqs
        Lk = tsa.elimination_matrix(k)
        Kkk = tsa.commutation_matrix(k, k)
        Ik = np.eye(k)

        # B = Lk @ (np.eye(k**2) + commutation_matrix(k, k)) @ \
        #     np.kron(self.P, np.eye(k)) @ Lk.T
        # return Lk.T @ L.inv(B)

        B = Lk @ (np.kron(Ik, self.P) @ Kkk + np.kron(self.P, Ik)) @ Lk.T

        return np.dot(Lk.T, L.inv(B))

    def fevd_table(self):
        """Not implemented."""
        raise NotImplementedError
