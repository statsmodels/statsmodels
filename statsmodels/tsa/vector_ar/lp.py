"""
Local Projections (LP) estimator for impulse response functions.

Reference
---------
Jordà, Ò. (2005). Estimation and Inference of Impulse Responses by Local
Projections. *American Economic Review*, 95(1), 161-182.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.validation import array_like


__all__ = ["LocalProjections"]


def _hac_bandwidth(h, nw_lags=None):
    """Return Newey-West bandwidth for an h-step-ahead LP regression."""
    if nw_lags is not None:
        return nw_lags
    # At horizon h the LP residual is an MA(h) by construction; use at
    # least h lags.  Fall back to a small positive value at h=0.
    return max(h, int(np.ceil(4.0 * (max(h, 1) / 100.0) ** (2.0 / 9.0))))


def _nw_cov(X, resid, nlags):
    """Newey-West HAC sandwich covariance (X'X)^{-1} S (X'X)^{-1} / T.

    Parameters
    ----------
    X : ndarray, shape (T, k)
    resid : ndarray, shape (T,)
    nlags : int

    Returns
    -------
    V : ndarray, shape (k, k)
    """
    T, k = X.shape
    scores = X * resid[:, None]           # (T, k)  score contributions
    S = scores.T @ scores / T             # gamma_0
    for lag in range(1, nlags + 1):
        w = 1.0 - lag / (nlags + 1.0)    # Bartlett kernel
        gamma = scores[lag:].T @ scores[:-lag] / T
        S += w * (gamma + gamma.T)
    XtX_inv = np.linalg.inv(X.T @ X / T)
    return XtX_inv @ S @ XtX_inv / T


class LocalProjectionsResults:
    """Results from a fitted :class:`LocalProjections` model.

    Attributes
    ----------
    irfs : ndarray, shape (H+1, n_endog, n_shock)
        Estimated impulse responses.  ``irfs[h, i, j]`` is the response of
        variable ``i`` to shock ``j`` at horizon ``h``.
    stderr : ndarray, shape (H+1, n_endog, n_shock)
        Newey-West standard errors for each IRF coefficient.
    model : LocalProjections
        The model instance.
    nobs : int
        Number of observations used (after trimming for lags and max horizon).
    """

    def __init__(self, model, irfs, stderr):
        self.model = model
        self.irfs = irfs
        self.stderr = stderr
        self.nobs = model._nobs

    @property
    def horizons(self):
        return self.model.horizons

    def conf_int(self, alpha=0.05):
        """Pointwise confidence intervals for all impulse responses.

        Parameters
        ----------
        alpha : float
            Significance level (default 0.05 gives 95 % intervals).

        Returns
        -------
        ci : ndarray, shape (H+1, n_endog, n_shock, 2)
            ``ci[h, i, j, 0]`` = lower bound,
            ``ci[h, i, j, 1]`` = upper bound.
        """
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        lower = self.irfs - z * self.stderr
        upper = self.irfs + z * self.stderr
        return np.stack([lower, upper], axis=-1)

    def cumulative_effects(self):
        """Cumulative sum of IRFs across horizons.

        Returns
        -------
        ndarray, shape (H+1, n_endog, n_shock)
        """
        return np.cumsum(self.irfs, axis=0)

    def plot_irfs(self, impulse=None, response=None, alpha=0.1,
                  figsize=None, n_cols=None):
        """Plot impulse response functions with confidence bands.

        Parameters
        ----------
        impulse : int or str or None
            Index or name of the shock variable.  Plots all shocks if None.
        response : int or str or None
            Index or name of the response variable.  Plots all responses
            if None.
        alpha : float
            Significance level for confidence bands.
        figsize : tuple or None
        n_cols : int or None

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        var_names = self.model.endog_names
        shock_names = self.model.shock_names
        H = self.horizons
        ci = self.conf_int(alpha=alpha)
        x = np.arange(H + 1)

        imp_idx = (list(range(len(shock_names)))
                   if impulse is None
                   else [shock_names.index(impulse)
                         if isinstance(impulse, str) else impulse])
        resp_idx = (list(range(len(var_names)))
                    if response is None
                    else [var_names.index(response)
                          if isinstance(response, str) else response])

        n_plots = len(imp_idx) * len(resp_idx)
        n_cols = n_cols or min(3, n_plots)
        n_rows = int(np.ceil(n_plots / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=figsize or (4 * n_cols, 3 * n_rows),
            squeeze=False,
        )

        plot_num = 0
        for j in imp_idx:
            for i in resp_idx:
                ax = axes[plot_num // n_cols, plot_num % n_cols]
                ax.plot(x, self.irfs[:, i, j], color="steelblue", lw=1.5)
                ax.fill_between(
                    x, ci[:, i, j, 0], ci[:, i, j, 1],
                    alpha=0.25, color="steelblue",
                )
                ax.axhline(0, color="black", lw=0.8, ls="--")
                ax.set_title(f"{shock_names[j]} → {var_names[i]}")
                ax.set_xlabel("Horizon")
                plot_num += 1

        for k in range(plot_num, n_rows * n_cols):
            axes[k // n_cols, k % n_cols].set_visible(False)

        fig.tight_layout()
        return fig

    def __repr__(self):
        return (
            f"LocalProjectionsResults("
            f"n_endog={self.irfs.shape[1]}, "
            f"n_shock={self.irfs.shape[2]}, "
            f"horizons={self.horizons}, "
            f"nobs={self.nobs})"
        )


class LocalProjections:
    r"""Local Projections estimator for impulse response functions.

    For each horizon :math:`h = 0, 1, \ldots, H`, the impulse response is
    estimated by running a separate OLS regression

    .. math::

        y_{t+h} = \alpha_h
                + \boldsymbol{\beta}_h \mathbf{z}_t
                + \boldsymbol{\Gamma}_h \mathbf{x}_t
                + \varepsilon_{t+h},

    where :math:`\mathbf{z}_t` are the shock variables and
    :math:`\mathbf{x}_t` contains lagged controls.  Standard errors are
    Newey-West HAC-corrected with a bandwidth of at least :math:`h` lags to
    account for the MA(:math:`h`) serial correlation in the residuals.

    Parameters
    ----------
    endog : array_like, shape (T, n) or (T,)
        Endogenous variables.  A 1-D input is treated as a single variable.
    shock_idx : int or list of int, optional
        Column index/indices within *endog* whose contemporaneous value
        enters as the shock.  Defaults to ``[0]`` (the first variable).
    lags : int, optional
        Number of lagged values of *all* endog variables to include as
        controls.  Defaults to ``1``.
    horizons : int, optional
        Maximum IRF horizon :math:`H`.  Regressions are run for
        :math:`h = 0, 1, \ldots, H`.  Defaults to ``12``.
    exog : array_like, shape (T, k) or None, optional
        Additional exogenous controls (e.g. time dummies, external
        instruments).  Do **not** include a constant; it is added
        automatically.
    trend : {"n", "c", "ct"}, optional
        Deterministic trend: ``"n"`` none, ``"c"`` constant (default),
        ``"ct"`` constant plus linear trend.
    nw_lags : int or None, optional
        Override the automatic Newey-West bandwidth.  When *None* (default)
        the bandwidth at horizon :math:`h` is
        ``max(h, ceil(4*(max(h,1)/100)^{2/9}))``.

    References
    ----------
    Jordà, Ò. (2005). Estimation and Inference of Impulse Responses by Local
    Projections. *American Economic Review*, 95(1), 161–182.

    Examples
    --------
    >>> import numpy as np
    >>> from statsmodels.tsa.vector_ar.lp import LocalProjections
    >>> rng = np.random.default_rng(0)
    >>> T, n = 200, 2
    >>> e = rng.standard_normal((T, n))
    >>> y = np.cumsum(e, axis=0)
    >>> lp = LocalProjections(y, shock_idx=0, lags=2, horizons=8)
    >>> res = lp.fit()
    >>> res.irfs.shape
    (9, 2, 1)
    """

    def __init__(
        self,
        endog,
        shock_idx=None,
        lags=1,
        horizons=12,
        exog=None,
        trend="c",
        nw_lags=None,
    ):
        # Capture column names before array_like strips them.
        _endog_col_names = (list(endog.columns)
                            if isinstance(endog, pd.DataFrame) else None)

        endog = array_like(endog, "endog", ndim=None)
        if endog.ndim == 1:
            endog = endog[:, None]
        if endog.ndim != 2:
            raise ValueError("endog must be 1-D or 2-D.")

        self.endog = np.asarray(endog, dtype=float)
        T, n = self.endog.shape

        if shock_idx is None:
            shock_idx = [0]
        elif isinstance(shock_idx, (int, np.integer)):
            shock_idx = [int(shock_idx)]
        else:
            shock_idx = list(shock_idx)
        for idx in shock_idx:
            if not (0 <= idx < n):
                raise ValueError(
                    f"shock_idx={idx} out of range for endog with {n} columns."
                )
        self.shock_idx = shock_idx

        if lags < 0:
            raise ValueError("lags must be non-negative.")
        if horizons < 0:
            raise ValueError("horizons must be non-negative.")
        if trend not in ("n", "c", "ct"):
            raise ValueError("trend must be one of 'n', 'c', 'ct'.")

        self.lags = int(lags)
        self.horizons = int(horizons)
        self.trend = trend
        self.nw_lags = nw_lags

        if exog is not None:
            exog = np.asarray(exog, dtype=float)
            if exog.ndim == 1:
                exog = exog[:, None]
            if exog.shape[0] != T:
                raise ValueError(
                    "exog must have the same number of rows as endog."
                )
        self.exog = exog

        # Variable names for output / plotting.
        if _endog_col_names is not None:
            self.endog_names = _endog_col_names
        else:
            self.endog_names = [f"y{i}" for i in range(n)]
        self.shock_names = [self.endog_names[j] for j in shock_idx]

        # Usable obs: first `lags` rows consumed by lag construction;
        # last `horizons` rows have no h-step-ahead LHS.
        self._nobs = T - lags - horizons

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_regressors(self, t_start, t_end):
        """Construct the RHS design matrix for t = t_start ... t_end-1.

        Column order
        ------------
        [shock_0, ..., shock_s | lag1_y0, ..., lag1_yn, lag2_y0, ... |
         exog | trend terms]

        Returns
        -------
        X : ndarray, shape (n_obs, k)
        shock_cols : list of int
            Column indices of the shock variables.
        """
        endog = self.endog
        parts = []

        # Contemporaneous shocks (columns come first so their indices are
        # always 0, 1, ..., n_shock-1).
        parts.append(endog[t_start:t_end, self.shock_idx])
        shock_cols = list(range(len(self.shock_idx)))

        # Lagged controls.
        for lag in range(1, self.lags + 1):
            parts.append(endog[t_start - lag : t_end - lag, :])

        # Exogenous.
        if self.exog is not None:
            parts.append(self.exog[t_start:t_end])

        # Deterministic terms.
        n_obs = t_end - t_start
        if self.trend in ("c", "ct"):
            parts.append(np.ones((n_obs, 1)))
        if self.trend == "ct":
            parts.append(np.arange(t_start, t_end, dtype=float)[:, None])

        return np.concatenate(parts, axis=1), shock_cols

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self):
        """Estimate LP-IRFs by running H+1 OLS regressions.

        Returns
        -------
        LocalProjectionsResults
        """
        H = self.horizons
        lags = self.lags
        endog = self.endog
        T, n = endog.shape
        n_shock = len(self.shock_idx)

        t_start = lags
        t_end = T - H          # exclusive upper bound

        if t_end <= t_start:
            raise ValueError(
                f"Not enough observations: need T > lags + horizons = "
                f"{lags + H}, got T = {T}."
            )

        X, shock_cols = self._build_regressors(t_start, t_end)

        irfs = np.zeros((H + 1, n, n_shock))
        stderr = np.zeros((H + 1, n, n_shock))

        for h in range(H + 1):
            Y_h = endog[t_start + h : t_end + h, :]   # (n_obs, n)
            nlags = _hac_bandwidth(h, self.nw_lags)

            for i in range(n):
                res = OLS(Y_h[:, i], X).fit()
                V = _nw_cov(X, res.resid, nlags)
                for s_pos, s_col in enumerate(shock_cols):
                    irfs[h, i, s_pos] = res.params[s_col]
                    stderr[h, i, s_pos] = np.sqrt(max(V[s_col, s_col], 0.0))

        return LocalProjectionsResults(self, irfs, stderr)
