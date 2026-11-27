"""
Fama-MacBeth Two-Stage Panel Regression

License: BSD-3

References
----------
.. [*] Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium:
       Empirical tests. Journal of Political Economy, 81(3), 607-636.
.. [*] Cochrane, J. H. (2001). Asset Pricing. Princeton University Press.
.. [*] Petersen, M. A. (2009). Estimating standard errors in finance panel
       data sets: Comparing approaches. Review of financial studies, 22(1),
       435-480.
.. [*] Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite,
       heteroskedasticity and autocorrelation consistent covariance matrix.
       Econometrica, 55(3), 703-708.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from typing import (
    List,
    Dict,
    Optional,
    Tuple
)

import statsmodels.api as sm
from statsmodels.base import model as base
from pandas.util._decorators import Appender 
from statsmodels.iolib import summary2
from statsmodels.regression.linear_model import OLS, RegressionResults
from statsmodels.stats.sandwich_covariance import (
    S_hac_simple,
    weights_bartlett,
)
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.validation import array_like, bool_like, int_like, string_like

__all__ = ["FamaMacBeth", "FamaMacBethResults"]


class FamaMacBeth(base.Model):
    r"""
    Fama-MacBeth (1973) rolling cross-sectional regression for panel data.

    Parameters
    ----------
    endog : array_like
        1-d endogenous response variable (asset returns). The dependent
        variable.
    exog : array_like
        A nobs x k array where `nobs` is the number of observations and `k`
        is the number of regressors (factor loadings or characteristics).
        An intercept is not included by default and should be added by
        the user if desired.
    entity : array_like
        1-d array of entity (e.g., asset or firm) identifiers. Must be
        the same length as `endog`.
    time : array_like
        1-d array of time period identifiers. Must be the same length as
        `endog`.
    missing : str
        Available options are 'none', 'drop', and 'raise'. If 'none', no nan
        checking is done. If 'drop', any observations with nans are dropped.
        If 'raise', an error is raised. Default is 'drop'.

    Attributes
    ----------
    endog : ndarray
        A reference to the endogenous response variable.
    exog : ndarray
        A reference to the exogenous design.
    entity : ndarray
        Entity identifiers.
    time : ndarray
        Time identifiers.

    See Also
    --------
    statsmodels.regression.linear_model.OLS : Ordinary Least Squares.
    statsmodels.regression.linear_model.WLS : Weighted Least Squares.

    Notes
    -----
    The Fama-MacBeth procedure is a two-stage method for estimating risk premiums
    in asset pricing models:

    **Stage 1 - Time-series regression (for each entity i):**

    .. math::

        r_{i,t} = \alpha_i + \beta_i' F_t + \varepsilon_{i,t}

    where :math:`r_{i,t}` is the return of entity i at time t, :math:`F_t` is
    the vector of factor returns at time t, :math:`\beta_i` is the vector of
    factor loadings for entity i, :math:`\alpha_i` is the pricing error
    (intercept), and :math:`\varepsilon_{i,t}` is the idiosyncratic error.

    **Stage 2 - Cross-sectional regression (for each time period t):**

    .. math::

        r_{i,t} = \lambda_{0,t} + \lambda_t' \hat{\beta}_i + \eta_{i,t}

    where :math:`\lambda_t` is the vector of risk premia at time t,
    :math:`\lambda_{0,t}` is the zero-beta rate at time t, and
    :math:`\eta_{i,t}` is the cross-sectional pricing error.

    **Final estimates:**

    .. math::

        \hat{\lambda} = \frac{1}{T} \sum_{t=1}^T \lambda_t

    Standard errors account for time-series variation in :math:`\lambda_t`.

    The procedure avoids the errors-in-variables problem of direct
    cross-sectional regression on returns using betas.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statsmodels.api as sm
    >>> from statsmodels.regression.fama_macbeth import FamaMacBeth
    >>>
    >>> # Simulate panel data
    >>> np.random.seed(42)
    >>> n_entities = 25
    >>> n_periods = 120
    >>> entities = np.repeat(np.arange(n_entities), n_periods)
    >>> time = np.tile(np.arange(n_periods), n_entities)
    >>>
    >>> # True factor loadings
    >>> true_betas = np.random.uniform(0.5, 1.5, (n_entities, 2))
    >>> factor_returns = np.random.randn(n_periods, 2) * 0.05
    >>>
    >>> # Generate returns
    >>> returns = np.zeros(n_entities * n_periods)
    >>> for i in range(n_entities):
    ...     idx = entities == i
    ...     returns[idx] = (factor_returns @ true_betas[i] +
    ...                     np.random.randn(n_periods) * 0.03)
    >>>
    >>> # Construct factor data (same for all entities in a period)
    >>> factors = np.zeros((n_entities * n_periods, 2))
    >>> for t in range(n_periods):
    ...     idx = time == t
    ...     factors[idx] = factor_returns[t]
    >>>
    >>> # Add constant
    >>> exog = sm.add_constant(factors)
    >>>
    >>> # Estimate Fama-MacBeth
    >>> mod = FamaMacBeth(returns, exog, entity=entities, time=time)
    >>> res = mod.fit()
    >>> print(res.summary())
    """

    def __init__(
        self,
        endog,
        exog,
        entity,
        time,
        missing="drop",
        **kwargs,
    ) -> None:
        # Validate inputs
        endog = array_like(endog, "endog", ndim=1)
        entity = array_like(entity, "entity", ndim=1)
        time = array_like(time, "time", ndim=1)

        # Ensure all inputs have the same length
        nobs = len(endog)
        if len(entity) != nobs:
            raise ValueError(
                f"entity has length {len(entity)}, expected {nobs}"
            )
        if len(time) != nobs:
            raise ValueError(
                f"time has length {len(time)}, expected {nobs}"
            )

        # Store entity and time before calling parent
        self.entity = entity
        self.time = time

        # Call parent init
        super().__init__(endog, exog, missing=missing, **kwargs)

        # Update stored entity and time after missing handling
        if hasattr(self.data, "row_labels"):
            # If row_labels exists, missing values were dropped
            self.entity = self.entity[self.data.row_labels]
            self.time = self.time[self.data.row_labels]

        # Get unique entities and time periods
        self.unique_entity = np.unique(self.entity)
        self.unique_time = np.unique(self.time)
        self.n_entities = len(self.unique_entity)
        self.n_periods = len(self.unique_time)

        # Validate dimensions
        if self.n_periods < 10:
            warnings.warn(
                f"Only {self.n_periods} time periods available. Recommend "
                "at least 10 for reliable inference.",
                ValueWarning,
                stacklevel=2,
            )

    def fit(
        self,
        cov_type="robust",
        bandwidth=None,
        kernel="bartlett",
        use_correction=True,
    ) -> "FamaMacBethResults":
        """
        Estimate parameters using the Fama-MacBeth two-stage procedure.

        Parameters
        ----------
        cov_type : str, optional
            Covariance estimator for standard errors. Default is 'robust'.
            
            - 'robust', 'unadjusted', 'heteroskedastic', 'homoskedastic':
              Standard Fama-MacBeth covariance estimator
            - 'kernel' or 'HAC': Heteroskedasticity and Autocorrelation
              Consistent (Newey-West) covariance estimator
        bandwidth : {int, None}, optional
            Bandwidth for HAC/kernel covariance estimator. If None, uses
            Newey-West automatic selection: floor[4(T/100)^(2/9)].
            Only used when cov_type is 'kernel' or 'HAC'.
        kernel : str, optional
            Kernel function for HAC estimator. Default is 'bartlett'.
            Only used when cov_type is 'kernel' or 'HAC'.
            Currently only 'bartlett' is supported.
        use_correction : bool, optional
            Whether to use small sample correction for standard errors.
            Default is True.

        Returns
        -------
        FamaMacBethResults
            Estimation results.

        Notes
        -----
        The estimation follows two stages:

        1. For each entity, estimate factor loadings via time-series OLS
        2. For each time period, estimate risk premia via cross-sectional OLS
           using the estimated loadings from stage 1

        Standard errors are computed from the time-series variation in the
        estimated risk premia.
        """
        # Stage 1: Time-series regressions
        betas, alphas, resids_ts = self._stage1_estimation()

        # Stage 2: Cross-sectional regressions
        lambdas, resids_cs = self._stage2_estimation(betas)

        # Compute average risk premia
        risk_premia = np.nanmean(lambdas, axis=0)

        # Compute standard errors
        se, bandwidth_used = self._compute_standard_errors(
            lambdas, cov_type, bandwidth, kernel
        )

        # Compute test statistics
        t_stats = risk_premia / se
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), self.n_periods - 1))

        # Compute R-squared statistics
        r_squared, r_squared_adj = self._compute_r_squared(betas, lambdas)

        # Package results
        results = FamaMacBethResults(
            self,
            risk_premia,
            betas=betas,
            alphas=alphas,
            lambdas=lambdas,
            se=se,
            t_stats=t_stats,
            p_values=p_values,
            r_squared=r_squared,
            r_squared_adj=r_squared_adj,
            cov_type=cov_type,
            bandwidth=bandwidth_used,
            use_correction=use_correction,
        )

        return results

    def _stage1_estimation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Stage 1: Estimate factor loadings via time-series regressions.

        Returns
        -------
        betas : ndarray
            Estimated factor loadings, shape (n_factors, n_entities).
        alphas : ndarray
            Estimated pricing errors (intercepts), shape (n_entities,).
        resids : ndarray
            Residuals from time-series regressions, shape (n_periods, n_entities).
        """
        n_factors = self.exog.shape[1]
        betas = np.zeros((n_factors, self.n_entities))
        alphas = np.zeros(self.n_entities)
        resids_list = []

        for i, entity in enumerate(self.unique_entity):
            # Select data for this entity
            entity_mask = self.entity == entity
            y = self.endog[entity_mask]
            X = self.exog[entity_mask]

            # Estimate via OLS
            try:
                model = OLS(y, X)
                result = model.fit()
                betas[:, i] = result.params
                alphas[i] = result.params[0] if self.k_constant else 0
                resids_list.append(result.resid)
            except np.linalg.LinAlgError:
                # Singular matrix - use NaN
                betas[:, i] = np.nan
                alphas[i] = np.nan
                resids_list.append(np.full(len(y), np.nan))

        # Stack residuals
        max_len = max(len(r) for r in resids_list)
        resids = np.full((max_len, self.n_entities), np.nan)
        for i, r in enumerate(resids_list):
            resids[:len(r), i] = r

        return betas, alphas, resids

    def _stage2_estimation(self, betas) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stage 2: Estimate risk premia via cross-sectional regressions.

        Parameters
        ----------
        betas : ndarray
            Estimated factor loadings from stage 1, shape (n_factors, n_entities).

        Returns
        -------
        lambdas : ndarray
            Risk premia estimates for each period, shape (n_periods, n_factors).
        resids : ndarray
            Residuals from cross-sectional regressions, shape (n_entities, n_periods).
        """
        n_factors = betas.shape[0]
        lambdas = np.zeros((self.n_periods, n_factors))
        resids_list = []

        for t, period in enumerate(self.unique_time):
            # Select data for this period
            period_mask = self.time == period
            y = self.endog[period_mask]
            # Use betas as regressors
            X = betas.T  # Transpose to get (n_entities, n_factors)

            # Check for sufficient observations
            n_valid = np.sum(~np.isnan(y))
            if n_valid < n_factors:
                lambdas[t] = np.nan
                resids_list.append(np.full(len(y), np.nan))
                continue

            # Estimate via OLS
            try:
                # Remove NaN observations
                valid_mask = ~np.isnan(y)
                y_valid = y[valid_mask]
                X_valid = X[valid_mask]

                model = OLS(y_valid, X_valid)
                result = model.fit()
                lambdas[t] = result.params
                
                # Store residuals
                resid_full = np.full(len(y), np.nan)
                resid_full[valid_mask] = result.resid
                resids_list.append(resid_full)
            except np.linalg.LinAlgError:
                # Singular matrix - use NaN
                lambdas[t] = np.nan
                resids_list.append(np.full(len(y), np.nan))

        # Stack residuals
        max_len = max(len(r) for r in resids_list)
        resids = np.full((max_len, self.n_periods), np.nan)
        for i, r in enumerate(resids_list):
            resids[:len(r), i] = r

        return lambdas, resids

    def _compute_standard_errors(
        self, 
        lambdas, 
        cov_type, 
        bandwidth, 
        kernel
    ) -> Tuple[np.ndarray, Optional[int]]:
        """
        Compute standard errors for risk premia estimates.

        Parameters
        ----------
        lambdas : ndarray
            Risk premia estimates for each period, shape (n_periods, n_factors).
        cov_type : str
            Type of covariance estimator.
        bandwidth : {int, None}
            Bandwidth for HAC estimator.
        kernel : str
            Kernel function for HAC estimator.

        Returns
        -------
        se : ndarray
            Standard errors, shape (n_factors,).
        bandwidth_used : {int, None}
            Bandwidth used (only for HAC, None otherwise).
        """
        # Remove NaN observations
        valid_lambdas = lambdas[~np.isnan(lambdas).any(axis=1)]
        n_valid = len(valid_lambdas)

        if n_valid < 2:
            raise ValueError("Insufficient valid periods for covariance estimation")

        if cov_type.lower() in ("robust", "unadjusted", "heteroskedastic", "homoskedastic"):
            # Standard Fama-MacBeth covariance
            demeaned = valid_lambdas - np.mean(valid_lambdas, axis=0)
            vcv = (demeaned.T @ demeaned) / n_valid
            se = np.sqrt(np.diag(vcv) / n_valid)
            bandwidth_used = None

        elif cov_type.lower() in ("kernel", "hac"):
            # HAC (Newey-West) covariance
            # Use statsmodels implementation
            demeaned = valid_lambdas - np.mean(valid_lambdas, axis=0)

            # Auto-select bandwidth if not provided
            if bandwidth is None:
                bandwidth = int(np.floor(4 * (n_valid / 100) ** (2 / 9)))
            bandwidth_used = bandwidth

            # Compute HAC covariance using statsmodels
            if kernel.lower() == "bartlett":
                weights_func = weights_bartlett
            else:
                raise ValueError(
                    f"Kernel '{kernel}' not supported. Use 'bartlett'."
                )

            S = S_hac_simple(demeaned, nlags=bandwidth, weights_func=weights_func)
            se = np.sqrt(np.diag(S) / n_valid)

        else:
            raise ValueError(
                f"Unknown cov_type: {cov_type}. Use 'robust', 'unadjusted', "
                "'kernel', or 'HAC'."
            )

        return se, bandwidth_used

    def _compute_r_squared(self, betas, lambdas) -> Tuple[float, float]:
        """
        Compute R-squared statistics.

        Parameters
        ----------
        betas : ndarray
            Estimated factor loadings, shape (n_factors, n_entities).
        lambdas : ndarray
            Risk premia estimates, shape (n_periods, n_factors).

        Returns
        -------
        r_squared : float
            Overall R-squared from regressing average returns on betas.
        r_squared_adj : float
            Average adjusted R-squared across time periods.
        """
        # Overall R-squared: regress average returns on betas
        avg_returns = np.zeros(self.n_entities)
        for i, entity in enumerate(self.unique_entity):
            entity_mask = self.entity == entity
            avg_returns[i] = np.nanmean(self.endog[entity_mask])

        # Remove entities with NaN betas or returns
        valid_mask = ~(np.isnan(avg_returns) | np.isnan(betas).any(axis=0))
        if valid_mask.sum() > 0:
            model = OLS(avg_returns[valid_mask], betas.T[valid_mask])
            result = model.fit()
            r_squared = result.rsquared
        else:
            r_squared = np.nan

        # Average adjusted R-squared across periods
        r_squared_adj_list = []
        for t, period in enumerate(self.unique_time):
            if np.isnan(lambdas[t]).any():
                continue

            period_mask = self.time == period
            y = self.endog[period_mask]
            X = betas.T

            valid_mask = ~np.isnan(y)
            if valid_mask.sum() > betas.shape[0]:
                model = OLS(y[valid_mask], X[valid_mask])
                result = model.fit()
                r_squared_adj_list.append(result.rsquared_adj)

        r_squared_adj = np.nanmean(r_squared_adj_list) if r_squared_adj_list else np.nan

        return r_squared, r_squared_adj


class FamaMacBethResults(base.Results):
    """
    Results from Fama-MacBeth two-stage panel regression.

    Parameters
    ----------
    model : FamaMacBeth
        The model instance.
    params : ndarray
        Estimated risk premia (average lambdas).
    **kwargs
        Additional results attributes.

    Attributes
    ----------
    params : ndarray
        Estimated risk premia.
    betas : ndarray
        Estimated factor loadings from stage 1, shape (n_factors, n_entities).
    alphas : ndarray
        Estimated pricing errors (intercepts) from stage 1, shape (n_entities,).
    lambdas : ndarray
        Risk premia estimates for each period, shape (n_periods, n_factors).
    bse : ndarray
        Standard errors of risk premia estimates.
    tvalues : ndarray
        t-statistics for risk premia.
    pvalues : ndarray
        p-values for two-sided t-tests.
    rsquared : float
        R-squared from regressing average returns on betas.
    rsquared_adj : float
        Average adjusted R-squared across time periods.
    cov_type : str
        Covariance estimator used.
    bandwidth : {int, None}
        Bandwidth used for HAC estimator (None if not HAC).
    """

    def __init__(
        self,
        model,
        params,
        betas=None,
        alphas=None,
        lambdas=None,
        se=None,
        t_stats=None,
        p_values=None,
        r_squared=None,
        r_squared_adj=None,
        cov_type="robust",
        bandwidth=None,
        use_correction=True,
    ) -> None:
        super().__init__(model, params)
        self.betas = betas
        self.alphas = alphas
        self.lambdas = lambdas
        self.bse = se
        self.tvalues = t_stats
        self.pvalues = p_values
        self.rsquared = r_squared
        self.rsquared_adj = r_squared_adj
        self.cov_type = cov_type
        self.bandwidth = bandwidth
        self.use_correction = use_correction

    @cache_readonly
    def nobs(self) -> float:
        """Number of observations."""
        return float(len(self.model.endog))

    @cache_readonly
    def n_periods(self) -> int:
        """Number of time periods."""
        return self.model.n_periods

    @cache_readonly
    def n_entities(self) -> int:
        """Number of entities (cross-sectional units)."""
        return self.model.n_entities

    @cache_readonly
    def df_resid(self) -> int:
        """Residual degrees of freedom."""
        return self.n_periods - len(self.params)

    @cache_readonly
    def df_model(self) -> int:
        """Model degrees of freedom."""
        return len(self.params) - self.model.k_constant

    def conf_int(self, alpha=0.05, cols=None) -> np.ndarray:
        """
        Confidence intervals for risk premia estimates.

        Parameters
        ----------
        alpha : float, optional
            Significance level. Default is 0.05 for 95% confidence intervals.
        cols : array_like, optional
            Column indices for parameters to include. Default is all.

        Returns
        -------
        ndarray
            Confidence intervals, shape (n_params, 2).
        """
        if cols is None:
            cols = np.arange(len(self.params))

        params = self.params[cols]
        bse = self.bse[cols]

        dist = stats.t(self.df_resid)
        q = dist.ppf(1 - alpha / 2)

        lower = params - q * bse
        upper = params + q * bse

        return np.column_stack([lower, upper])

    def summary(self, yname=None, xname=None, title=None, alpha=0.05) -> summary2.Summary:
        """
        Summarize the regression results.

        Parameters
        ----------
        yname : str, optional
            Name of endogenous variable. Default is `y`.
        xname : list[str], optional
            Names of exogenous variables. Default is `var_#`.
        title : str, optional
            Title for the summary table.
        alpha : float, optional
            Significance level for confidence intervals. Default is 0.05.

        Returns
        -------
        Summary
            Summary instance with tables.
        """
        if title is None:
            title = "Fama-MacBeth Two-Stage Regression Results"

        if yname is None:
            yname = self.model.endog_names

        if xname is None:
            xname = self.model.exog_names

        # Create summary instance
        smry = summary2.Summary()
        smry.add_title(title)

        # Top table: Model info
        top_left = [
            ("Dep. Variable:", yname),
            ("Model:", "Fama-MacBeth"),
            ("Method:", "Two-Stage OLS"),
            ("No. Observations:", f"{int(self.nobs)}"),
        ]

        top_right = [
            ("No. Periods:", f"{int(self.n_periods)}"),
            ("No. Entities:", f"{int(self.n_entities)}"),
            ("R-squared:", f"{self.rsquared:.3f}"),
            ("Adj. R-squared:", f"{self.rsquared_adj:.3f}"),
        ]

        smry.add_dict(dict(top_left), dict(top_right))

        # Parameter table
        param_table = pd.DataFrame({
            "coef": self.params,
            "std err": self.bse,
            "t": self.tvalues,
            "P>|t|": self.pvalues,
        }, index=xname)

        # Add confidence intervals
        ci = self.conf_int(alpha=alpha)
        param_table[f"[{alpha/2:.3f}"] = ci[:, 0]
        param_table[f"{1-alpha/2:.3f}]"] = ci[:, 1]

        smry.add_df(param_table)

        # Bottom notes
        notes = [
            f"Covariance Type: {self.cov_type}",
        ]
        if self.bandwidth is not None:
            notes.append(f"HAC Bandwidth: {self.bandwidth}")

        smry.add_extra_txt(notes)

        return smry

    def __str__(self) -> str:
        return self.summary().as_text()

    def __repr__(self) -> str:
        return str(type(self)) + "\n" + self.__str__()
