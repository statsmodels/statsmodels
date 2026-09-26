"""
Diagnostic tests for ordered models.

Created on 2026-09-13
License: BSD-3
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from statsmodels.iolib.table import SimpleTable, default_html_fmt
from statsmodels.iolib.tableformatting import fmt_base
from statsmodels.tools.tools import add_constant


class BrantResults:
    """
    Results class for the Brant test of the parallel regression assumption.

    Parameters
    ----------
    statistic : pd.Series or np.ndarray
        Chi-squared statistics for the Omnibus test and each variable.
    pvalues : pd.Series or np.ndarray
        P-values corresponding to each chi-squared statistic.
    df : pd.Series or np.ndarray
        Degrees of freedom for each test.
    exog_names : list of str
        Names of explanatory variables tested.
    """

    def __init__(
        self,
        statistic: pd.Series | np.ndarray,
        pvalues: pd.Series | np.ndarray,
        df: pd.Series | np.ndarray,
        exog_names: list[str],
    ):
        self.statistic = statistic
        self.pvalues = pvalues
        self.df = df
        self.exog_names = exog_names

        # Convenience scalar properties for omnibus
        if isinstance(statistic, pd.Series):
            self.chi2_omnibus = float(statistic.get("Omnibus", statistic.iloc[0]))
            self.pvalue_omnibus = float(pvalues.get("Omnibus", pvalues.iloc[0]))
            self.df_omnibus = int(df.get("Omnibus", df.iloc[0]))
        else:
            self.chi2_omnibus = float(statistic[0])
            self.pvalue_omnibus = float(pvalues[0])
            self.df_omnibus = int(df[0])

    def summary_frame(self) -> pd.DataFrame:
        """
        Return the test results as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['chi2', 'df', 'p_value'].
        """
        if isinstance(self.statistic, pd.Series):
            index = self.statistic.index
            chi2_vals = self.statistic.values
            df_vals = self.df.values
            pval_vals = self.pvalues.values
        else:
            index = ["Omnibus"] + list(self.exog_names)
            chi2_vals = self.statistic
            df_vals = self.df
            pval_vals = self.pvalues

        return pd.DataFrame(
            {"chi2": chi2_vals, "df": df_vals, "p_value": pval_vals},
            index=index,
        )

    def summary(self) -> SimpleTable:
        """
        Return a formatted summary table of the Brant test results.

        Returns
        -------
        SimpleTable
            Table with columns: Variable, chi2, df, p-value.
        """
        from copy import deepcopy

        frame = self.summary_frame()
        labels = list(frame.index)

        table_data = []
        for label in labels:
            row = frame.loc[label]
            table_data.append(
                [
                    label,
                    f"{row['chi2']:.4f}",
                    f"{int(row['df'])}",
                    f"{row['p_value']:.4f}",
                ]
            )

        headers = ["Variable", "chi2", "df", "p-value"]
        title = "Brant Test of Parallel Regression (Proportional Odds) Assumption"

        fmt = deepcopy(fmt_base)
        fmt["data_fmts"] = ["%-15s", "%10s", "%6s", "%10s"]
        fmt_html = deepcopy(default_html_fmt)

        table = SimpleTable(
            table_data,
            headers=headers,
            title=title,
            txt_fmt=fmt,
            html_fmt=fmt_html,
        )
        return table

    def __str__(self) -> str:
        return str(self.summary())

    def __repr__(self) -> str:
        return self.__str__()


def brant_test(results, by_var: bool = True) -> BrantResults:
    r"""
    Brant test of the parallel regression assumption for OrderedModel.

    The Brant test (Brant 1990) tests the parallel regression (proportional odds)
    assumption in ordinal logistic regression models. It evaluates whether the
    slope coefficients are equal across separate binary logistic regressions
    corresponding to each cutpoint.

    Parameters
    ----------
    results : Results instance
        A fitted `OrderedModel` results instance (or its wrapper) with
        `distr='logit'`.
    by_var : bool, default True
        If True, returns individual chi-squared statistics for each
        explanatory variable in addition to the overall Omnibus test.

    Returns
    -------
    BrantResults
        An instance containing:
        - `statistic`: Omnibus and per-variable chi-squared statistics
        - `pvalues`: Corresponding p-values from the chi-squared distribution
        - `df`: Degrees of freedom for each test
        - `summary()`: Formatted SimpleTable summary
        - `summary_frame()`: Pandas DataFrame representation

    Raises
    ------
    ValueError
        If the model does not use `distr='logit'` or has fewer than 3 categories.

    References
    ----------
    .. [1] Brant, R. (1990). "Assessing proportionality in the proportional odds
           model for ordinal logistic regression." Biometrics, 1171-1178.
    """
    # Unwrap ResultsWrapper if present
    res = getattr(results, "_results", results)

    # Check distribution
    distr = getattr(res.model, "distr", None)
    distr_name = getattr(distr, "name", str(distr))
    if distr_name not in ("logistic", "logit"):
        raise ValueError(
            "Brant test is only valid for OrderedModel with distr='logit' "
            f"(proportional odds logistic regression), got distr='{distr_name}'."
        )

    # Check number of categories
    k_levels = getattr(res.model, "k_levels", None)
    if k_levels is None or k_levels < 3:
        raise ValueError(
            "Brant test requires an OrderedModel with at least 3 categories."
        )

    J = k_levels
    y = np.asarray(res.model.endog, dtype=int)
    categories = np.unique(y)
    if len(categories) != J:
        # Fallback if categories are not dense integers
        categories = np.sort(categories)

    X_orig = np.asarray(res.model.exog, dtype=float)
    X = add_constant(X_orig, prepend=True, has_constant="skip")
    K = X_orig.shape[1]

    # Get predictor variable names
    exog_names = None
    if hasattr(res.model, "data") and hasattr(res.model.data, "param_names"):
        names = res.model.data.param_names
        if names is not None and len(names) >= K:
            exog_names = list(names[:K])
    if exog_names is None and hasattr(res.model, "exog_names"):
        names = res.model.exog_names
        if names is not None and len(names) >= K:
            exog_names = list(names[:K])
    if exog_names is None:
        exog_names = [f"x{i+1}" for i in range(K)]

    # Fit J - 1 binary logit models
    from statsmodels.discrete.discrete_model import Logit

    beta_hat = []
    var_hat = []
    pi_hat = []

    for m in range(J - 1):
        cut_val = categories[m]
        z_m = (y > cut_val).astype(float)
        bin_mod = Logit(z_m, X)
        bin_res = bin_mod.fit(disp=False)
        beta_hat.append(bin_res.params)
        var_hat.append(bin_res.cov_params())
        pi_hat.append(bin_res.predict())

    # Concatenate slopes (excluding intercept at index 0)
    beta_star = np.concatenate([b[1:] for b in beta_hat])

    # Construct var_beta: shape ((J-1)*K, (J-1)*K)
    var_beta = np.zeros(((J - 1) * K, (J - 1) * K))

    # Diagonal blocks
    for m in range(J - 1):
        var_beta[m * K : (m + 1) * K, m * K : (m + 1) * K] = var_hat[m][1:, 1:]

    # Off-diagonal covariance blocks (Brant 1990)
    Xt = X.T
    for m in range(J - 2):
        for l_idx in range(m + 1, J - 1):
            pi_m = pi_hat[m]
            pi_l = pi_hat[l_idx]
            w_ml = pi_l - pi_m * pi_l
            w_m = pi_m - pi_m * pi_m
            w_l = pi_l - pi_l * pi_l

            inv_wm = np.linalg.pinv(Xt @ (w_m[:, None] * X))
            wml_mid = Xt @ (w_ml[:, None] * X)
            inv_wl = np.linalg.pinv(Xt @ (w_l[:, None] * X))

            cov_block = (inv_wm @ wml_mid @ inv_wl)[1:, 1:]
            var_beta[m * K : (m + 1) * K, l_idx * K : (l_idx + 1) * K] = cov_block
            var_beta[l_idx * K : (l_idx + 1) * K, m * K : (m + 1) * K] = cov_block.T

    # Difference contrast matrix D: compares beta_0 with beta_i for i = 1 ... J-2
    I_K = np.eye(K)
    D_rows = []
    for i in range(1, J - 1):
        row_blocks = [np.zeros((K, K)) for _ in range(J - 1)]
        row_blocks[0] = I_K
        row_blocks[i] = -I_K
        D_rows.append(np.hstack(row_blocks))
    D = np.vstack(D_rows)

    # Omnibus test
    D_beta = D @ beta_star
    cov_D = D @ var_beta @ D.T
    inv_cov_D = np.linalg.pinv(cov_D)
    chi2_omnibus = float(D_beta.T @ inv_cov_D @ D_beta)
    df_omnibus = (J - 2) * K
    pval_omnibus = float(stats.chi2.sf(chi2_omnibus, df_omnibus))

    statistics = [chi2_omnibus]
    pvalues = [pval_omnibus]
    dfs = [df_omnibus]
    index_names = ["Omnibus"]

    if by_var:
        for k in range(K):
            s = [k + m * K for m in range(J - 1)]
            D_s = D[:, s]
            non_zero = ~np.all(np.isclose(D_s, 0), axis=1)
            D_sk = D_s[non_zero]
            beta_sk = beta_star[s]
            var_sk = var_beta[np.ix_(s, s)]
            cov_D_sk = D_sk @ var_sk @ D_sk.T
            inv_cov_sk = np.linalg.pinv(cov_D_sk)
            chi2_k = float((D_sk @ beta_sk).T @ inv_cov_sk @ (D_sk @ beta_sk))
            df_k = J - 2
            pval_k = float(stats.chi2.sf(chi2_k, df_k))

            statistics.append(chi2_k)
            pvalues.append(pval_k)
            dfs.append(df_k)
            index_names.append(exog_names[k])

    stat_series = pd.Series(statistics, index=index_names, name="chi2")
    pval_series = pd.Series(pvalues, index=index_names, name="p_value")
    df_series = pd.Series(dfs, index=index_names, name="df")

    return BrantResults(
        statistic=stat_series,
        pvalues=pval_series,
        df=df_series,
        exog_names=exog_names,
    )
