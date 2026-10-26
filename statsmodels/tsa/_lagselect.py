"""
Least squares fits of a nested sequence of lag regressions

Lag order selection fits a sequence of regressions where each model adds a
block of lags to the previous one. All of them can be computed from a single
QR decomposition of the regressors of the largest model and the dependent
variable(s).
"""

from typing import NamedTuple

import numpy as np

__all__ = ["NestedLagOLSResults", "nested_lag_ols"]


class NestedLagOLSResults(NamedTuple):
    """
    Results of the nested regressions computed by ``nested_lag_ols``

    Order ``p`` refers to the regression on ``exog`` and the first ``p``
    lag blocks, ``p = 0, 1, ..., len(lags)``.

    Parameters
    ----------
    r : ndarray
        The upper triangular factor of the regressors of the largest model
        and the dependent variables, including any indicator columns.
    nobs : ndarray
        The number of observations used by each order.
    df_model : ndarray
        The number of regressors of each order, excluding indicator columns.
    ssr : ndarray
        The sum of squared residuals of each order, of shape (norder,) if
        endog is 1-d. The residual cross products of shape
        (norder, neqs, neqs) if endog is 2-d.
    params_last : ndarray
        The coefficient(s) of the last regressor of each order. NaN for an
        order without regressors.
    tvalues_last : ndarray
        The t-value(s) of the coefficient(s) of the last regressor of each
        order, using ``ssr / (nobs - df_model)`` as the residual variance.
    """

    r: np.ndarray
    nobs: np.ndarray
    df_model: np.ndarray
    ssr: np.ndarray
    params_last: np.ndarray
    tvalues_last: np.ndarray


def _as_blocks(lags, nobs):
    if isinstance(lags, np.ndarray):
        if lags.ndim != 2 or lags.shape[0] != nobs:
            raise ValueError("lags must have shape (nobs, nlags)")
        # one lag order per column
        return [lags], np.ones(lags.shape[1], dtype=int)
    blocks = []
    for block in lags:
        block = np.asarray(block)
        if block.ndim == 1:
            block = block[:, None]
        if block.ndim != 2 or block.shape[0] != nobs or block.shape[1] == 0:
            raise ValueError("each block in lags must have shape (nobs, k), k > 0")
        blocks.append(block)
    widths = np.array([block.shape[1] for block in blocks], dtype=int)
    return blocks, widths


def nested_lag_ols(endog, lags, exog=None, *, rcond, start=None, check_fit=True):
    """
    Least squares fits of a nested sequence of lag regressions from one QR

    Parameters
    ----------
    endog : ndarray
        The dependent variable, (nobs,), or variables, (nobs, neqs).
    lags : ndarray or list[ndarray]
        The lag blocks in the order in which they enter the regressions. If
        an ndarray of shape (nobs, nlags), each column is one block. If a
        list, each block has shape (nobs, k_j). Order p uses the first p
        blocks. The blocks are copied in groups of rows, so views of the data
        avoid forming the full matrix of lags.
    exog : ndarray, optional
        Regressors included in every order, of shape (nobs, k_exog).
    rcond : float or None
        The relative cutoff for small singular values of the least squares
        solver that the results replace. None is returned unless the bounds
        in the Notes show that the regressors of every order have a smallest
        singular value larger than ``10 * rcond`` times their largest one,
        so that the solver finds them to have full rank. None skips this
        check, which is only appropriate if it has been done already, e.g.,
        when the data are the rows of a triangular factor that has passed
        it.
    start : array_like, optional
        The first observation used by each order, a nondecreasing sequence
        of len(lags) + 1 integers starting at 0. If None, every order uses
        all observations. See Notes.
    check_fit : bool, optional
        If True (default), return None if the largest model fits (almost)
        perfectly, or its residuals are (almost) collinear, so that the
        sum of squared residuals is dominated by rounding.

    Returns
    -------
    NestedLagOLSResults or None
        The fits of orders 0, 1, ..., len(lags). None if the data are not
        finite or a check fails. The caller should then fit each order
        separately.

    Notes
    -----
    Let ``X = [exog, lags]`` with ``k`` columns and ``[X, endog] = QR``. The
    regression of order ``p`` uses the first ``c_p`` columns of ``X``, and

    * its residual cross product is ``R[c_p:, k:].T @ R[c_p:, k:]``, which
      is accumulated from the bottom so it does not suffer from
      cancellation;
    * the coefficient of its last regressor is
      ``R[c_p - 1, k:] / R[c_p - 1, c_p - 1]``.

    The rows are processed in blocks of about 1MB, and the triangular
    factor is updated block by block.

    If ``start`` is given, order ``p`` uses the observations
    ``start[p], ..., nobs - 1``. A regressor that is one at observation
    ``t`` and zero elsewhere removes that observation from a regression, so
    an indicator for each observation ``t`` with
    ``start[j - 1] <= t < start[j]`` is added just before the lag block
    ``j``. The values of lag block ``j`` before ``start[j]`` do not affect
    the results but must be finite. The observations from ``start[-1]`` on
    are not affected by the indicators and are first reduced to a
    triangular factor, which is then stacked with the remaining
    observations.

    The rank check bounds the singular values of all orders at once. The
    regressors of an order are a subset of the columns of ``X`` over a
    subset of its rows that includes the observations from ``start[-1]``
    on. Removing columns or adding rows does not decrease the smallest
    singular value, and taking a submatrix does not increase the largest,
    so the smallest singular value of ``X[start[-1]:]`` and the largest of
    ``X`` bound those of every order.
    """
    endog = np.asarray(endog)
    univariate = endog.ndim == 1
    if univariate:
        endog = endog[:, None]
    nobs, neqs = endog.shape
    if exog is None:
        exog = np.empty((nobs, 0))
    exog = np.asarray(exog)
    if exog.ndim != 2 or exog.shape[0] != nobs:
        raise ValueError("exog must have shape (nobs, k_exog)")
    blocks, widths = _as_blocks(lags, nobs)
    nlags = widths.shape[0]
    k_exog = exog.shape[1]
    k = k_exog + int(widths.sum())
    ncols = k + neqs
    if start is None:
        start = np.zeros(nlags + 1, dtype=int)
    else:
        start = np.asarray(start, dtype=int)
        if (
            start.shape != (nlags + 1,)
            or start[0] != 0
            or np.any(np.diff(start) < 0)
            or start[-1] > nobs
        ):
            raise ValueError(
                "start must be a nondecreasing sequence of len(lags) + 1 "
                "integers starting at 0"
            )
    nhead = int(start[-1])

    def fill(rows, first, last):
        rows[:, :k_exog] = exog[first:last]
        col = k_exog
        for block in blocks:
            rows[:, col : col + block.shape[1]] = block[first:last]
            col += block.shape[1]
        rows[:, k:] = endog[first:last]

    # Blocks of about 1MB keep the data in cache
    block_rows = max(4 * ncols, 2**17 // ncols)
    r = np.empty((0, ncols))
    for first in range(nhead, nobs, block_rows):
        last = min(first + block_rows, nobs)
        block = np.empty((r.shape[0] + last - first, ncols))
        block[: r.shape[0]] = r
        fill(block[r.shape[0] :], first, last)
        r = np.linalg.qr(block, mode="r")
    if r.shape[0] < ncols:
        # Fewer observations than columns
        r = np.vstack([r, np.zeros((ncols - r.shape[0], ncols))])
    head = np.empty((nhead, ncols))
    fill(head, 0, nhead)
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(head))):
        return None

    if rcond is not None and k:
        sv = np.linalg.svd(r[:k, :k], compute_uv=False)
        sv_max = sv[0]
        if nhead:
            sv_max = np.linalg.norm(np.vstack([r[:k, :k], head[:, :k]]), 2)
        # The factor 10 is a margin for rounding
        if not sv[-1] > 10 * rcond * sv_max:
            return None
    # The residual norm of each equation in the largest model, relative to
    # the norm of the dependent variable over its sample
    ynorm = np.sqrt(np.sum(r[:, k:] ** 2, 0))

    cum_widths = np.concatenate([[0], np.cumsum(widths)])
    if nhead:
        # Insert the indicators of the first nhead observations. The
        # indicator of observation t enters with lag block g, where
        # start[g - 1] <= t < start[g].
        group = np.searchsorted(start, np.arange(nhead), side="right")
        ind_cols = k_exog + np.arange(nhead) + cum_widths[group - 1]
        cols = np.arange(ncols)
        for j in range(nlags):
            loc = k_exog + cum_widths[j]
            cols[loc : loc + widths[j]] += start[j + 1]
        cols[k:] += nhead
        stacked = np.zeros((ncols + nhead, ncols + nhead))
        stacked[:ncols, cols] = r
        stacked[ncols:, cols] = head
        stacked[ncols + np.arange(nhead), ind_cols] = 1.0
        r = np.linalg.qr(stacked, mode="r")
    kk = k + nhead

    r_resid = r[kk:, kk:]
    if check_fit and not np.all(np.abs(np.diag(r_resid)) > 1e-5 * ynorm):
        # (Almost) perfect fit or collinear residuals, the residual cross
        # product is dominated by rounding
        return None

    # loc[p] is the number of columns of r in the regressors of order p
    loc = k_exog + start + cum_widths
    qy = r[:kk, kk:]
    if univariate:
        # ssr_tail[j] is the sum of squares of qy[j:]
        ssr_tail = np.append(np.cumsum(qy[::-1, 0] ** 2)[::-1], 0.0)
        ssr = r_resid[0, 0] ** 2 + ssr_tail[loc]
        ssr_diag = ssr[:, None]
    else:
        outer = qy[:, :, None] * qy[:, None, :]
        ssr_tail = np.zeros((kk + 1, neqs, neqs))
        ssr_tail[:kk] = np.cumsum(outer[::-1], axis=0)[::-1]
        ssr = r_resid.T @ r_resid + ssr_tail[loc]
        ssr_diag = np.diagonal(ssr, axis1=1, axis2=2)

    nobs_order = nobs - start
    df_model = k_exog + cum_widths
    params_last = np.full((nlags + 1, neqs), np.nan)
    tvalues_last = np.full((nlags + 1, neqs), np.nan)
    has_reg = df_model > 0
    last = loc[has_reg] - 1
    diag = r[last, last][:, None]
    params_last[has_reg] = qy[last] / diag
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.sqrt(ssr_diag / (nobs_order - df_model)[:, None])
        tvalues_last[has_reg] = np.sign(diag) * qy[last] / scale[has_reg]
    if univariate:
        params_last = params_last[:, 0]
        tvalues_last = tvalues_last[:, 0]
    return NestedLagOLSResults(r, nobs_order, df_model, ssr, params_last, tvalues_last)
