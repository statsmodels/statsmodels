"""
Measures and diagnostics for multicollinearity of explanatory variables

The functions work on a data matrix of explanatory variables or on a
covariance or correlation matrix.

Created on Tue Apr 28 15:09:45 2015

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
import pandas as pd

from statsmodels.stats.moment_helpers import cov2corr
from statsmodels.tools.validation import array_like, string_like

_EPS = np.finfo(float).eps


def _corr_nonconstant(data, is_moment, has_constant):
    """
    Correlation matrix of the columns of data that are not constant

    Parameters
    ----------
    data : array_like
        Data matrix or, if `is_moment` is True, covariance or correlation
        matrix.
    is_moment : bool
        Indicates whether `data` is a moment matrix.
    has_constant : {"raise", "skip"}
        How to treat constant columns.

    Returns
    -------
    corr : ndarray
        Correlation matrix of the columns that are not constant.
    is_const : ndarray of bool
        Indicator for the constant columns of `data`, one element for each
        column.
    """
    has_constant = string_like(has_constant, "has_constant", options=("raise", "skip"))
    x = array_like(data, "data", ndim=2)
    if not np.isfinite(x).all():
        raise ValueError("data must not contain missing or infinite values")

    if is_moment:
        if x.shape[0] != x.shape[1]:
            raise ValueError("a moment matrix must be square")
        if np.abs(x - x.T).max() > 1e-8 * np.abs(x).max():
            raise ValueError("a moment matrix must be symmetric")
        is_const = np.diag(x) == 0
        if (np.diag(x) < 0).any():
            raise ValueError(
                "a moment matrix must not have a negative diagonal element"
            )
    else:
        is_const = np.ptp(x, axis=0) == 0

    if is_const.any() and has_constant == "raise":
        raise ValueError(
            "data has constant columns at positions "
            f"{np.nonzero(is_const)[0].tolist()}. VIF is defined for a "
            "regression that includes a constant, remove the constant "
            "columns or use has_constant='skip'."
        )

    keep = ~is_const
    if is_moment:
        corr = cov2corr(x[np.ix_(keep, keep)])
    else:
        z = x[:, keep] - x[:, keep].mean(0)
        z /= np.sqrt((z**2).mean(0))
        corr = z.T @ z / z.shape[0]
    return corr, is_const


def _eigh_corr(corr):
    """
    Eigenvalue decomposition of a correlation matrix

    Raises a ValueError if the matrix is not positive semi-definite.
    """
    if corr.shape[0] == 0:
        return np.empty(0), np.empty((0, 0))
    evals, evecs = np.linalg.eigh(corr)
    if evals[0] < -np.sqrt(_EPS):
        raise ValueError("the moment matrix is not positive semi-definite")
    return evals, evecs


def _inv_diag(evals, evecs):
    """
    Diagonal of the inverse of a correlation matrix

    Variables that are exactly collinear with other variables have an
    infinite value, as is the case if the matrix is numerically singular.

    Parameters
    ----------
    evals : ndarray
        Eigenvalues in ascending order.
    evecs : ndarray
        Eigenvectors in the columns.
    """
    k_vars = evals.shape[0]
    if k_vars < 2:
        return np.ones(k_vars)

    # numerical rank, using the same type of tolerance as matrix_rank
    singular = evals <= evals[-1] * k_vars * _EPS
    evecs2 = evecs**2
    res = (evecs2[:, ~singular] / evals[~singular]).sum(1)
    if singular.any():
        # A variable is collinear with the others if it has a nonzero
        # weight in the null space. All other variables have a finite VIF
        # that is given by the pseudo inverse. The weight of the other
        # variables in the numerical null space is not exactly zero. It is
        # of the order of the squared relative error of the eigenvectors,
        # plus a term that is proportional to the size of the numerically
        # zero eigenvalues, which are not exactly zero.
        gap = evals[~singular].min()
        spill = np.abs(evals[singular]).max()
        threshold = (10 * _EPS * evals[-1] / gap) ** 2 + spill / gap
        res[evecs2[:, singular].sum(1) > threshold] = np.inf
    # VIF is larger than one by definition, remove floating point noise
    return np.maximum(res, 1.0)


def vif(data, *, is_moment=False, has_constant="raise"):
    r"""
    Variance inflation factors of all variables

    The variance inflation factor, VIF, of a variable is :math:`1 / (1 -
    R^2)`, where :math:`R^2` is the coefficient of determination of the
    regression of the variable on all other variables and a constant. It
    measures by how much the variance of the estimate of the coefficient of
    the variable is inflated by the correlation with the other variables.

    Parameters
    ----------
    data : array_like
        The explanatory variables with observations in rows and variables in
        columns. The data should not include a constant, see `has_constant`.
        If `is_moment` is True, then this is a square covariance or
        correlation matrix of the variables.
    is_moment : bool, optional
        If True, then `data` is a symmetric, positive semi-definite moment
        matrix of the variables instead of the observations. Only the
        correlations are used, so the matrix can be a covariance or a
        correlation matrix. The default is False.
    has_constant : {"raise", "skip"}, optional
        How to treat constant columns, which have zero variance. If "raise",
        the default, then a ValueError is raised. If "skip", then the
        constant columns are ignored and the corresponding elements of the
        result are NaN. The other variables are not affected because the
        regressions always include a constant.

    Returns
    -------
    ndarray or Series
        The variance inflation factors, one for each column of `data`. This
        is a pandas Series with the column labels as index if `data` is a
        pandas DataFrame. The VIF of a variable that is a linear combination
        of other variables is infinite.

    Raises
    ------
    ValueError
        If `data` contains missing or infinite values, if `data` has
        constant columns and `has_constant` is "raise", or if `is_moment` is
        True and `data` is not a valid moment matrix.

    See Also
    --------
    statsmodels.stats.outliers_influence.variance_inflation_factor
        VIF of a single variable, estimated by OLS.
    vif_ridge
        VIF of the parameter estimates of ridge regression.
    collinear_index
        Columns that are perfectly collinear with preceding columns.

    Notes
    -----
    The VIF of all variables are the diagonal elements of the inverse of the
    correlation matrix of the variables. This needs one matrix decomposition
    instead of one regression for each variable, which is much faster than
    calling
    :func:`~statsmodels.stats.outliers_influence.variance_inflation_factor`
    for each column. Up to numerical precision, the results are the same as
    those of that function with its default ``standardize=True``. The latter
    limits the VIF to :math:`10^{15}` and warns if the design matrix is
    ill-conditioned, while this function returns ``inf`` for variables that
    are collinear with other variables.

    **Constants**

    The VIF requires a constant in the regressions. The variables are
    demeaned, which is equivalent to including a constant, and the constant
    column of a design matrix, for example ``res.model.exog``, has to be
    removed or skipped. A set of dummy variables that sums to one is an
    implicit constant. If the design matrix does not include another
    constant, then it has to be dropped, because the dummy variables are
    perfectly collinear after demeaning and their VIF is infinite.

    If `is_moment` is True and the moment matrix is not centered, for
    example the cross-product matrix ``x.T @ x``, then the VIF are those of
    regressions without a constant.

    **Singular and nearly singular data**

    The computation uses the eigenvalues of the correlation matrix. The VIF
    is infinite for variables that are numerically perfectly collinear with
    other variables, i.e. if the correlation matrix is singular up to
    floating point precision. A variable that is not part of such a linear
    relationship has a finite VIF. VIFs above about :math:`10^{12}` should be
    read as indicating that the variable is collinear because they are
    affected by floating point noise. Floating point precision of the data
    matters: a linear combination of single precision (float32) variables
    is collinear only up to the precision of the single precision data, and
    is not detected as perfectly collinear.

    References
    ----------
    .. [1] Belsley, D. A., E. Kuh, and R. E. Welsch. 1980. Regression
       Diagnostics: Identifying Influential Data and Sources of
       Collinearity. Wiley.

    Examples
    --------
    The VIF of the explanatory variables of the Longley data, which are
    highly collinear.

    >>> import statsmodels.api as sm
    >>> from statsmodels.stats.api import vif
    >>> exog = sm.datasets.longley.load_pandas().exog
    >>> vif(exog).round(1)
    GNPDEFL     135.5
    GNP        1788.5
    UNEMP        33.6
    ARMED         3.6
    POP         399.2
    YEAR        759.0
    dtype: float64

    The design matrix of a model that is estimated with a formula has a
    constant, which is skipped.

    >>> import statsmodels.formula.api as smf
    >>> mod = smf.ols("TOTEMP ~ GNPDEFL + GNP + UNEMP + ARMED + POP + YEAR",
    ...               data=sm.datasets.longley.load_pandas().data)
    >>> vif(mod.data.orig_exog, has_constant="skip").round(1)
    Intercept       NaN
    GNPDEFL       135.5
    GNP          1788.5
    UNEMP          33.6
    ARMED           3.6
    POP           399.2
    YEAR          759.0
    dtype: float64
    """
    corr, is_const = _corr_nonconstant(data, is_moment, has_constant)
    res = np.full(is_const.shape[0], np.nan)
    res[~is_const] = _inv_diag(*_eigh_corr(corr))
    if isinstance(data, pd.DataFrame):
        res = pd.Series(res, index=data.columns)
    return res


def vif_ridge(data, pen_factors, *, is_moment=False, has_constant="raise"):
    r"""
    Variance inflation factors for ridge regression

    The penalization is on the standardized variables, so the VIF depend on
    the correlation matrix of the variables. The ridge factors are added to
    the diagonal of the correlation matrix.

    Parameters
    ----------
    data : array_like
        The explanatory variables with observations in rows and variables in
        columns. The data should not include a constant, see `has_constant`.
        If `is_moment` is True, then this is a square covariance or
        correlation matrix of the variables.
    pen_factors : array_like
        One-dimensional array of non-negative ridge penalization factors.
    is_moment : bool, optional
        If True, then `data` is a symmetric, positive semi-definite moment
        matrix of the variables instead of the observations. The default is
        False.
    has_constant : {"raise", "skip"}, optional
        How to treat constant columns, which have zero variance. If "raise",
        the default, then a ValueError is raised. If "skip", then the
        constant columns are ignored and the corresponding elements of the
        result are NaN.

    Returns
    -------
    ndarray or DataFrame
        The variance inflation factors with the penalization factors in rows
        and the variables in columns. This is a pandas DataFrame with the
        penalization factors as index and the column labels of `data` as
        columns if `data` is a pandas DataFrame.

    Raises
    ------
    ValueError
        If `data` contains missing or infinite values, if `data` has
        constant columns and `has_constant` is "raise", if a penalization
        factor is negative, or if `is_moment` is True and `data` is not a
        valid moment matrix.

    See Also
    --------
    vif : VIF of regression without penalization.

    Notes
    -----
    With the correlation matrix :math:`R` and the penalization factor
    :math:`k`, the VIF of ridge regression are the diagonal elements of
    :math:`(R + k I)^{-1} R (R + k I)^{-1}`. The result for :math:`k=0` is
    the same as that of :func:`vif`. The correlation matrix is singular if
    some variables are exactly collinear, in which case the penalization
    factors have to be positive. A factor of zero is handled as in
    :func:`vif`.

    All penalization factors are computed with one eigenvalue decomposition
    of the correlation matrix.

    References
    ----------
    .. [1] Marquardt, D. W. 1970. Generalized inverses, ridge regression,
       biased linear estimation, and nonlinear estimation. Technometrics
       12 (3): 591-612.

    Examples
    --------
    The ridge penalization removes most of the collinearity of the Longley
    data. The penalization factors are added to the diagonal of the
    correlation matrix, so they are on the scale of standardized variables.
    The columns are standardized first to make this explicit, the result is
    the same as for the original data, which `vif_ridge` standardizes
    through the correlation matrix. The row with a penalization factor of
    zero contains the VIF without penalization.

    >>> import statsmodels.api as sm
    >>> from statsmodels.stats.api import vif_ridge
    >>> exog = sm.datasets.longley.load_pandas().exog
    >>> exog_std = (exog - exog.mean()) / exog.std()
    >>> vif_ridge(exog_std, [0, 0.01, 0.1]).round(1)
          GNPDEFL     GNP  UNEMP  ARMED    POP   YEAR
    0.00    135.5  1788.5   33.6    3.6  399.2  759.0
    0.01     17.2     3.5    2.7    2.1   13.8   10.0
    0.10      0.8     0.3    1.5    1.2    0.6    0.2
    """
    corr, is_const = _corr_nonconstant(data, is_moment, has_constant)
    pen = array_like(pen_factors, "pen_factors", ndim=1)
    if not np.isfinite(pen).all() or (pen < 0).any():
        raise ValueError("pen_factors must be finite and non-negative")

    evals, evecs = _eigh_corr(corr)
    evals_pos = np.maximum(evals, 0)
    # diagonal of V diag(w / (w + k)^2) V' for all penalization factors k
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = evals_pos[:, None] / (evals_pos[:, None] + pen) ** 2
    vif_pen = ((evecs**2) @ scale).T
    zero = pen == 0
    if zero.any():
        vif_pen[zero] = _inv_diag(evals, evecs)

    res = np.full((pen.shape[0], is_const.shape[0]), np.nan)
    res[:, ~is_const] = vif_pen
    if isinstance(data, pd.DataFrame):
        res = pd.DataFrame(res, index=pen, columns=data.columns)
    return res


def collinear_index(data, tol=None):
    """
    Find the columns that are perfectly collinear with preceding columns

    The columns are processed in sequence. A column is collinear if it is in
    the space spanned by the preceding columns that are not collinear.

    Parameters
    ----------
    data : array_like
        The design matrix with observations in rows and variables in
        columns. The matrix is not demeaned and the columns are not
        standardized. If a constant is part of the design, then it should be
        in the first column, for example by using
        :func:`~statsmodels.tools.tools.add_constant`.
    tol : float, optional
        A column is collinear if the norm of its residuals in the regression
        on the preceding columns is less than or equal to `tol` times the
        norm of the column. The default is ``max(nobs, k_vars)`` times the
        machine epsilon, which is the default of
        :func:`numpy.linalg.matrix_rank`.

    Returns
    -------
    idx_collinear : ndarray or Index
        Position of the columns that are collinear with preceding columns.
        This is an Index with the column labels if `data` is a pandas
        DataFrame.
    idx_keep : ndarray or Index
        Position of the columns that are not collinear with preceding
        columns. This is an Index with the column labels if `data` is a
        pandas DataFrame.

    See Also
    --------
    vif : VIF for all columns, which is infinite for collinear columns.

    Notes
    -----
    Removing the collinear columns, i.e. selecting `idx_keep`, gives a
    design matrix of full column rank. Which of the columns that are part of
    a collinear relationship is flagged depends on the order of the columns,
    only the last column of each relationship is flagged.

    The residuals are computed with Gram-Schmidt orthogonalization of the
    columns with reorthogonalization. A column that is flagged does not
    contribute to the orthogonal basis for the columns that follow it.

    Examples
    --------
    A constant and a full set of dummy variables are collinear.

    >>> import numpy as np
    >>> from statsmodels.stats.api import collinear_index
    >>> dummies = np.tile(np.eye(3), (4, 1))
    >>> x = np.column_stack((np.ones(12), dummies))
    >>> idx_collinear, idx_keep = collinear_index(x)
    >>> idx_collinear
    array([3])
    >>> idx_keep
    array([0, 1, 2])
    """
    x = array_like(data, "data", ndim=2)
    if not np.isfinite(x).all():
        raise ValueError("data must not contain missing or infinite values")
    nobs, k_vars = x.shape
    if tol is None:
        tol = max(nobs, k_vars) * _EPS

    norm = np.linalg.norm(x, axis=0)
    basis = np.empty((nobs, k_vars))
    n_basis = 0
    is_collinear = np.zeros(k_vars, dtype=bool)
    for j in range(k_vars):
        resid = x[:, j]
        for _ in range(2):
            # two passes of classical Gram-Schmidt for orthogonality
            b = basis[:, :n_basis]
            resid = resid - b @ (b.T @ resid)
        resid_norm = np.linalg.norm(resid)
        if resid_norm <= tol * norm[j]:
            is_collinear[j] = True
        else:
            basis[:, n_basis] = resid / resid_norm
            n_basis += 1

    idx_collinear = np.nonzero(is_collinear)[0]
    idx_keep = np.nonzero(~is_collinear)[0]
    if isinstance(data, pd.DataFrame):
        return data.columns[idx_collinear], data.columns[idx_keep]
    return idx_collinear, idx_keep
