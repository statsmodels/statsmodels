"""
Regression graphics for generalized linear models and GEE.

These functions are not intended to be directly called by users, there
are wrappers in GLMResults and GEEResults that use these functions.
"""

import numpy as np
from statsmodels.graphics import utils
import statsmodels.regression.linear_model as lm
from statsmodels.nonparametric.smoothers_lowess import lowess

def added_variable_plot(results, focus_col,
                        resid_type="resid_deviance",
                        use_weights=True,
                        glm_fit_kwargs=None, ax=None):

    fig, ax = utils.create_mpl_ax(ax)

    endog_resid, focus_exog_resid =\
                 added_variable_resids(results, focus_col,
                                       resid_type=resid_type,
                                       use_weights=use_weights,
                                       glm_fit_kwargs=glm_fit_kwargs)

    ax.plot(focus_exog_resid, endog_resid, 'o', alpha=0.6)
    ax.set_xlabel("exog residuals", size=15)
    ax.set_ylabel("endog residuals", size=15)

    return fig

def partial_residual_plot(results, focus_col,
                          frac=0.3, ax=None,
                          **lowess_kwargs):

    pr = partial_resids(results, focus_col)
    focus_exog = results.model.exog[:, focus_col]

    fig, ax = utils.create_mpl_ax(ax)
    ax.plot(focus_exog, pr, 'o', alpha=0.6)

    if frac > 0:
        ii = np.argsort(focus_exog)
        x0 = focus_exog[ii]
        y0 = pr[ii]
        lres = lowess(y0, x0, frac=frac, **lowess_kwargs)
        ax.plot(lres[:, 0], lres[:, 1], 'orange', lw=3, alpha=0.7)

    # TODO: Use the name of the variable if available
    ax.set_xlabel("Exog column %d" % focus_col, size=15)
    ax.set_ylabel("Partial residual", size=15)

    return fig

def ceres_plot(results, focus_col,
               ceres_frac={},
               frac=0.66, cond_means=None,
               ax=None):
    """
    Construct a CERES plot for a fitted generalized linear model.

    Parameters
    ----------
    results : GLMResults instance
        The fitted GLM model for which the CERES plot is constructed.
    ceres_frac : dict
        Map from column indices of results.model.exog to lowess
        smoothing parameters (frac keyword argument to lowess). Not
        used if `cond_means` is provided.
    frac : float
        The `frac` keyword argument to lowess for the lowess curve
        that is drawn on the CERES plot.  If non-positive, no
        lowess curve is drawn.
    cond_means : array-like, optional
        If provided, the columns of this array are the conditional
        means E[exog | focus exog], where exog ranges over some
        or all of the columns of exog other than focus exog.
    ax : matplotlib Axes instance
        The axes on which the CERES plot is drawn

    Returns
    -------
    The matplotlib figure on which the CERES plot is drawn.

    Notes
    -----
    If `cond_means` is not provided, it is obtained by smoothing each
    column of exog (except the focus column) against the focus column.
    The values of `ceres_frac` control these lowess smooths.
    """

    model = results.model
    n = model.exog.shape[0]
    m = model.exog.shape[1] - 1

    # Indices of non-focus columns
    ii = range(len(results.params))
    ii = list(ii)
    ii.pop(focus_col)

    if cond_means is None:
        cond_means = np.zeros((n, m))
        x0 = model.exog[:, focus_col]
        for j, i in enumerate(ii):
            y0 = model.exog[:, i]
            cf = ceres_frac[i] if i in ceres_frac else 0.66
            cond_means[:, j] = lowess(y0, x0, frac=cf, return_sorted=False)

    new_exog = np.concatenate((model.exog[:, ii], cond_means), axis=1)

    # Refit the model using the adjusted exog values
    klass = model.__class__
    kwargs = {}
    for key in model._init_keys:
        if hasattr(model, key):
            kwargs[key] = getattr(model, key)
    new_model = klass(model.endog, new_exog, **kwargs)
    new_result = new_model.fit()

    # The partial residual, with respect to l(x2) (notation of Cook 1998)
    presid = model.endog - new_result.fittedvalues
    presid *= model.family.link.deriv(new_result.fittedvalues)
    if cond_means.shape[1] > 0:
        presid += np.dot(cond_means, new_result.params[m:])

    fig, ax = utils.create_mpl_ax(ax)
    ax.plot(model.exog[:, focus_col], presid, 'o', alpha=0.6)

    if frac > 0:
        x0 = model.exog[:, focus_col]
        lres = lowess(presid, x0, frac=frac)
        ax.plot(lres[:, 0], lres[:, 1], 'orange', lw=3, alpha=0.7)

    # TODO: Use the name of the variable if available
    ax.set_xlabel("Exog column %d" % focus_col, size=15)
    ax.set_ylabel("CERES residual", size=15)

    return fig

def partial_resids(results, focus_col):
    """
    Returns partial residuals for the fitted GLM with respect to a
    subset of predictors called the 'focus predictors'.

    Parameters
    ----------
    results : GLMResults instance
        A fitted generalized linear model.
    focus col : int
        The column with respect to which the partial residuals are
        calculated.  The value corresponds to a column of model.exog.

    Returns
    -------
    An array of partial residuals.

    References
    ----------
    RD Cook and R Croos-Dabrera (1998).  Partial residual plots in
    generalized linear models.  Journal of the American Statistical
    Association, 93:442.
    """

    # The calculation follows equation (8) from Cook's paper.
    model = results.model
    resid = model.endog - results.fittedvalues
    resid *= model.family.link.deriv(results.fittedvalues)

    focus_val = results.params[focus_col] * model.exog[:, focus_col]

    return focus_val + resid

def added_variable_resids(results, focus_col,
                          resid_type="resid_deviance",
                          use_weights=True, glm_fit_kwargs=None):
    """
    Residualize the endog variable and a 'focus' exog variable in a
    GLM with respect to the other exog variables.

    Parameters
    ----------
    results : GLM results instance
        The fitted model incuding all predictors
    focus_col : integer
        The column of results.model.exog that is residualized against
        the other predictors.
    resid_type : string
        The type of residuals to use for the GLM dependent variable.
    use_weights : bool
        If True, the residuals for the focus predictor are computed
        using WLS, with the weights from the IRLS calculations for
        fitting the GLM.  If False, unweighted regression is used.
    glm_fit_kwargs : dict, optional
        Keyword arguments to be passed to fit when refitting the GLM.

    Returns
    -------
    endog_resid : array-like
        The residuals for the original exog
    focus_exog_resid : array-like
        The residuals for the focus predictor
    """

    model = results.model
    exog = model.exog
    endog = model.endog
    focus_exog = exog[:, focus_col]
    offset = model.offset if hasattr(model, "offset") else None
    exposure = model.exposure if hasattr(model, "exposure") else None

    ii = range(exog.shape[1])
    ii = list(ii)
    ii.pop(focus_col)
    reduced_exog = exog[:, ii]
    start_params = results.params[ii]

    klass = model.__class__

    # TODO: should we be able to assume that if something is in
    # init_keys then it is an attribute of the model?  Currently we
    # can have exposure in init_keys but not in the model.
    kwargs = {}
    for key in model._init_keys:
        if hasattr(model, key):
            kwargs[key] = getattr(model, key)

    new_model = klass(endog, reduced_exog, **kwargs)
    args = {"start_params": start_params}
    if glm_fit_kwargs is not None:
        args.update(glm_fit_kwargs)
    new_result = new_model.fit(**args)
    if not new_result.converged:
        raise ValueError("fit did not converge when calculating added variable residuals")

    try:
        endog_resid = getattr(new_result, resid_type)
    except AttributeError:
        raise ValueError("'%s' residual type not available" % resid_type)

    weights = model.family.weights(results.fittedvalues)

    # GEE doesn't have data_weights yet
    if hasattr(model, "data_weights"):
        weights = weights * model.data_weights

    if use_weights:
        lm_results = lm.WLS(focus_exog, reduced_exog, weights).fit()
    else:
        lm_results = lm.OLS(focus_exog, reduced_exog).fit()
    focus_exog_resid = lm_results.resid

    return endog_resid, focus_exog_resid
