_plot_added_variable_doc = """\
    Create an added variable plot for a fitted regression model.

    Parameters
    ----------
    %(extra_params_doc)sfocus_exog : int or string
        The column index of exog, or a variable name, indicating the
        variable whose role in the regression is to be assessed.
    resid_type : string
        The type of residuals to use for the dependent variable.  If
        None, uses `resid_deviance` for GLM/GEE and `resid` otherwise.
    use_glm_weights : bool
        Only used if the model is a GLM or GEE.  If True, the
        residuals for the focus predictor are computed using WLS, with
        the weights obtained from the IRLS calculations for fitting
        the GLM. If False, unweighted regression is used.
    fit_kwargs : dict, optional
        Keyword arguments to be passed to fit when refitting the
        model.
    ax : Axes instance
        Matplotlib Axes instance

    Returns
    -------
    fig : matplotlib Figure
        A matplotlib figure instance.
"""

_plot_partial_residuals_doc = """\
    Create a partial residual, or 'component plus residual' plot for a
    fited regression model.

    Parameters
    ----------
    %(extra_params_doc)sfocus_exog : int or string
        The column index of exog, or variable name, indicating the
        variable whose role in the regression is to be assessed.
    ax : Axes instance
        Matplotlib Axes instance

    Returns
    -------
    fig : matplotlib Figure
        A matplotlib figure instance.
"""

_plot_ceres_residuals_doc = """\
    Produces a CERES (Conditional Expectation Partial Residuals)
    plot for a fitted regression model.

    Parameters
    ----------
    %(extra_params_doc)sfocus_exog : integer or string
        The column index of results.model.exog, or the variable name,
        indicating the variable whose role in the regression is to be
        assessed.
    frac : float
        Lowess tuning parameter for the adjusted model used in the
        CERES analysis.  Not used if `cond_means` is provided.
    cond_means : array-like, optional
        If provided, the columns of this array span the space of the
        conditional means E[exog | focus exog], where exog ranges over
        some or all of the columns of exog (other than the focus exog).
    ax : matplotlib.Axes instance, optional
        The axes on which to draw the plot. If not provided, a new
        axes instance is created.

    Returns
    -------
    fig : matplotlib.Figure instance
        The figure on which the partial residual plot is drawn.

    References
    ----------
    RD Cook and R Croos-Dabrera (1998).  Partial residual plots in
    generalized linear models.  Journal of the American
    Statistical Association, 93:442.

    RD Cook (1993). Partial residual plots.  Technometrics 35:4.

    Notes
    -----
    `cond_means` is intended to capture the behavior of E[x1 |
    x2], where x2 is the focus exog and x1 are all the other exog
    variables.  If all the conditional mean relationships are
    linear, it is sufficient to set cond_means equal to the focus
    exog.  Alternatively, cond_means may consist of one or more
    columns containing functional transformations of the focus
    exog (e.g. x2^2) that are thought to capture E[x1 | x2].

    If nothing is known or suspected about the form of E[x1 | x2],
    set `cond_means` to None, and it will be estimated by
    smoothing each non-focus exog against the focus exog.  The
    values of `frac` control these lowess smooths.

    If cond_means contains only the focus exog, the results are
    equivalent to a partial residual plot.

    If the focus variable is believed to be independent of the
    other exog variables, `cond_means` can be set to an (empty)
    nx0 array.
"""
