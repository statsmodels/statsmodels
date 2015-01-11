import statsmodels.api as sm
import pandas as pd
import patsy
import numpy as np
import warnings

"""
A predict-like function that constructs means and pointwise confidence
bands for the function f(x) = E[Y | X*=x, X1=x1, ...], where X* is the
focus variable and X1, X2, ... are non-focus variables.  This is
especially useful when conducting a functional regression in which the
role of x is modeled with b-splines or other basis functions.
"""


def _make_formula_exog(result, focus_var, summaries, values, num_points):
    """
    Create dataframes for exploring a fitted model as a function of one variable.

    This works for models fit with a formula.

    Returns
    -------
    dexog : data frame
        A data frame in which the focus variable varies and the other variables
        are fixed at specified or computed values.
    fexog : data frame
        The data frame `dexog` processed through the model formula.
    """

    model = result.model
    exog = model.data.frame

    colnames = summaries.keys() + values.keys() + [focus_var]

    # Check for variables whose values are not set either through
    # `values` or `summaries`.  Since the model data frame can contain
    # extra variables not referenced in the formula RHS, this may not
    # be a problem, so just warn.  There is no obvious way to extract
    # from a formula all the variable names that it references.
    varl = set(exog.columns.tolist()) - set([model.endog_names])
    unmatched = varl - set(colnames)
    unmatched = list(unmatched)
    if len(unmatched) > 0:
        warnings.warn("%s in data frame but not in summaries or values."
                      % ", ".join(["'%s'" % x for x in unmatched]))

    fexog = pd.DataFrame(index=range(num_points), columns=colnames)

    # The values of the 'focus variable' are a sequence of percentiles
    pctls = np.linspace(0, 100, num_points).tolist()
    fvals = np.percentile(exog[focus_var], pctls)
    fexog.loc[:, focus_var] = fvals

    # The values of the other variables may be given by summary functions...
    for ky in summaries.iterkeys():
        fexog.loc[:, ky] = summaries[ky](exog.loc[:, ky])

    # or they may be provided as given values.
    for ky in values.iterkeys():
        fexog.loc[:, ky] = values[ky]

    dexog = patsy.dmatrix(model.data.orig_exog.design_info.builder, fexog, return_type='dataframe')
    return dexog, fexog, fvals


def _make_exog(result, focus_var, summaries, values, num_points):
    """
    Create dataframes for exploring a fitted model as a function of one variable.

    This works for models fit without a formula.

    Returns
    -------
    exog : data frame
        A data frame in which the focus variable varies and the other variables
        are fixed at specified or computed values.
    """

    model = result.model
    model_exog = model.exog
    exog_names = model.exog_names

    exog = np.zeros((num_points, model_exog.shape[1]))

    # Check for variables whose values are not set either through
    # `values` or `summaries`.
    colnames = values.keys() + summaries.keys() + [focus_var]
    unmatched = set(exog_names) - set(colnames)
    unmatched = list(unmatched)
    if len(unmatched) > 0:
        warnings.warn("%s in model but not in `summaries` or `values`."
                      % ", ".join(["'%s'" % x for x in unmatched]))

    # The values of the 'focus variable' are a sequence of percentiles
    pctls = np.linspace(0, 100, num_points).tolist()
    ix = exog_names.index(focus_var)
    fvals = np.percentile(model_exog[:, ix], pctls)
    exog[:, ix] = fvals

    # The values of the other variables may be given by summary functions...
    for ky in summaries.iterkeys():
        ix = exog_names.index(ky)
        exog[:, ix] = summaries[ky](model_exog[:, ix])

    # or they may be provided as given values.
    for ky in values.iterkeys():
        ix = exog_names.index(ky)
        exog[:, ix] = values[ky]

    return exog, fvals


def predict_functional(result, focus_var, summaries=None, values=None,
                       cvrg_prob=0.95, simultaneous=False, num_points=10, **kwargs):
    """
    Returns predictions of a fitted model as a function of a given covariate.

    The value of the focus variable varies along a sequence of its
    quantiles, calculated from the data used to fit the model.  The
    other variables are held constant either at given values, or at
    values computed by applying given summary functions to the data
    used to fit the model.

    Parameters
    ----------
    result : statsmodels result object
        A results object for the fitted model.
    focus_var : string
        The name of the 'focus variable'.
    summaries : dict-like
        A map from names of non-focus variables to summary functions.
        Each summary function is applied to the data used to fit the
        model, to obtain a value at which the variable is held fixed.
    values : dict-like
        Values at which a given non-focus variable is held fixed.
    cvrg_prob : float
        The coverage probability.
    simultaneous : bool
        If true, the confidence band is simultaneous, otherwise it is
        pointwise.
    num_points : integer
        The number of equally-spaced quantile points where the
        prediction is made.
    kwargs :
        Arguments passed to the `predict` method.

    Returns
    -------
    pred : array-like
        The predicted mean values.
    cb : array-like
        An array with two columns, containing respectively the lower
        and upper limits of a confidence band.
    fvals : array-like
        The values of the focus variable at which the prediction is
        made.

    Notes
    -----
    All variables in the model except for the focus variable should be
    included as a key in either `summaries` or `values`.

    These are conditional means, based on specified values of the
    non-focus variables.  They are not marginal (unconditional) means.

    Examples
    --------
    Fit a model using a formula in which the predictors are age
    (modeled with splines), ethnicity (which is categorical), gender,
    and income.  Then we obtain the fitted mean values as a function
    of age for females with mean income and the most common
    ethnicity.

    >>> model = sm.OLS.from_formula('y ~ bs(age, df=4) + C(ethnicity) + gender + income', data)
    >>> result = model.fit()
    >>> mode = lambda x : x.value_counts().argmax()
    >>> summaries = {'income': np.mean, ethnicity=mode}
    >>> values = {'gender': 'female'}
    >>> pr, cb, x = predict_functional(result, 'age', summaries, values)

    Fit a model using arrays.  Plot the means as a function of x3,
    holding x1 fixed at its mean value in the data used to fit the
    model, and holding x2 fixed at 1.

    >>> model = sm.OLS(y ,x)
    >>> result = model.fit()
    >>> summaries = {'x1': np.mean}
    >>> values = {'x2': 1}
    >>> pr, cb, x = predict_functional(result, 'x3', summaries, values)
    """

    if summaries is None:
        summaries = {}
    if values is None:
        values = {}

    ky = set(values.keys()) & set(summaries.keys())
    if len(ky) > 0:
        raise ValueError("%s included in both `values` and `summaries`" %
                         ", ".join(ky))

    # Branch depending on whether the model was fit with a formula.
    if hasattr(result.model.data, "frame"):
        dexog, fexog, fvals = _make_formula_exog(result, focus_var, summaries, values, num_points)
    else:
        exog, fvals = _make_exog(result, focus_var, summaries, values, num_points)
        dexog, fexog = exog, exog

    pred = result.predict(exog=fexog, **kwargs)
    t_test = result.t_test(dexog)

    if simultaneous:
        sd = t_test.sd
        cb = np.zeros((num_points, 2))

        # Scheffe's method
        from scipy.stats.distributions import f as fdist
        df1 = result.model.exog.shape[1]
        df2 = result.model.exog.shape[0] - df1
        qf = fdist.cdf(cvrg_prob, df1, df2)
        fx = sd * np.sqrt(df1 * qf)
        cb[:, 0] = pred - fx
        cb[:, 1] = pred + fx

    else:
        cb = t_test.conf_int(alpha=1-cvrg_prob)

    return pred, cb, fvals


def _glm_basic_scr(result, exog, cvrg_prob):
    """
    The basic SCR from (Sun et al. Annals of Statistics 2000).

    Parameters
    ----------
    result : results instance
        The fitted GLM results instance
    exog : array-like
        The exog values spanning the interval
    cvrg_prob : float
        Coverage probability.

    Returns
    -------
    An array with two columns, containing the lower and upper
    confidence bounds, respectively.

    Notes
    -----
    The rows of `exog` should be a sequence of covariate values
    obtained by taking one 'free variable' x and varying it over an
    interval.  The matrix `exog` is thus the basis functions and any
    other covariates evaluated as x varies.
    """

    model = result.model
    n = model.exog.shape[0]

    # Get the Hessian without recomputing.
    cov = result.cov_params()
    hess = np.linalg.inv(cov)

    # Proposition 3.1 of Sun et al.
    A = hess / n
    B = np.linalg.cholesky(A).T # Upper Cholesky triangle

    # The variance and SD of the linear predictor at each row of exog.
    sigma2 = (np.dot(exog, cov) * exog).sum(1)
    sigma = np.sqrt(sigma2)

    # Calculate kappa_0 (formula 42 from Sun et al)
    bz = np.linalg.solve(B.T, exog.T).T
    bz /= np.sqrt(n)
    bz /= sigma[:, None]
    bzd = np.diff(bz, 1, axis=0)
    bzdn = (bzd**2).sum(1)
    kappa_0 = np.sqrt(bzdn).sum()

    from scipy.stats.distributions import norm

    # The root of this function is the multiplier for the confidence
    # band, see Sun et al. equation 35.
    def func(c):
        return kappa_0 * np.exp(-c**2/2) / np.pi + 2*(1 - norm.cdf(c)) - (1 - cvrg_prob)

    from scipy.optimize import brentq

    c, rslt = brentq(func, 1, 10, full_output=True)
    if rslt.converged == False:
        raise ValueError("Root finding error in basic SCR")

    return sigma, c

def predict_functional_glm(result, focus_var, summaries=None, values=None, cvrg_prob=0.95,
                           simultaneous=False, num_points=10, linear=False, **kwargs):

    if summaries is None:
        summaries = {}
    if values is None:
        values = {}

    ky = set(values.keys()) & set(summaries.keys())
    if len(ky) > 0:
        raise ValueError("%s included in both `values` and `summaries`" %
                         ", ".join(ky))

    # Branch depending on whether the model was fit with a formula.
    if hasattr(result.model.data, "frame"):
        dexog, fexog, fvals = _make_formula_exog(result, focus_var, summaries, values, num_points)
    else:
        exog, fvals = _make_exog(result, focus_var, summaries, values, num_points)
        dexog, fexog = exog, exog

    kwargs_pred = kwargs.copy()
    kwargs_pred.update({"linear": True})
    pred = result.predict(exog=fexog, **kwargs_pred)
    t_test = result.t_test(dexog)

    if simultaneous:

        sigma, c = _glm_basic_scr(result, exog, cvrg_prob)
        cb = np.zeros((exog.shape[0], 2))
        cb[:, 0] = pred - c*sigma
        cb[:, 1] = pred + c*sigma

    else:
        cb = t_test.conf_int(alpha=1-cvrg_prob)

    if not linear:
        link = result.family.link
        pred = link.inverse(pred)
        cb = link.inverse(cb)

    return pred, cb, fvals

predict_functional_glm.__doc__ = predict_functional.__doc__.replace("predict_focus", "predict_focus_glm")
