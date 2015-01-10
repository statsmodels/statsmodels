import statsmodels.api as sm
import pandas as pd
import patsy
import numpy as np

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


def predict_functional(result, focus_var, summaries, values, num_points=10, **kwargs):
    """
    Returns predictions of a fitted model relative to a given 'focus variable'.

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
    num_points : integer
        The number of rows of the resulting data frame.
    kwargs :
        Arguments such as `linear` passed to the `predict` method.

    Returns
    -------
    pred : array-like
        The predicted mean values.
    ci : array-like
        An array with two columns, containing respectively the lower and upper
        confidence limit.
    fvals : array-like
        The values of the focus variable at which the prediction is made.

    Notes
    -----
    All variables in the model except for the focus variable should be
    included as a key in either `summaries` or `values`.

    These are conditional means, based on specified values of the
    non-focus variables.  They are not marginal (unconditional) means.

    Example
    -------
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
    >>> pr = predict_focus(result, 'age', summaries, values)

    Fit a model using arrays.  Plot the means as a function of x3,
    holding x1 fixed at its mean value in the data used to fit the
    model, and holding x2 fixed at 1.

    >>> model = sm.OLS(y ,x)
    >>> result = model.fit()
    >>> summaries = {'x1': np.mean}
    >>> values = {'x2': 1}
    >>> pr = predict_focus(result, 'x3', summaries, values)
    """

    ky = set(values.keys()) & set(summaries.keys())
    if len(ky) > 0:
        raise ValueError("%s included in both `values` and `summaries`" %
                         ", ".join(ky))

    if hasattr(result.model.data, "frame"):
        dexog, fexog, fvals = _make_formula_exog(result, focus_var, summaries, values, num_points)
    else:
        exog, fvals = _make_exog(result, focus_var, summaries, values, num_points)
        dexog, fexog = exog, exog

    pred = result.predict(exog=fexog, **kwargs)
    t_test = result.t_test(dexog)
    ci = t_test.conf_int()

    return pred, ci, fvals

def predict_functional_glm(result, focus_var, summaries, values, num_points=10, **kwargs):

    if not hasattr(result.model, "family"):
        raise ValueError("result must be a fitted GLM or GEE results instance")

    pred, ci, fvals = predict_functional(result, focus_var, summaries, values, num_points=10, linear=True)

    link = result.family.link
    pred = link.inverse(pred)
    ci = link.inverse(ci)

    return pred, ci, fvals

predict_functional_glm.__doc__ = predict_functional.__doc__.replace("predict_focus", "predict_focus_glm")
