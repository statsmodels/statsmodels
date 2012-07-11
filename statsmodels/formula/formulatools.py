import statsmodels.tools.data as data_util
from patsy import dmatrices

# if users want to pass in a different formula framework, they can
# add their handler here. how to do it interactively?

# this is a mutable object, so editing it should show up in the below
formula_handler = {}

def handle_formula_data(Y, X, formula, depth=0):
    """
    Returns endog, exog, and the model specification from arrays and formula

    Parameters
    ----------
    Y : array-like
        Either endog (the LHS) of a model specification or all of the data.
        Y must define __getitem__ for now.
    X : array-like
        Either exog or None. If all the data for the formula is provided in
        Y then you must explicitly set X to None.
    formula : str or patsy.model_desc
        You can pass a handler by import formula_handler and adding a
        key-value pair where the key is the formula object class and
        the value is a function that returns endog, exog, formula object

    Returns
    -------
    endog : array-like
        Should preserve the input type of Y,X
    exog : array-like
        Should preserve the input type of Y,X. Could be None.
    """
    # half ass attempt to handle other formula objects
    if isinstance(formula, tuple(formula_handler.keys())):
        return formula_handler[type(formula)]

    if X is not None:
        if data_util._is_using_pandas(Y, X):
            return dmatrices(formula, (Y, X), 2, return_type='dataframe')
        else:
            return dmatrices(formula, (Y, X), 2, return_type='dataframe')
    else:
        if data_util._is_using_pandas(Y, None):
            return dmatrices(formula, Y, 2, return_type='dataframe')
        else:
            return dmatrices(formula, Y, 2, return_type='dataframe')


def _remove_intercept_patsy(terms):
    """
    Remove intercept from Patsy terms.
    """
    from patsy.desc import INTERCEPT
    if INTERCEPT in terms:
        terms.remove(INTERCEPT)
    return terms

def _has_intercept(design_info):
    from patsy.desc import INTERCEPT
    return INTERCEPT in design_info.terms

def _intercept_idx(design_info):
    """
    Returns boolean array index indicating which column holds the intercept
    """
    from patsy.desc import INTERCEPT
    from numpy import array
    return array([INTERCEPT == i for i in design_info.terms])

def make_hypotheses_matrices(model_results, test_formula):
    """
    """
    from patsy.constraint import linear_constraint
    exog_names = model_results.model.exog_names
    LC = linear_constraint(test_formula, exog_names)
    return LC
