import statsmodels.tools.data as data_util

from charlton import design_and_matrices, DesignMatrixColumnInfo
from charlton.desc import INTERCEPT
from numpy import array, argsort, zeros, dtype, c_ as concat
from numpy.lib.recfunctions import append_fields


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
    formula : str or charlton.model_desc
        You can pass a handler by import formula_handler and adding a
        key-value pair where the key is the formula object class and
        the value is a function that returns endog, exog, formula object

    Returns
    -------
    endog : array-like
        Should preserve the input type of Y,X
    exog : array-like
        Should preserve the input type of Y,X. Could be None.
    formula : ModelSpec
        In the default case this is a model specification from Charlton.

    Notes
    -----
    It is possible to override the signature and pass a whole array /
    data frame to Y.
    """
    # half ass attempt to handle other formula objects
    if isinstance(formula, tuple(formula_handler.keys())):
        return formula_handler[type(formula)]

    # I think we can rely on endog-only models not using a formula
    if data_util._is_using_pandas(Y, X):
        (endog, exog,
         model_spec) = handle_formula_data_pandas(Y, X, formula, depth)
    elif isinstance(Y, dict) and (X is None or
                                  (X is not None and isinstance(X, dict))):
        (endog, exog,
         model_spec) = handle_formula_dict(Y, X, formula, depth)
    else: # just assume ndarrays for now, support other objects as needed
        if Y.dtype.names is not None:
            (endog, exog,
            model_spec) = handle_formula_data_recarray(Y, X, formula, depth)
        else: # pragma : no cover
            # this actually won't ever be called for from_formula
            (endog, exog,
             model_spec) = handle_formula_data_ndarray(Y, X, formula, depth)
    return endog, exog, model_spec

def handle_formula_data_pandas(Y, X, formula, depth):
    from pandas import Series, DataFrame
    #NOTE: assumes exog is a DataFrame which might not be the case
    # not important probably because this won't be the API
    if X is not None:
        df = X.join(Y)
    else:
        df = Y
    # eval_env=1 means resolve formula in caller's namespace
    model_spec, endog, exog = design_and_matrices(formula, df, eval_env=3)
    # preserve the meta-information from Charlton but pass back pandas
    # charlton should just let these types pass through as-is...
    endog_ci = endog.column_info
    #NOTE: univariate endog only right now
    endog = Series(endog.squeeze(), index=df.index,
                           name=endog_ci.column_names[0])
    endog.column_info = endog_ci
    exog_ci = exog.column_info
    exog = DataFrame(exog, index=df.index, columns=exog_ci.column_names)
    exog.column_info = exog_ci
    return endog, exog, model_spec

def handle_formula_data_recarray(Y, X, formula, depth):
    """
    Notes
    -----
    This returns a charlton.DesignMatrix, but they're caught at the model level
    """
    if X is not None:
        df = append_fields(Y, X.dtype.names,
                           X.view((float, len(X.dtype.names))).T, usemask=False)
    else:
        df = Y
    nvars = len(df.dtype.names)
    model_spec, endog, exog = design_and_matrices(formula, df, eval_env=depth+3)
    return endog, exog, model_spec

def handle_formula_data_ndarray(Y, X, formula, depth): # pragma : no cover
    raise NotImplementedError("You must use a data structure that defines "
                              "__getitem__ for the names in the formula")
    #if X is not None:
    #    df = concat[Y, X]
    #else:
    #    df = Y
    #nvars = df.shape[1]
    ##NOTE: if only Y is given and it isn't a structured array there's no
    ## way to specify a formula, ie., we have to assume Y contains y and X
    ## contains x1, x2, x3, etc. if they're given as arrays
    #
    #model_spec, endog, exog = design_and_matrices(formula, df, eval_env=depth+3)
    #return endog, exog, model_spec

def handle_formula_dict(Y, X, formula, depth):
    if X is not None:
        try:
            overlap = set(Y).intersection(set(X))
            assert not len(overlap)
        except:
            raise ValueError("The keys of Y and X overlap: %s" % list(overlap))
        df = Y.update(X)
    else:
        df = Y

    model_spec, endog, exog = design_and_matrices(formula, df, eval_env=depth+3)
    return endog, exog, model_spec

def _remove_intercept_charlton(terms):
    """
    Remove intercept from Charlton terms.
    """
    if INTERCEPT in terms:
        terms.remove(INTERCEPT)
    return terms

def make_hypotheses_matrices(model_results, test_formula):
    """
    """
    from charlton.constraint import linear_constraint
    exog_names = model_results.model.exog_names
    LC = linear_constraint(test_formula, exog_names)
    return LC
