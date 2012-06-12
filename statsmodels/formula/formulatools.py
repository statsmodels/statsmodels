import statsmodels.tools.data as data_util

from charlton import design_and_matrices, DesignMatrixColumnInfo
from charlton.desc import INTERCEPT
from numpy import array, argsort, zeros, c_ as concat


# if users want to pass in a different formula framework, they can
# add their handler here. how to do it interactively?

# this is a mutable object, so editing it should show up in the
# below
formula_handler = {}

def handle_formula_data(Y, X, formula):
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
         model_spec) = handle_formula_data_pandas(Y, X, formula)
    else: # just assume ndarrays for now
        (endog, exog,
         model_spec) = handle_formula_data_ndarray(Y, X, formula)
    return endog, exog, model_spec

def handle_formula_data_pandas(Y, X, formula):
    from pandas import Series, DataFrame
    #NOTE: assumes exog is a DataFrame which might not be the case
    # not important probably because this won't be the API
    if X is not None:
        df = X.join(Y)
    else:
        df = Y
    # eval_env=1 means resolve formula in caller's namespace
    model_spec, endog, exog = design_and_matrices(formula, df, eval_env=1)
    # preserve the meta-information from Charlton but pass back pandas
    # charlton should just let these types pass through as-is...
    endog_ci = endog.column_info
    #NOTE: univariate endog only right now
    endog = Series(endog.squeeze(), index=df.index, name=endog_ci.column_names)
    endog.column_info = endog_ci
    exog_ci = exog.column_info
    exog = DataFrame(exog, index=df.index, columns=exog_ci.column_names)
    exog.column_info = exog_ci
    return endog, exog, model_spec


def handle_formula_data_ndarray(Y, X, formula):
    if X is not None:
        df = concat[Y, X]
    else:
        df = Y
    nvars = df.shape[1]
    #NOTE: if only Y is given and it isn't a structured array there's no
    # way to specify a formula, ie., we have to assume Y contains y and X
    # contains x1, x2, x3, etc. if they're given as arrays

    #NOTE: will be overwritten later anyway
    #TODO: make this duplication unnecessary
    names = ['x%d'] * nvars % map(str, range(nvars))
    df.column_info = DesignMatrixColumnInfo(names)
    model_spec, endog, exog = design_and_matrices(formula, df, eval_env=1)
    #endog, exog = model_spec.make_matrices(df)
    return endog, exog, model_spec

def _remove_intercept_charlton(terms):
    """
    Remove intercept from Charlton terms.
    """
    if INTERCEPT in terms:
        del terms[INTERCEPT]
    return terms

def _assign(column_info, intercept=False):
    """
    column_info is the ModelMatrixColumnInfo object. If intercept is True, it
    is kept in the returned results. Otherwise, it is removed.

    This returns an array assign and a list index. assign is a mapping from
    the column of a design matrix to the term in index
    """
    k_vars = len(column_info.column_names)
    col_order = []
    index = []
    terms_info = column_info.term_to_columns
    for term, cols in terms_info.iteritems():
        col_order.append(cols[0])
        index.append(term.name())

    index = array(index)[argsort(col_order)].tolist()
    assign = zeros(k_vars, dtype=int)
    for term, cols in terms_info.iteritems():
        assign[cols[0]:cols[1]] = index.index(term.name())

    if '1' in index and not intercept:
        assign = assign[~(assign == index.index('1'))]
        index.remove('1')

    return assign, index

