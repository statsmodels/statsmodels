import statsmodels.tools.data as data_util

from charlton.spec import ModelSpec
from charlton.model_matrix import ModelMatrixColumnInfo
from numpy import c_ as concat

def handle_formula_data(endog, exog, formula):
    # I think we can rely on endog-only models not using a formula
    if data_util._is_using_pandas(endog, exog):
        (endog, exog,
         model_spec) = handle_formula_data_pandas(endog, exog, formula)
    else: # just assume ndarrays for now
        (matrices,
         model_spec) = handle_formula_data_ndarray(endog, exog, formula)
    #model_spec = ModelSpec.from_desc_and_data(formula, df)
    #matrices = model_spec.make_matrices(df)
    return endog, exog, model_spec

def handle_formula_data_pandas(endog, exog, formula):
    from pandas import Series, DataFrame
    #NOTE: assumes exog is a DataFrame which might not be the case
    # not important probably because this won't be the API
    df = exog.join(endog)
    df.column_info = ModelMatrixColumnInfo(df.columns.tolist())
    model_spec = ModelSpec.from_desc_and_data(formula, df)
    matrices = model_spec.make_matrices(df)
    endog, exog = matrices
    # preserve the meta-information from Charlton but pass back pandas
    # charlton should just let these types pass through as-is?
    endog_ci = endog.column_info
    #NOTE: univariate endog only right now
    endog = Series(endog.squeeze(), index=df.index, name=endog_ci.column_names)
    endog.column_info = endog_ci
    exog_ci = exog.column_info
    exog = DataFrame(exog, index=df.index, columns=exog_ci.column_names)
    exog.column_info = exog_ci
    return endog, exog, model_spec


def handle_formula_data_ndarray(endog, exog, formula):
    df = concat[endog, exog]
    nvars = df.shape[1]
    #NOTE: will be overwritten later anyway
    #TODO: make this duplication unnecessary
    names = ['x%d'] * nvars % map(str, range(nvars))
    df.column_info = ModelMatrixColumnInfo(names)
    model_spec = ModelSpec.from_desc_and_data(formula, df)
    matrices = model_spec.make_matrices(df)
    endog, exog = matrices
    return endog, exog, model_spec
