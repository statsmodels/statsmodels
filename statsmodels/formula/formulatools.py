import statsmodels.tools.data as data_util

from charlton.spec import ModelSpec
from charlton.model_matrix import ModelMatrixColumnInfo
from numpy import c_ as concat

def handle_formula_data(endog, exog, formula):
    #NOTE: I think we can rely on endog-only models not using a formula
    if data_util._is_using_pandas(endog, exog):
        #NOTE: assumes exog is a DataFrame which might not be the case
        # not important probably because this won't be the API
        df = exog.join(endog)
        df.column_info = ModelMatrixColumnInfo(df.columns.tolist())
    else: # just assume ndarrays for now
        df = concat[endog, exog]
        nvars = df.shape[1]
        #NOTE: will be overwritten later anyway
        #TODO: make this unnecessary
        names = ['x%d'] * nvars % map(str, range(nvars))
        df.column_info = ModelMatrixColumnInfo(names)
    model_spec = ModelSpec.from_desc_and_data(formula, df)
    matrices = model_spec.make_matrices(df)
    endog, exog = matrices
    return endog, exog, model_spec
