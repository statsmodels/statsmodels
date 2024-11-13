from statsmodels.compat.pandas import PD_LT_2

import pandas as pd


def _safe_is_pandas_categorical_dtype(dt):
    if PD_LT_2:
        return pd.api.types.is_categorical_dtype(dt)
    return isinstance(dt, pd.CategoricalDtype)


def monkey_patch_cat_dtype():
    try:
        import patsy.util
        patsy.util.safe_is_pandas_categorical_dtype = (
            _safe_is_pandas_categorical_dtype
        )
    except ImportError:
        pass
