from statsmodels.tools.data import _is_using_pandas

def _maybe_get_pandas_wrapper(X, trim=None):
    """
    If using pandas returns a function to wrap the results, e.g., wrapper(X)
    trim is an integer for the symmetric truncation of the series in some
    filters.
    otherwise returns None
    """
    if _is_using_pandas(X, None):
        index = X.index
        if trim is not None:
            index = X.index[trim:-trim]
        if hasattr(X, "columns"):
            return lambda x : X.__class__(x, index=index, columns=X.columns)
        else:
            return lambda x : X.__class__(x, index=index, name=X.name)
    else:
        return
