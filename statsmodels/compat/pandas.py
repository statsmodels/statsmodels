import pandas
from ..tools.testing import is_pandas_min_version

if is_pandas_min_version('0.17.0'):
    def sort_values(df, *args, **kwargs):
        return df.sort_values(*args, **kwargs)
else:
    def sort_values(df, *args, **kwargs):
        return df.sort(*args, **kwargs)
