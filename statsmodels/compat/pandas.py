from __future__ import absolute_import
import pandas
from distutils.version import LooseVersion

version = LooseVersion(pandas.__version__)

if version >= '0.17.0':
    def sort_values(df, *args, **kwargs):
        return df.sort_values(*args, **kwargs)
elif version >= '0.14.0':
    def sort_values(df, *args, **kwargs):
        kwargs.setdefault('inplace', False)  # always set inplace with 'False' as default
        return df.sort(*args, **kwargs)
else:  # Before that, sort didn't have 'inplace' for non data-frame
    def sort_values(df, *args, **kwargs):
        if isinstance(df, pandas.DataFrame):
            return df.sort(*args, **kwargs)
        # Just make sure inplace is 'False' by default, but doesn't appear in the final arguments
        # Here, setdefaults will ensure the del operation always succeeds
        inplace = kwargs.setdefault('inplace', False)
        del kwargs['inplace']
        if not inplace:
            df = df.copy()
        df.sort(*args, **kwargs)
        return df
