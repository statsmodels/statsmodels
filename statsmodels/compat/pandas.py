from __future__ import absolute_import

from distutils.version import LooseVersion

import numpy as np
import pandas


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

try:
    from pandas import RangeIndex
except ImportError:
    RangeIndex = tuple()

# Float64Index introduced in Pandas 0.13.0
try:
    from pandas import Float64Index
except:
    Float64Index = tuple()

try:
    from pandas.api.types import is_numeric_dtype
except ImportError:
    try:
        from pandas.core.common import is_numeric_dtype
    except ImportError:
        # Pandas <= 0.14
        def is_numeric_dtype(arr_or_dtype):
            # Crude implementation only suitable for array-like types
            try:
                tipo = arr_or_dtype.dtype.type
            except AttributeError:
                tipo = type(None)
            return (issubclass(tipo, (np.number, np.bool_)) and
                    not issubclass(tipo, (np.datetime64, np.timedelta64)))

try:
    import pandas.tseries.tools as datetools
    import pandas.tseries.frequencies as frequencies
except ImportError:
    from pandas.core import datetools
    frequencies = datetools

