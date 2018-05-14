from __future__ import absolute_import

from distutils.version import LooseVersion

import numpy as np
import pandas


version = LooseVersion(pandas.__version__)

if version >= LooseVersion('0.17.0'):
    def sort_values(df, *args, **kwargs):
        return df.sort_values(*args, **kwargs)
else:
    def sort_values(df, *args, **kwargs):
        # always set inplace with 'False' as default
        kwargs.setdefault('inplace', False)
        return df.sort(*args, **kwargs)

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

if version >= '0.20':
    from pandas.tseries import frequencies
    data_klasses = (pandas.Series, pandas.DataFrame, pandas.Panel)
else:
    try:
        import pandas.tseries.frequencies as frequencies
    except ImportError:
        from pandas.core import datetools as frequencies  # noqa

    data_klasses = (pandas.Series, pandas.DataFrame, pandas.Panel,
                    pandas.WidePanel)
