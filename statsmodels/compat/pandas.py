from __future__ import absolute_import

from distutils.version import LooseVersion

import pandas
from pandas import RangeIndex, Float64Index  # noqa:F401


version = LooseVersion(pandas.__version__)


def sort_values(df, *args, **kwargs):
    return df.sort_values(*args, **kwargs)


try:
    from pandas.api.types import is_numeric_dtype  # noqa:F401
except ImportError:
    from pandas.core.common import is_numeric_dtype  # noqa:F401

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
