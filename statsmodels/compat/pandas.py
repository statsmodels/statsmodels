from __future__ import absolute_import

from distutils.version import LooseVersion

import pandas as pd


version = LooseVersion(pd.__version__)


try:
    from pandas.api.types import is_numeric_dtype  # noqa:F401
except ImportError:
    from pandas.core.common import is_numeric_dtype  # noqa:F401

if version >= '0.20':
    from pandas.tseries import frequencies
    data_klasses = (pd.Series, pd.DataFrame, pd.Panel)
else:
    try:
        import pandas.tseries.frequencies as frequencies
    except ImportError:
        from pandas.core import datetools as frequencies  # noqa

    data_klasses = (pd.Series, pd.DataFrame, pd.Panel, pd.WidePanel)
