from __future__ import absolute_import

from distutils.version import LooseVersion

import pandas

__all__ = ['assert_frame_equal', 'assert_index_equal', 'assert_series_equal',
           'data_klasses', 'frequencies', 'is_numeric_dtype', 'testing',
           'cache_readonly']

version = LooseVersion(pandas.__version__)
pandas_lt_25_0 = version < LooseVersion('0.25.0')

try:
    from pandas.api.types import is_numeric_dtype
except ImportError:
    from pandas.core.common import is_numeric_dtype

try:
    from pandas.tseries import offsets as frequencies
except ImportError:
    from pandas.tseries import frequencies

data_klasses = (pandas.Series, pandas.DataFrame)
if pandas_lt_25_0:
    data_klasses += (pandas.Panel,)

if version >= '0.20':
    from pandas.util._decorators import cache_readonly
else:
    from pandas.util.decorators import cache_readonly

try:
    import pandas.testing as testing
except ImportError:
    import pandas.util.testing as testing

assert_frame_equal = testing.assert_frame_equal
assert_index_equal = testing.assert_index_equal
assert_series_equal = testing.assert_series_equal
