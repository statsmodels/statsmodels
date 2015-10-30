import pandas
from distutils.version import LooseVersion

version = LooseVersion(pandas.__version__)

if version >= '0.17.0':
    def sort_values(df, *args, **kwargs):
        return df.sort_values(*args, **kwargs)
else:
    def sort_values(df, *args, **kwargs):
        return df.sort(*args, **kwargs)
