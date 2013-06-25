import sys
import shutil
import pickle
from os import environ
from os import makedirs
from os.path import basename
from os.path import expanduser
from os.path import exists
from os.path import expanduser
from os.path import join
from StringIO import StringIO
import time
from urllib2 import urlopen, HTTPError

import numpy as np
from numpy import genfromtxt, array
from pandas import read_csv


class Dataset(dict):
    def __init__(self, **kw):
        # define some default attributes, so pylint can find them
        self.endog = None
        self.exog = None
        self.data = None
        self.names = None

        dict.__init__(self,kw)
        self.__dict__ = self
        # Some datasets have string variables. If you want a raw_data
        # attribute you must create this in the dataset's load function.
        try: # some datasets have string variables
            self.raw_data = self.data.view((float, len(self.names)))
        except:
            pass

    def __repr__(self):
        return str(self.__class__)

def process_recarray(data, endog_idx=0, exog_idx=None, stack=True, dtype=None):
    names = list(data.dtype.names)

    if isinstance(endog_idx, int):
        endog = array(data[names[endog_idx]], dtype=dtype)
        endog_name = names[endog_idx]
        endog_idx = [endog_idx]
    else:
        endog_name = [names[i] for i in endog_idx]

        if stack:
            endog = np.column_stack(data[field] for field in endog_name)
        else:
            endog = data[endog_name]

    if exog_idx is None:
        exog_name = [names[i] for i in xrange(len(names))
                 if i not in endog_idx]
    else:
        exog_name = [names[i] for i in exog_idx]

    if stack:
        exog = np.column_stack(data[field] for field in exog_name)
    else:
        exog = data[exog_name]

    if dtype:
        endog = endog.astype(dtype)
        exog = exog.astype(dtype)

    dataset = Dataset(data=data, names=names, endog=endog, exog=exog,
                      endog_name=endog_name, exog_name=exog_name)

    return dataset

def process_recarray_pandas(data, endog_idx=0, exog_idx=None, dtype=None,
                            index_idx=None):
    from pandas import DataFrame

    data = DataFrame(data, dtype=dtype)
    names = data.columns

    if isinstance(endog_idx, int):
        endog_name = names[endog_idx]
        endog = data[endog_name]
        if exog_idx is None:
            exog = data.drop([endog_name], axis=1)
        else:
            exog = data.filter(names[exog_idx])
    else:
        endog = data.ix[:, endog_idx]
        endog_name = list(endog.columns)
        if exog_idx is None:
            exog = data.drop(endog_name, axis=1)
        elif isinstance(exog_idx, int):
            exog = data.filter([names[exog_idx]])
        else:
            exog = data.filter(names[exog_idx])

    if index_idx is not None: #NOTE: will have to be improved for dates
        from pandas import Index
        endog.index = Index(data.ix[:, index_idx])
        exog.index = Index(data.ix[:, index_idx])
        data = data.set_index(names[index_idx])

    exog_name = list(exog.columns)
    dataset = Dataset(data=data, names=list(names), endog=endog, exog=exog,
                      endog_name=endog_name, exog_name=exog_name)
    return dataset

def _maybe_reset_index(data):
    """
    All the Rdatasets have the integer row.labels from R if there is no
    real index. Strip this for a zero-based index
    """
    from pandas import Index
    if data.index.equals(Index(range(1,len(data)+1))):
        data = data.reset_index(drop=True)
    return data

def _get_cache(cache):
    if cache is False:
        # do not do any caching or load from cache
        cache = None
    elif cache is True: # use default dir for cache
        cache = get_data_home(None)
    else:
        cache = get_data_home(cache)
    return cache

def _cache_it(data, cache_path):
    if sys.version_info[0] >= 3:
        # for some reason encode("zip") won't work for me in Python 3?
        import zlib
        open(cache_path, "wb").write(zlib.compress(pickle.dumps(data)))
    else:
        open(cache_path, "wb").write(pickle.dumps(data).encode("zip"))

def _open_cache(cache_path):
    if sys.version_info[0] >= 3:
        #NOTE: don't know why but decode('zip') doesn't work on my
        # Python 3 build
        import zlib
        data = zlib.decompress(open(cache_path, 'rb').read())
        data = pickle.loads(data)
    else:
        data = open(cache_path, 'rb').read().decode('zip')
        data = pickle.loads(data)
    return data

def _urlopen_cached(url, cache):
    """
    Tries to load data from cache location otherwise downloads it. If it
    downloads the data and cache is not None then it will put the downloaded
    data in the cache path.
    """
    from_cache = False
    if cache is not None:
        cache_path = join(cache,
                          url.split("://")[-1].replace('/', ',') +".zip")
        try:
            data = _open_cache(cache_path)
            from_cache = True
        except:
            pass

    # not using the cache or didn't find it in cache
    if not from_cache:
        data = urlopen(url).read()
        if cache is not None: # then put it in the cache
            _cache_it(data, cache_path)
    return data, from_cache


def _get_data(base_url, dataname, cache, extension="csv"):
    url = base_url + (dataname + ".%s") % extension
    try:
        data, from_cache = _urlopen_cached(url, cache)
    except HTTPError, err:
        if '404' in str(err):
            raise ValueError("Dataset %s was not found." % dataname)
        else:
            raise err

    #Python 3, don't think there will be any unicode in r datasets
    if sys.version[0] == '3':  # pragma: no cover
        data = data.decode('ascii', errors='strict')
    return StringIO(data), from_cache


def _get_dataset_meta(dataname, package, cache):
    # get the index, you'll probably want this cached because you have
    # to download info about all the data to get info about any of the data...
    index_url = ("https://raw.github.com/vincentarelbundock/Rdatasets/master/"
                 "datasets.csv")
    data, _ = _urlopen_cached(index_url, cache)
    #Python 3
    if sys.version[0] == '3':  # pragma: no cover
        data = data.decode('ascii', errors='strict')
    index = read_csv(StringIO(data))
    idx = np.logical_and(index.Item == dataname, index.Package == package)
    dataset_meta = index.ix[idx]
    return dataset_meta["Title"].item()

def get_rdataset(dataname, package="datasets", cache=False):
    """
    Parameters
    ----------
    dataname : str
        The name of the dataset you want to download
    package : str
        The package in which the dataset is found. The default is the core
        'datasets' package.
    cache : bool or str
        If True, will download this data into the STATSMODELS_DATA folder.
        The default location is a folder called statsmodels_data in the
        user home folder. Otherwise, you can specify a path to a folder to
        use for caching the data. If False, the data will not be cached.

    Returns
    -------
    dataset : Dataset instance
        A `statsmodels.data.utils.Dataset` instance. This objects has
        attributes::

        * data - A pandas DataFrame containing the data
        * title - The dataset title
        * package - The package from which the data came
        * from_cache - Whether not cached data was retrieved
        * __doc__ - The verbatim R documentation.


    Notes
    -----
    If the R dataset has an integer index. This is reset to be zero-based.
    Otherwise the index is preserved. The caching facilities are dumb. That
    is, no download dates, e-tags, or otherwise identifying information
    is checked to see if the data should be downloaded again or not. If the
    dataset is in the cache, it's used.
    """
    #NOTE: use raw github bc html site might not be most up to date
    data_base_url = ("https://raw.github.com/vincentarelbundock/Rdatasets/"
                     "master/csv/"+package+"/")
    docs_base_url = ("https://raw.github.com/vincentarelbundock/Rdatasets/"
                     "master/doc/"+package+"/rst/")
    cache = _get_cache(cache)
    data, from_cache = _get_data(data_base_url, dataname, cache)
    data = read_csv(data, index_col=0)
    data = _maybe_reset_index(data)

    title = _get_dataset_meta(dataname, package, cache)
    doc, _ = _get_data(docs_base_url, dataname, cache, "rst")

    return Dataset(data=data, __doc__=doc.read(), package=package, title=title,
                   from_cache=from_cache)

### The below function were taken from sklearn

def get_data_home(data_home=None):
    """Return the path of the statsmodels data dir.

    This folder is used by some large dataset loaders to avoid
    downloading the data several times.

    By default the data dir is set to a folder named 'statsmodels_data'
    in the user home folder.

    Alternatively, it can be set by the 'STATSMODELS_DATA' environment
    variable or programatically by giving an explit folder path. The
    '~' symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.
    """
    if data_home is None:
        data_home = environ.get('STATSMODELS_DATA',
                               join('~', 'statsmodels_data'))
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home


def clear_data_home(data_home=None):
    """Delete all the content of the data home cache."""
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)
