import shutil
from os import environ
from os import makedirs
from os.path import basename
from os.path import expanduser
from os.path import exists
from os.path import expanduser
from os.path import join
from StringIO import StringIO
import time
import httplib2

import numpy as np
from numpy import genfromtxt, array
from pandas import read_csv


class Dataset(dict):
    def __init__(self, **kw):
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

def _get_rdatasets_name(dataname):
    """
    Grabs a list of all csv files and tries to match versus one in a case
    insensitive way. Could be extended to do fuzzy matching
    """
    from urllib2 import urlopen
    from HTMLParser import HTMLParser
    base_url = ("https://github.com/vincentarelbundock/Rdatasets/tree/"
                "master/csv")
    html = urlopen(base_url).read()
    class MyHTMLParser(HTMLParser):
        result = None
        def handle_starttag(self, tag, attrs):
            if tag == "a":
                for name, value in attrs:
                    if name == "href" and value.endswith('.csv'):
                        #from IPython.core.debugger import Pdb; Pdb().set_trace()
                        base = basename(value)
                        if base.lower() == dataname + ".csv":
                            self.result = base[:-4]
                            raise StopIteration

    parser = MyHTMLParser()
    try:
        parser.feed(html)
    except StopIteration:
        pass
    result = parser.result
    return result

def _open_w_404_handling(connection, base_url, dataname, extension):
    url = base_url + ("%s." + extension) % dataname
    response, data = connection.request(url)
    if response.status == 404:
        # try a little harder to find the dataset
        new_dataname = _get_rdatasets_name(dataname)
        if new_dataname:
            url = base_url + ("%s." + extension) % new_dataname
            response, data = connection.request(url)
        else:
            raise ValueError("Dataset %s was not found." % dataname)
        if response.status == 404:
            raise ValueError("Dataset %s was not found." % dataname)

    # python 3 compatibility
    import sys
    if sys.version[0] == '3':  # pragma: no cover
        data = data.decode('ascii', errors='strict')
    return StringIO(data), response.fromcache


def _get_cache(cache):
    if cache is False:
        # do not do any caching or load from cache
        cache = None
    elif cache is True: # use default dir for cache
        cache = get_data_home(None)
    else:
        cache = get_data_home(cache)
    return cache


def _get_data(base_url, dataname, cache, extension="csv"):
    connection = httplib2.Http(cache)
    data, from_cache = _open_w_404_handling(connection, base_url, dataname,
                                           extension)
    return data, from_cache

def _get_data_doc(base_url, dataname, cache):
    #TODO: will need to include parsed HTML docs in repo first
    data, _ = _get_data(base_url, dataname, cache, "rst")
    # return it

def _get_dataset_meta(dataname, cache):
    # get the index, you'll probably want this cached because you have
    # to download info about all the data to get info about any of the data...
    index_url = ("https://raw.github.com/vincentarelbundock/Rdatasets/master/"
                 "datasets.csv")
    connection = httplib2.Http(cache)
    response, data = connection.request(index_url)
    index = read_csv(StringIO(data))
    idx = index.Dataset.str.lower() == dataname.lower()
    dataset_meta = index.ix[idx]
    return (dataset_meta["R Package"].item(),
            dataset_meta["Description"].item())

def get_rdataset(dataname, cache=False):
    """
    Parameters
    ----------
    dataname : str
        The name of the dataset you want to download
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
        * descr - A brief description of the data
        * package - The package the data came from
        * from_cache - Whether not cached data was retrieved


    Notes
    -----
    If the R dataset has an integer index. This is reset to be zero-based.
    Otherwise the index is preserved. While the function will do its best not
    to be case-sensitive. If you want to use the caching facilities, you need
    to give the case-sensitive name of the dataset.
    """
    #NOTE: use raw github bc html site might not be most up to date
    data_base_url = ("https://raw.github.com/vincentarelbundock/Rdatasets/"
                     "master/csv/")
    docs_base_url = ("https://raw.github.com/vincentarelbundock/Rdatasets/"
                     "master/doc/")
    cache = _get_cache(cache)
    data, from_cache = _get_data(data_base_url, dataname, cache)
    data = read_csv(data, index_col=0)
    data = _maybe_reset_index(data)

    package, descr = _get_dataset_meta(dataname, cache)

    #doc = _get_data_doc(docs_base_url, dataname, cache)
    return Dataset(data=data, __doc__=None, package=package, descr=descr,
                   from_cache=from_cache)

### The below function were taken from sklearn

def get_data_home(data_home=None):
    """Return the path of the scikit-learn data dir.

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
