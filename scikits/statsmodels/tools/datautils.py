import os
import time
import numpy as np
from numpy import genfromtxt, array

from pandas import DataFrame

class Dataset(dict):
    def __init__(self, **kw):
        dict.__init__(self,kw)
        self.__dict__ = self
# Some datasets have string variables. If you want a raw_data attribute you
# must create this in the dataset's load function.
        try: # some datasets have string variables
            self.raw_data = self.data.view((float, len(self.names)))
        except:
            pass

    def __repr__(self):
        return str(self.__class__)

def process_recarray(data, endog_idx=0, dtype=None):
    names = list(data.dtype.names)

    if isinstance(endog_idx, int):
        endog = array(data[names[endog_idx]], dtype=dtype)
        endog_name = names[endog_idx]
        endog_idx = [endog_idx]
    else:
        endog_name = [names[i] for i in endog_idx]
        endog = np.column_stack(data[field] for field in endog_name)

    exog_name = [names[i] for i in xrange(len(names))
                 if i not in endog_idx]

    exog = np.column_stack(data[field] for field in exog_name)

    if dtype:
        endog = endog.astype(dtype)
        exog = exog.astype(dtype)

    dataset = Dataset(data=data, names=names, endog=endog, exog=exog,
                      endog_name=endog_name, exog_name=exog_name)

    return dataset

def process_recarray_pandas(data, endog_idx=0, dtype=None):
    exog = DataFrame(data, dtype=dtype)
    names = list(exog.columns)

    if isinstance(endog_idx, int):
        endog_name = names[endog_idx]
        endog = exog.pop(endog_name)
    else:
        endog = exog.ix[:, endog_idx]
        endog_name = list(endog.columns)
        exog = exog.drop(endog_name, axis=1)

    exog_name = list(exog.columns)
    dataset = Dataset(data=data, names=names, endog=endog, exog=exog,
                      endog_name = endog_name, exog_name=exog_name)
    return dataset
