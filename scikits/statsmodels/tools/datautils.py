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

def process_recarray(data, endog_idx=0, stack=True, dtype=None):
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

    exog_name = [names[i] for i in xrange(len(names))
                 if i not in endog_idx]

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

def process_recarray_pandas(data, endog_idx=0, dtype=None):
    data = DataFrame(data, dtype=dtype)
    names = list(data.columns)

    if isinstance(endog_idx, int):
        endog_name = names[endog_idx]
        endog = data[endog_name]
        exog = data.drop([endog_name], axis=1)
    else:
        endog = data.ix[:, endog_idx]
        endog_name = list(endog.columns)
        exog = data.drop(endog_name, axis=1)

    exog_name = list(exog.columns)
    dataset = Dataset(data=data, names=names, endog=endog, exog=exog,
                      endog_name = endog_name, exog_name=exog_name)
    return dataset
