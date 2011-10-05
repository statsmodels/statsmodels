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
    endog = array(data[names[0]], dtype=dtype)
    endog_name = names[0]

    exog = column_stack(data[i] for i in names[1:])
    if dtype:
        exog = exog.astype(float)

    exog_name = names[1:]
    dataset = Dataset(data=data, names=names, endog=endog, exog=exog,
                      endog_name=endog_name, exog_name=exog_name)

    return dataset

def process_recarray_pandas(data, endog_idx=0, dtype=None):
    names = list(data.dtype.names)
    endog_name = names[endog_idx]
    exog = DataFrame(data, dtype=dtype)
    endog = exog.pop(endog_name)
    exog_name = list(exog.columns)
    dataset = Dataset(data=data, names=names, endog=endog, exog=exog,
                      endog_name = endog_name, exog_name=exog_name)
    return dataset
