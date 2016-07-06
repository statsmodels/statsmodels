import pandas as pd
import numpy as np
from functools import partial
from patsy import dmatrix

NUMPY_TYPES = [np.ndarray, np.float64]

class DataInterface(object):

    def __init__(self, permitted_types, internal_type, data=None, external_type=None, model=None, use_formula=False):

        self.permitted_types = permitted_types
        self.internal_type = internal_type
        self.columns = getattr(data, 'columns', None)
        self.name = getattr(data, 'name', None)
        self.index = getattr(data, 'index', None)
        self.ndim = getattr(data, 'ndim', None)
        self.model = model
        self.use_formula = use_formula

        if external_type is not None:
            self.external_type = external_type
        elif data is not None:
            self.external_type = type(data)
        else:
            self.external_type = np.ndarray

        self.dtype = get_dtype(data)

    def to_statsmodels(self, data):

        if data is None:
            return None

        elif self.use_formula and self.model is not None and hasattr(self.model, 'formula'):
                return dmatrix(self.model.data.design_info.builder, data)

        elif type(data) in self.permitted_types:
            return data

        elif self.internal_type == np.ndarray:
             return self.to_numpy(data)

        else:
            raise TypeError('Type conversion to {} from {} is not possible.'.format(self.internal_type, type(data)))

    def from_statsmodels(self, data):

        internal_type = type(data)

        if internal_type in NUMPY_TYPES:
            return self.from_numpy(data)

        else:
            raise TypeError('Type conversion from {} to {} is not possible.'.format(internal_type, self.external_type))

    def to_numpy(self, data):

        from_type = type(data)

        if from_type in NUMPY_TYPES:
            to_return = data

        elif from_type == list:
            to_return = np.array(data)

        elif from_type == np.recarray:
            to_return = data.view(np.ndarray)

        elif from_type == pd.Series:

            to_return = data.values

        elif from_type == pd.DataFrame:
            to_return = data.values

        else:
            try:
                to_return = np.asarray(data)
            except:
                TypeError('Type conversion to numpy from {} is not possible.'.format(from_type))


        if self.model is not None:
            return clean_ndim(to_return, self.model)
        else:
            return to_return

    def from_numpy(self, data):

        if type(data) == self.external_type:
            return data

        elif self.external_type == list:
            return data.tolist()

        elif self.external_type == np.recarray:
            return data.view(np.recarray)

        elif self.external_type == pd.Series:
            return pd.Series(data=data, index=self.index, dtype=self.dtype)

        elif self.external_type == pd.DataFrame:
            ndim = getattr(data, 'ndim', None)

            if ndim in [1, None]:
                return pd.Series(data=data, index=self.index)
            elif self.ndim == ndim:
                return pd.DataFrame(data=data, columns=self.columns, dtype=self.dtype)
            else:
                return pd.DataFrame(data=data, dtype=self.dtype)

        else:
            return data


NumPyInterface = partial(DataInterface, [np.ndarray], np.ndarray)
PandasInterface = partial(DataInterface, [np.ndarray, pd.Series, pd.DataFrame], np.ndarray)

def clean_ndim(data, model):

    if data.ndim == 1 and (model.exog.ndim == 1 or model.exog.shape[1] == 1):
        data = data[:, None]

    data = np.atleast_2d(data)  # needed in count model shape[1]

    return data

def get_dtype(data):

    if hasattr(data, 'dtype'):
        return data.dtype
    elif hasattr(data, 'dtypes'):
        return data.dtypes
    else:
        return None
