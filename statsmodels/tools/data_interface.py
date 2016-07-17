import pandas as pd
import numpy as np
from functools import partial
from patsy import dmatrix
from patsy.design_info import DesignMatrix

NUMPY_TYPES = [np.ndarray, np.float64]
PANDAS_TYPES = [pd.Series, pd.DataFrame]

class DataInterface(object):

    def __init__(self, permitted_types, internal_type=np.ndarray, data=None, external_type=None, model=None,
                 use_formula=False):

        self.permitted_types = permitted_types
        self.internal_type = internal_type
        self.columns = getattr(data, 'columns', None)
        self.name = getattr(data, 'name', None)
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
             return self.to_numpy_array(data)

        else:
            raise TypeError('Type conversion to {} from {} is not possible.'.format(self.internal_type, type(data)))

    def from_statsmodels(self, data):

        from_type = type(data)

        if from_type == DesignMatrix:
            if self.model is None:
                raise ValueError('When a DesignMatrix is returned, a model must be specified.')
            else:
                data = dmatrix(self.model.data.design_info.builder, data)
                from_type = type(data)

        if data is None:
            return None

        elif from_type in NUMPY_TYPES:
            return self.from_numpy_array(data)

        elif from_type in PANDAS_TYPES:
            return self.from_pandas(data)

        else:
            raise TypeError('Type conversion from {} to {} is not possible.'.format(from_type, self.external_type))

    def to_numpy_array(self, data):

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


    def to_pandas(self, data):

        from_type = type(data)

        if from_type in PANDAS_TYPES:
            return data

        else:
            np_data = self.to_numpy_array(data)

            if np_data.ndim == 1:
                return pd.Series(np_data, name=self.name)
            else:
                return pd.DataFrame(np_data, columns=self.columns)

    def from_numpy_array(self, data):

        from_type = type(data)

        if from_type == self.external_type:
            return data

        elif self.external_type == list:
            return data.tolist()

        elif self.external_type == np.recarray:
            return data.view(np.recarray)

        elif self.external_type == pd.Series:
            index = getattr(data, 'index', None)
            return pd.Series(data=data, index=index, dtype=self.dtype)

        elif self.external_type == pd.DataFrame:
            ndim = getattr(data, 'ndim', None)

            if ndim in [1, None]:
                index = getattr(data, 'index', None)

                return pd.Series(data=data, index=index)
            elif self.ndim == ndim:
                return pd.DataFrame(data=data, columns=self.columns, dtype=self.dtype)
            else:
                return pd.DataFrame(data=data, dtype=self.dtype)

        else:
            return data

    def from_pandas(self, data):

        from_type = type(data)

        if from_type == self.external_type:
            return data

        elif self.external_type == np.ndarray:
            return  data.values

        elif from_type == pd.Series and self.external_type == pd.DataFrame:
            return data.to_frame()

        elif from_type == pd.DataFrame and self.external_type == pd.Series:
            if data.ndim == 1:
                return pd.Series(data.values, index=data.index, name=data.columns[0])
            else:
                raise TypeError('Cannot convert multi dimentional DataFrame to a Series')

        elif self.external_type == list:
            return data.values.tolist()


NumPyInterface = partial(DataInterface, [np.ndarray])
PandasInterface = partial(DataInterface, [np.ndarray, pd.Series, pd.DataFrame])

def get_dtype(data):

    if hasattr(data, 'dtype'):
        return data.dtype
    elif hasattr(data, 'dtypes'):
        return data.dtypes
    else:
        return None

def is_1d(data):

    try:
        if np.asarray(data).ndim > 1 and np.asarray(data).squeeze().ndim == 1:
            return False
        elif np.asarray(data).ndim == 1:
            return True
    except:
        pass

    try:
        if data.ndim > 1:
            return False
        elif data.ndim == 1:
            return True
    except:
        pass

    raise ValueError('Cannot determine number of dimensions')
