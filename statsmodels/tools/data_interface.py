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
        self.model = model
        self.use_formula = use_formula

        self.columns = None
        self.name = None
        self.dtype = None
        self.index = None
        self.ndim = None
        self.is_col_vector = None

        if external_type is not None:
            self.external_type = external_type

        elif data is not None:
            self.external_type = type(data)

        else:
            self.external_type = np.ndarray

    def init_data_interface(self, data):

        self.columns = getattr(data, 'columns', None)
        self.name = getattr(data, 'name', None)
        self.dtype = get_dtypes(data)
        self.ndim = get_ndim(data)
        self.is_col_vector = is_col_vector(data)

    def to_transpose(self, data):

        data_ndim = get_ndim(data)
        data_col_vector = is_col_vector(data)

        if self.ndim == 1 and data_ndim == 1:
            if self.is_col_vector == data_col_vector:
                return False
            else:
                return True
        else:
            return False

    def to_statsmodels(self, data):

        if data is None:
            return None
        else:
            self.init_data_interface(data)

        if self.use_formula and self.model is not None and hasattr(self.model, 'formula'):
            return dmatrix(self.model.data.design_info.builder, data)

        elif type(data) in self.permitted_types:
            return data

        elif self.internal_type == np.ndarray:
            return self.to_numpy_array(data)

        elif self.internal_type in PANDAS_TYPES:
            self.to_pandas(data)

        else:
            raise TypeError('Type conversion to {} from {} is not possible.'.format(self.internal_type, type(data)))

    def from_statsmodels(self, data):

        from_type = type(data)
        self.index = getattr(data, 'index', None)

        if data is None:
            return None

        elif from_type == DesignMatrix:
            return data

        elif from_type in NUMPY_TYPES:
            data_to_return = self.from_numpy_array(data)

            if self.to_transpose(data_to_return):
                return transpose(data)

            else:
                return data_to_return

        elif from_type in PANDAS_TYPES:
            data_to_return = self.from_pandas(data)

            if self.to_transpose(data_to_return):
                return transpose(data)

            else:
                return data_to_return

        else:
            raise TypeError('Type conversion from {} to {} is not possible.'.format(from_type, self.external_type))

    def to_numpy_array(self, data):

        from_type = type(data)

        if from_type in NUMPY_TYPES:
            return data

        elif from_type == list:
            return np.array(data)

        elif from_type == np.recarray:
            return data.view(np.ndarray)

        elif from_type == pd.Series:

            return data.values

        elif from_type == pd.DataFrame:
            return data.values

        else:
            try:
                return np.asarray(data)
            except TypeError:
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
            return pd.Series(data=data, index=index)

        elif self.external_type == pd.DataFrame:
            ndim = getattr(data, 'ndim', None)

            if ndim in [1, None]:
                return pd.Series(data=data, index=self.index)

            elif self.ndim == ndim:
                return pd.DataFrame(data=data, columns=self.columns, index=self.index)

            else:
                return pd.DataFrame(data=data, index=self.index)

        else:
            return data

    def from_pandas(self, data):

        from_type = type(data)

        if from_type == self.external_type:
            return data

        elif self.external_type == np.ndarray:
            return data.values

        elif from_type == pd.Series and self.external_type == pd.DataFrame:
            return data.to_frame()

        elif from_type == pd.DataFrame and self.external_type == pd.Series:
            if data.ndim == 1:
                return pd.Series(data.values, index=data.index, name=data.columns[0])
            else:
                raise TypeError('Cannot convert multi dimensional DataFrame to a Series')

        elif self.external_type == list:
            return data.values.tolist()

        else:
            return data


NumPyInterface = partial(DataInterface, [np.ndarray])
PandasInterface = partial(DataInterface, [np.ndarray, pd.Series, pd.DataFrame])


def get_ndim(data):
    if type(data) == pd.Series:
        return 1

    if type(data) == pd.DataFrame:
        if safe_get(data.shape, 1, None) == 1 and safe_get(data.shape, 0, None) > 1:
            return 1

        elif safe_get(data.shape, 1, None) > 1 and safe_get(data.shape, 0, None) == 1:
            return 1

        else:
            return data.ndim

    try:
        data = np.asarray(data)
        data_ndim = data.ndim
        data_squeeze_ndim = data.squeeze().ndim

        if data_ndim > 1 and data_squeeze_ndim == 1:
            return 1
        else:
            return data_ndim

    except:
        return None


def is_col_vector(data):
    if type(data) == pd.Series:
        return False

    if type(data) == pd.DataFrame:
        if safe_get(data.shape, 1, None) == 1:
            return True
        else:
            return False

    try:
        data = np.asarray(data)
        data_ndim = data.ndim
        data_squeeze_ndim = data.squeeze().ndim

        if data_ndim > 1 and data_squeeze_ndim == 1:
            return True
        else:
            return False
    except:
        raise ValueError('Cannot determine if the vector is a column')


def transpose(data):

    transpose_type = type(data)

    if is_col_vector(data):

        if transpose_type == np.ndarray:
            return data.squeeze()

        elif transpose_type == pd.DataFrame:
            return data.T.squeeze()

        else:
            raise TypeError('Cannot transpose {} into a row vector'.format(transpose_type))

    else:

        if transpose_type == np.ndarray:
            return data[np.newaxis].T

        elif transpose_type == pd.Series:
            data_col = data.values[np.newaxis].T
            return pd.DataFrame(data_col, index=data.index)

        elif transpose_type == pd.DataFrame:
            return data.T

        else:
            raise TypeError('Cannot transpose {} into a column vector'.format(transpose_type))


def safe_get(data, index, default):
    try:
        return data[index]

    except IndexError:
        return default


def get_dtypes(data):

    if hasattr(data, 'dtype'):
        return data.dtype

    elif hasattr(data, 'dtypes'):
        return data.dtypes

    else:
        return None


def apply_dtype_to_df(df, dtypes):

    for col, dtype in zip(df.columns, dtypes.values):
        df[col] = df[col].astype(dtype)

    return df
