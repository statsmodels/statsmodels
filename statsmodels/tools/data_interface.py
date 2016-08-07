import pandas as pd
import numpy as np
from functools import partial
from patsy import dmatrix
from patsy.design_info import DesignMatrix

NUMPY_TYPES = [np.ndarray, np.float64]
PANDAS_TYPES = [pd.Series, pd.DataFrame]


class DataInterface(object):

    def __init__(self, permitted_types, internal_type=None, data=None, external_type=None, model=None,
                 use_formula=False, require_2d=False):

        self.permitted_types = permitted_types
        self.internal_type = np.ndarray if internal_type is None else internal_type
        self.model = model
        self.use_formula = use_formula
        self.require_2d = require_2d

        if external_type is not None:
            self.external_type = external_type

        elif data is not None:
            self.external_type = type(data)

        else:
            self.external_type = np.ndarray

        self.columns = None
        self.name = None
        self.dtype = None
        self.index = None
        self.ndim = None
        self.is_nested_row_vector = None
        self.is_col_vector = None

    def init_data_interface(self, data):

        self.columns = getattr(data, 'columns', None)
        self.name = getattr(data, 'name', None)
        self.ndim = get_ndim(data)
        self.is_nested_row_vector = is_nested_row_vector(data)
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

        if type(data) in self.permitted_types:
            to_return = data

        elif self.internal_type == np.ndarray:
            to_return = self.to_numpy_array(data)

        elif self.internal_type in PANDAS_TYPES:
            to_return = self.to_pandas(data)

        elif self.internal_type == list:
            to_return = self.to_list(data)

        else:
            raise TypeError('Type conversion to {} from {} is not possible.'.format(self.internal_type, type(data)))

        if self.require_2d and self.ndim == 1 and not is_col_vector(to_return):
            return transpose(to_return)

        else:
            return to_return

    def from_statsmodels(self, data):

        if data is None:
            return None

        from_type = type(data)

        if from_type == DesignMatrix:
            return data

        self.index = getattr(data, 'index', None)

        if from_type in NUMPY_TYPES:
            data_to_return = self.from_numpy_array(data)

        elif from_type in PANDAS_TYPES:
            data_to_return = self.from_pandas(data)

        elif from_type == list:
            data_to_return = self.from_list(data)

        else:
            raise TypeError('Type conversion from {} to {} is not possible.'.format(from_type, self.external_type))

        # from pdb import set_trace
        # set_trace()

        if self.to_transpose(data_to_return):
            data_to_return = transpose(data)

        if self.ndim == 1 and (not self.is_nested_row_vector and is_nested_row_vector(data_to_return)):
            return data_to_return[0]

        elif self.ndim == 1 and (self.is_nested_row_vector and not is_nested_row_vector(data_to_return)):
            return to_nested_row_vector(data_to_return)

        else:
            return data_to_return

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
                raise TypeError('Type conversion to numpy from {} is not possible.'.format(from_type))

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

    def to_list(self, data):

        from_type = type(data)

        if from_type == list:
            return data

        else:
            return self.to_numpy_array(data).tolist()

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
            raise TypeError('Cannot convert from numpy array to {}'.format(self.external_type))

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
            raise TypeError('Cannot convert from {} to {}'.format(from_type, self.external_type))

    def from_list(self, data):

        from_type = type(data)

        if from_type == self.external_type:
            return data

        elif self.external_type == np.ndarray:
            return np.asarray(data)

        elif self.external_type == pd.DataFrame:
            return pd.DataFrame(data, index=self.index, columns=self.columns)

        elif self.external_type == pd.Series:
            if data.ndim == 1:
                return pd.Series(data, index=self.index, name=self.name)
            else:
                raise TypeError('Cannot convert multi dimensional DataFrame to a Series')

        else:
            raise TypeError('Cannot convert from list to {}'.format(self.external_type))


NumPyInterface = partial(DataInterface, [np.ndarray])
SeriesInterface = partial(DataInterface, [pd.Series], pd.Series)
DataFrameInterface = partial(DataInterface, [pd.DataFrame], pd.DataFrame)
ListInterface = partial(DataInterface, [list], list)


def get_ndim(data):

    if type(data) == pd.Series:
        return 1

    if type(data) != pd.DataFrame:

        try:
            data = np.asarray(data)

        except TypeError:
            raise TypeError('Cannot find dimension of {}'.format(type(data)))

    if get_shape_dim(data.shape, 0) > 1 and get_shape_dim(data.shape, 1) == 1:
        return 1

    elif get_shape_dim(data.shape, 0) == 1 and get_shape_dim(data.shape, 1) > 1:
        return 1

    else:
        return data.ndim


def is_nested_row_vector(data):

    data_type = type(data)

    if data_type == pd.Series:
        return False

    if data_type != pd.DataFrame and data_type != np.ndarray:

        try:
            data = np.asarray(data)

        except TypeError:
            raise TypeError('Cannot find dimension of {}'.format(type(data)))

    if get_shape_dim(data.shape, 0) == 1 and get_shape_dim(data.shape, 1) > 1:
        return True

    else:
        return False


def to_nested_row_vector(data):

    data_type = type(data)

    if data_type in [pd.Series, pd.DataFrame]:
        return pd.DataFrame([data.values])

    elif data_type == np.ndarray:
        return data[np.newaxis]

    elif data_type == list:
        return [data]

    else:
        raise TypeError('Cannot convert {} to a nested row vector'.format(data_type))


def is_col_vector(data):

    if type(data) == pd.Series:
        return False

    if type(data) != pd.DataFrame:

        try:
            data = np.asarray(data)

        except TypeError:
            raise TypeError('Cannot convert {} to array'.format(type(data)))

    if get_shape_dim(data.shape, 0) > 1 and get_shape_dim(data.shape, 1) == 1:
        return True

    else:
        return False


def transpose(data):

    transpose_type = type(data)

    if is_col_vector(data):

        if transpose_type == np.ndarray:
            return data.squeeze()

        elif transpose_type == pd.DataFrame:
            return data.T.squeeze()

        elif transpose_type == list:
            return np.asarray(data).squeeze().tolist()

        else:
            raise TypeError('Cannot transpose {} into a row vector'.format(transpose_type))

    else:

        if transpose_type == pd.Series:
            data_col = data.values[np.newaxis].T
            return pd.DataFrame(data_col, index=data.index)

        elif transpose_type == np.ndarray:

            if get_shape_dim(data.shape, 0) == 1 and get_shape_dim(data.shape, 1) > 1:
                data = data[0]

            return data[np.newaxis].T

        elif transpose_type == pd.DataFrame:
            return data.T

        elif transpose_type == list:

            data = np.asarray(data)

            if get_shape_dim(data.shape, 0) == 1 and get_shape_dim(data.shape, 1) > 1:
                data = data[0]

            return data[np.newaxis].T.tolist()

        else:
            raise TypeError('Cannot transpose {} into a column vector'.format(transpose_type))


def get_shape_dim(data, index):
    try:
        return data[index]

    except IndexError:
        return 0
