import pandas as pd
import numpy as np
from functools import partial
from patsy import dmatrix
from patsy.design_info import DesignMatrix

NUMPY_TYPES = [np.ndarray, np.float64, np.recarray]
PANDAS_TYPES = [pd.Series, pd.DataFrame]
DEFAULT_EXTERNAL_TYPE = np.ndarray


class DataInterface(object):

    def __init__(self, permitted_types, internal_type=None, external_type=None, model=None,
                 use_formula=False, require_col_vector=False, at_least_2d=False):

        self.permitted_types = permitted_types
        self.internal_type = np.ndarray if internal_type is None else internal_type
        self.model = model
        self.use_formula = use_formula
        self.require_col_vector = require_col_vector
        self.at_least_2d = at_least_2d
        self.external_type = external_type if external_type is not None else DEFAULT_EXTERNAL_TYPE

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

        if self.external_type == DEFAULT_EXTERNAL_TYPE and data is not None:
            if not np.isscalar(data):
                self.external_type = type(data)

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
            to_return = to_numpy_array(data)

        elif self.internal_type in PANDAS_TYPES:
            to_return = to_pandas(data, name=self.name, columns=self.columns)

        elif self.internal_type == list:
            to_return = to_list(data)

        else:
            raise TypeError('Type conversion to {} from {} is not possible.'.format(self.internal_type, type(data)))

        if self.require_col_vector and get_ndim(to_return) == 1 and not is_col_vector(to_return):
            to_return = transpose(to_return)

        if self.at_least_2d and get_ndim(to_return) == 1 and not is_col_vector(to_return):
            to_return = to_nested_row_vector(to_return)

        return to_return

    def from_statsmodels(self, data):

        if data is None:
            return None

        from_type = type(data)

        if from_type == DesignMatrix:
            return data

        self.index = getattr(data, 'index', None)

        if from_type in NUMPY_TYPES:
            data_to_return = from_numpy_array(data, self.external_type, index=self.index, name=self.name,
                                              columns=self.columns, from_ndim=self.ndim)

        elif from_type in PANDAS_TYPES:
            data_to_return = from_pandas(data, self.external_type)

        elif from_type == list:
            data_to_return = from_list(data, self.external_type, index=self.index, name=self.name, columns=self.columns)

        else:
            raise TypeError('Type conversion from {} to {} is not possible.'.format(from_type, self.external_type))

        if self.to_transpose(data_to_return):
            data_to_return = transpose(data)

        if self.ndim == 1 and (not self.is_nested_row_vector and is_nested_row_vector(data_to_return)):
            return data_to_return[0]

        elif self.ndim == 1 and (self.is_nested_row_vector and not is_nested_row_vector(data_to_return)):
            return to_nested_row_vector(data_to_return)

        else:
            return data_to_return


NumPyInterface = partial(DataInterface, [np.ndarray])
SeriesInterface = partial(DataInterface, [pd.Series], pd.Series)
DataFrameInterface = partial(DataInterface, [pd.DataFrame], pd.DataFrame)
PandasInterface = partial(DataInterface, PANDAS_TYPES, pd.DataFrame)
ListInterface = partial(DataInterface, [list], list)


def to_categorical(data, col=None, dictnames=False, drop=False):

    ndim = get_ndim(data)
    to_type = np.recarray

    if type(data) == list:
        data = flatten_list(data)

    if is_recarray(data):

        if ndim > 1:
            if col is None:
                raise ValueError('Col must not be None for multi dimensional recarray')
            else:
                if type(col) == int:
                    col = data.dtype.names[col]

                rec_column = data[col]
                to_dummies = transpose(rec_column)

                to_drop = drop_recarray_column(data, col)
                to_drop = recarray_to_pandas(to_drop)

        else:
            to_drop = None

            if data.dtype.names is not None:
                name = data.dtype.names[0]
                to_dummies = data[name]
            else:
                to_dummies = data

            if is_col_vector(to_dummies):
                to_dummies = transpose(to_dummies)

        to_dummies = pd.Series(to_dummies)

    else:
        to_type = type(data)
        to_dummies = to_pandas(data)

        if ndim > 1:
            to_drop = to_dummies.drop(col, 1)

            if isinstance(col, int):
                to_dummies = to_dummies.iloc[:, col]
            elif isinstance(col, str):
                to_dummies = to_dummies.loc[:, col]

        else:
            to_drop = None

    source_data = to_pandas(data)
    dummies = pd.get_dummies(to_dummies)

    interface = NumPyInterface(external_type=to_type)

    if not drop:
        result = pd.concat([source_data, dummies], axis=1)
        return interface.from_statsmodels(result)
    else:
        result = pd.concat([to_drop, dummies], axis=1)
        return interface.from_statsmodels(result)


def to_numpy_array(data):
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


def to_list(data):
    from_type = type(data)

    if from_type == list:
        return data

    else:
        return to_numpy_array(data).tolist()


def to_pandas(data, name=None, columns=None):
    from_type = type(data)

    if from_type in PANDAS_TYPES:
        return data

    elif is_recarray(data):
        return recarray_to_pandas(data)

    else:
        np_data = to_numpy_array(data)

        if np_data.ndim == 1:
            return pd.Series(np_data, name=name)

        else:
            return pd.DataFrame(np_data, columns=columns)


def from_numpy_array(data, to_type, index=None, name=None, columns=None, from_ndim=None):

    assert type(data) in [np.ndarray, np.recarray]

    from_type = type(data)

    if from_type == to_type:
        return data

    elif to_type == list:
        return data.tolist()

    elif to_type == np.recarray:
        return data.view(np.recarray)

    elif to_type == pd.Series:
        index = getattr(data, 'index', None)
        return pd.Series(data=data, index=index, name=name)

    elif to_type == pd.DataFrame:
        ndim = getattr(data, 'ndim', None)

        if ndim in [1, None]:
            return pd.Series(data=data, index=index, name=name)

        elif from_ndim == ndim:
            return pd.DataFrame(data=data, columns=columns, index=index)

        else:
            return pd.DataFrame(data=data, index=index)

    else:
        raise TypeError('Cannot convert from numpy array to {}'.format(to_type))


def from_pandas(data, to_type):

    assert type(data) in [pd.DataFrame, pd.Series]

    from_type = type(data)

    if from_type == to_type:
        return data

    elif to_type == np.ndarray:
        return data.values

    elif from_type == pd.Series and to_type == pd.DataFrame:
        return data

    elif from_type == pd.DataFrame and to_type == pd.Series:
        if data.ndim == 1:
            return pd.Series(data.values, index=data.index, name=data.columns[0])
        else:
            raise TypeError('Cannot convert multi dimensional DataFrame to a Series')

    elif to_type == list:
        return data.values.tolist()

    elif to_type == np.recarray:
        return data.to_records(index=False)

    else:
        raise TypeError('Cannot convert from {} to {}'.format(from_type, to_type))


def from_list(data, to_type, index=None, name=None, columns=None):

    assert type(data) == list

    if to_type == list:
        return data

    elif to_type == np.ndarray:
        return np.asarray(data)

    elif to_type == pd.DataFrame:
        return pd.DataFrame(data, index=index, columns=columns)

    elif to_type == pd.Series:
        if data.ndim == 1:
            return pd.Series(data, index=index, name=name)
        else:
            raise TypeError('Cannot convert multi dimensional DataFrame to a Series')

    else:
        raise TypeError('Cannot convert from list to {}'.format(to_type))


def get_ndim(data):

    data_type = type(data)

    if is_recarray(data):
        # if there is only one dtype, a len of 0 will be returned.
        return max(len(data.dtype), 1)

    if data_type == pd.Series:
        return 1

    if data_type != pd.DataFrame:

        try:
            data = np.asarray(data)

        except TypeError:
            raise TypeError('Cannot find dimension of {}'.format(data_type))

    if len(data.shape) > 2:
        dims = sum(1 for elem in data.shape if elem > 1)

        if dims > 2:
            raise ValueError('Data must have no more than two dimensions')
        else:
            return dims

    elif get_shape_dim(data.shape, 0) > 1 and get_shape_dim(data.shape, 1) == 1:
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
        return np.atleast_2d(data)

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

        if transpose_type in [np.ndarray, np.recarray]:
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

        elif transpose_type in [np.ndarray, np.recarray]:

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


def is_recarray(data):

    if isinstance(data, np.recarray):
        return True
    elif hasattr(data, 'dtype'):
        if data.dtype.names is not None:
            return True
        else:
            return False
    else:
        return False


def recarray_to_pandas(data):

    data_list = []

    if data.dtype.names is None:

        if is_col_vector(data):
            data = transpose(data)

        return pd.Series(data)

    else:

        for name in data.dtype.names:
            col = data[name]

            if is_col_vector(col):
                col = transpose(col)

            data_list.append(pd.Series(col, name=name))

        return pd.concat(data_list, axis=1)


def drop_recarray_column(rec, name):

    names = list(rec.dtype.names)

    if name in names:
        names.remove(name)

    result = rec[names]

    return result


def slice_2d(shape):

    s = []

    for index in shape:
        if index > 1:
            s.append(slice(None))
        else:
            s.append(0)

    return tuple(s)


def flatten_list(data):

    assert type(data) == list

    np_data = np.asarray(data)
    shape = np_data.shape

    dims = sum(1 for elem in shape if elem > 1)

    if dims > 2:
        raise ValueError('Data must have no more than two dimensions')
    else:
        slicer = slice_2d(shape)
        data = np_data[slicer].tolist()

    return data


def flatten_array(data):

    assert type(data) in [np.ndarray, np.recarray]

    shape = data.shape

    dims = sum(1 for elem in shape if elem > 1)

    if dims > 2:
        raise ValueError('Data must have no more than two dimensions')
    else:
        slicer = slice_2d(shape)
        data = data[slicer]

    return data