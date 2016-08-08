
from statsmodels.tools.data_interface import (NumPyInterface, ListInterface, SeriesInterface, DataFrameInterface,
                                              get_ndim, is_col_vector, transpose)

from pandas.util.testing import assert_frame_equal
import pandas as pd
import numpy as np


def test_list_numpy():

    data_list = [1, 2, 3]

    list_interface = NumPyInterface(external_type=list)
    list_to_numpy = list_interface.to_statsmodels(data_list)
    numpy_to_list = list_interface.from_statsmodels(list_to_numpy)

    assert data_list == numpy_to_list


def test_numpy_list():

    data = np.array([1, 2, 3])

    list_interface = ListInterface(external_type=np.ndarray)
    numpy_to_list = list_interface.to_statsmodels(data)
    list_to_numpy = list_interface.from_statsmodels(numpy_to_list)

    assert np.array_equal(data, list_to_numpy)


def test_series_numpy():

    data_list = [1, 2, 3]

    data_series = pd.Series(data_list)
    series_interface = NumPyInterface(external_type=pd.Series)
    series_to_numpy = series_interface.to_statsmodels(data_series)
    numpy_to_series = series_interface.from_statsmodels(series_to_numpy)

    assert data_series.equals(numpy_to_series)


def test_numpy_series():

    data = np.array([1, 2, 3])

    series_interface = SeriesInterface(external_type=np.ndarray)
    numpy_to_series = series_interface.to_statsmodels(data)
    series_to_numpy = series_interface.from_statsmodels(numpy_to_series)

    assert np.array_equal(data, series_to_numpy)


def test_data_frame_numpy():

    data_nested_list = [[1, 2, 3], [4, 5, 6]]

    data_frame = pd.DataFrame(data_nested_list)
    data_frame_interface = NumPyInterface(external_type=pd.DataFrame)
    data_frame_to_numpy = data_frame_interface.to_statsmodels(data_frame)
    numpy_to_data_frame = data_frame_interface.from_statsmodels(data_frame_to_numpy)

    assert data_frame.equals(numpy_to_data_frame)


def test_numpy_data_frame():

    data = np.array([[1, 2, 3], [4, 5, 6]])

    data_frame_interface = DataFrameInterface(external_type=np.ndarray)
    numpy_to_data_frame = data_frame_interface.to_statsmodels(data)
    data_frame_to_numpy = data_frame_interface.from_statsmodels(numpy_to_data_frame)

    assert np.array_equal(data, data_frame_to_numpy)


def test_numpy_numpy():

    data_list = [1.368312, 2.667389, 2.387636, 1.797382, 1.935495, 3.482808, 2.520573, 2.804281, 1.264108, 1.305208]
    data_list_nested = [data_list]

    data = np.array(data_list)
    data_nested = np.array(data_list_nested)
    data_transpose = transpose(data)

    numpy_interface = NumPyInterface()
    numpy_internal = numpy_interface.to_statsmodels(data)
    numpy_result = numpy_interface.from_statsmodels(numpy_internal)

    assert np.array_equal(data, numpy_result)

    numpy_interface = NumPyInterface()
    numpy_internal = numpy_interface.to_statsmodels(data_nested)
    numpy_result = numpy_interface.from_statsmodels(numpy_internal)

    assert np.array_equal(data_nested, numpy_result)

    numpy_interface = NumPyInterface()
    numpy_interface.to_statsmodels(data)
    numpy_result = numpy_interface.from_statsmodels(data_transpose)

    assert np.array_equal(data, numpy_result)

    numpy_interface = NumPyInterface()
    numpy_interface.to_statsmodels(data_transpose)
    numpy_result = numpy_interface.from_statsmodels(data)

    assert np.array_equal(data_transpose, numpy_result)

    numpy_interface = NumPyInterface()
    numpy_interface.to_statsmodels(data)
    numpy_result = numpy_interface.from_statsmodels(data_nested)

    assert np.array_equal(data, numpy_result)

    numpy_interface = NumPyInterface()
    numpy_interface.to_statsmodels(data_nested)
    numpy_result = numpy_interface.from_statsmodels(data)

    assert np.array_equal(data_nested, numpy_result)


def test_list_list():

    data = [1.368312, 2.667389, 2.387636, 1.797382, 1.935495, 3.482808, 2.520573, 2.804281, 1.264108, 1.305208]
    data_nested = [data]
    data_transpose = transpose(data)

    list_interface = ListInterface(external_type=list)
    list_internal = list_interface.to_statsmodels(data)
    list_result = list_interface.from_statsmodels(list_internal)

    assert data == list_result

    list_interface = ListInterface(external_type=list)
    list_internal = list_interface.to_statsmodels(data_nested)
    list_result = list_interface.from_statsmodels(list_internal)

    assert data_nested == list_result

    list_interface = ListInterface(external_type=list)
    list_interface.to_statsmodels(data)
    list_result = list_interface.from_statsmodels(data_transpose)

    assert data == list_result

    list_interface = ListInterface(external_type=list)
    list_interface.to_statsmodels(data_transpose)
    list_result = list_interface.from_statsmodels(data)

    assert data_transpose == list_result

    list_interface = ListInterface(external_type=list)
    list_interface.to_statsmodels(data)
    list_result = list_interface.from_statsmodels(data_nested)

    assert data == list_result

    list_interface = ListInterface(external_type=list)
    list_interface.to_statsmodels(data_nested)
    list_result = list_interface.from_statsmodels(data)

    assert data_nested == list_result


def test_ndim():

    list_row_vector = [1, 2, 3]
    list_row_vector2 = [[1, 2, 3]]
    list_col_vector = [[1], [2], [3]]
    list_matrix = [[1, 2, 3], [4, 5, 6]]

    np_row_vector = np.array(list_row_vector)
    np_row_vector2 = np.array(list_row_vector2)
    np_col_vector = np.array(list_col_vector)
    np_matrix = np.array(list_matrix)

    pd_row_vector = pd.Series(np_row_vector)
    pd_row_vector2 = pd.DataFrame(np_row_vector2)
    pd_col_vector = pd.DataFrame(np_col_vector)
    pd_matrix = pd.DataFrame(np_matrix)

    assert get_ndim(list_row_vector) == 1
    assert get_ndim(list_row_vector2) == 1
    assert get_ndim(list_col_vector) == 1
    assert get_ndim(list_matrix) == 2

    assert get_ndim(np_row_vector) == 1
    assert get_ndim(np_row_vector2) == 1
    assert get_ndim(np_col_vector) == 1
    assert get_ndim(np_matrix) == 2

    assert get_ndim(pd_row_vector) == 1
    assert get_ndim(pd_row_vector2) == 1
    assert get_ndim(pd_col_vector) == 1
    assert get_ndim(pd_matrix) == 2

    assert is_col_vector(list_row_vector) == False
    assert is_col_vector(list_row_vector2) == False
    assert is_col_vector(list_col_vector) == True
    assert is_col_vector(list_matrix) == False

    assert is_col_vector(np_row_vector) == False
    assert is_col_vector(np_row_vector2) == False
    assert is_col_vector(np_col_vector) == True
    assert is_col_vector(np_matrix) == False

    assert is_col_vector(pd_row_vector) == False
    assert is_col_vector(pd_row_vector2) == False
    assert is_col_vector(pd_col_vector) == True
    assert is_col_vector(pd_matrix) == False


def test_transpose():

    list_row_vector = [1, 2, 3]
    list_row_vector2 = [[1, 2, 3]]
    list_col_vector = [[1], [2], [3]]

    np_row_vector = np.array(list_row_vector)
    np_row_vector2 = np.array(list_row_vector2)
    np_col_vector = np.array(list_col_vector)

    pd_row_vector = pd.Series(np_row_vector)
    pd_row_vector2 = pd.DataFrame(np_row_vector2)
    pd_col_vector = pd.DataFrame(np_col_vector)

    assert list_col_vector == transpose(list_row_vector)
    assert list_col_vector == transpose(list_row_vector2)
    assert list_row_vector == transpose(list_col_vector)

    np.testing.assert_equal(np_col_vector, transpose(np_row_vector))
    np.testing.assert_equal(np_col_vector, transpose(np_row_vector2))
    np.testing.assert_equal(np_row_vector, transpose(np_col_vector))

    assert pd_row_vector.equals(transpose(pd_col_vector))
    assert_frame_equal(pd_col_vector, transpose(pd_row_vector))
    assert_frame_equal(pd_col_vector, transpose(pd_row_vector2))
