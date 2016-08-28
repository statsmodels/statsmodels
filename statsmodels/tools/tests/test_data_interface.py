from statsmodels.tools.data_interface import (NumPyInterface, ListInterface, SeriesInterface, DataFrameInterface,
                                              get_ndim, is_col_vector, transpose, to_categorical)

from pandas.util.testing import assert_frame_equal, assert_series_equal
from numpy.testing import assert_equal, assert_array_equal

import pandas as pd
import numpy as np

from pdb import set_trace


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

    assert_series_equal(data_series, numpy_to_series)


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

    assert_frame_equal(data_frame, numpy_to_data_frame)


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

    assert is_col_vector(list_row_vector) is False
    assert is_col_vector(list_row_vector2) is False
    assert is_col_vector(list_col_vector) is True
    assert is_col_vector(list_matrix) is False

    assert is_col_vector(np_row_vector) is False
    assert is_col_vector(np_row_vector2) is False
    assert is_col_vector(np_col_vector) is True
    assert is_col_vector(np_matrix) is False

    assert is_col_vector(pd_row_vector) is False
    assert is_col_vector(pd_row_vector2) is False
    assert is_col_vector(pd_col_vector) is True
    assert is_col_vector(pd_matrix) is False


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

    pd_col_vector_transpose = transpose(pd_col_vector)

    # Fix default behavior of changing Series name from None to 0
    pd_col_vector_transpose.name = None

    assert_series_equal(pd_row_vector, pd_col_vector_transpose)
    assert_frame_equal(pd_col_vector, transpose(pd_row_vector))
    assert_frame_equal(pd_col_vector, transpose(pd_row_vector2))


class TestCategorical(object):

    def __init__(self):

        stringabc = 'abcdefghijklmnopqrstuvwxy'

        self.des = np.random.randn(25, 2)
        self.instr = np.floor(np.arange(10, 60, step=2) / 10)

        x = np.zeros((25, 5))
        x[:5, 0] = 1
        x[5:10, 1] = 1
        x[10:15, 2] = 1
        x[15:20, 3] = 1
        x[20:25, 4] = 1
        self.dummy = x

        structdes = np.zeros((25, 1), dtype=[('var1', 'f4'), ('var2', 'f4'), ('instrument', 'f4'),
                                             ('str_instr', 'a10')])

        structdes['var1'] = self.des[:, 0][:, None]
        structdes['var2'] = self.des[:, 1][:, None]
        structdes['instrument'] = self.instr[:, None]

        string_var = [stringabc[0:5], stringabc[5:10], stringabc[10:15], stringabc[15:20], stringabc[20:25]]
        string_var *= 5
        self.string_var = np.array(sorted(string_var))

        structdes['str_instr'] = self.string_var[:, None]
        self.structdes = structdes
        self.recdes = structdes.view(np.recarray)


class TestCategoricalNumerical(TestCategorical):
    # TODO: use assert_raises to check that bad inputs are taken care of

    def test_array2d(self):
        set_trace()

        des = np.column_stack((self.des, self.instr, self.des))
        des = to_categorical(des, col=2)
        assert_array_equal(des[:, -5:], self.dummy)
        assert_equal(des.shape[1], 10)

    def test_array1d(self):
        set_trace()

        des = to_categorical(self.instr)
        assert_array_equal(des[:, -5:], self.dummy)
        assert_equal(des.shape[1], 6)

    def test_array2d_drop(self):
        des = np.column_stack((self.des, self.instr, self.des))
        des = to_categorical(des, col=2, drop=True)
        assert_array_equal(des[:, -5:], self.dummy)
        assert_equal(des.shape[1], 9)

    def test_array1d_drop(self):
        des = to_categorical(self.instr, drop=True)
        assert_array_equal(des, self.dummy)
        assert_equal(des.shape[1], 5)

    def test_recarray2d(self):
        des = to_categorical(self.recdes, col='instrument')
        # better way to do this?
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 9)

    def test_recarray2dint(self):
        des = to_categorical(self.recdes, col=2)
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 9)

    def test_recarray1d(self):
        instr = self.structdes['instrument'].view(np.recarray)
        dum = to_categorical(instr)
        test_dum = np.column_stack(([dum[_] for _ in dum.dtype.names[-5:]]))
        assert_array_equal(test_dum, self.dummy)
        assert_equal(len(dum.dtype.names), 6)

    def test_recarray1d_drop(self):
        instr = self.structdes['instrument'].view(np.recarray)
        dum = to_categorical(instr, drop=True)
        test_dum = np.column_stack(([dum[_] for _ in dum.dtype.names]))
        assert_array_equal(test_dum, self.dummy)
        assert_equal(len(dum.dtype.names), 5)

    def test_recarray2d_drop(self):
        des = to_categorical(self.recdes, col='instrument', drop=True)
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 8)

    def test_structarray2d(self):
        des = to_categorical(self.structdes, col='instrument')
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 9)

    def test_structarray2dint(self):
        des = to_categorical(self.structdes, col=2)
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 9)

    def test_structarray1d(self):
        instr = self.structdes['instrument'].view(dtype=[('var1', 'f4')])
        dum = to_categorical(instr)
        test_dum = np.column_stack(([dum[_] for _ in dum.dtype.names[-5:]]))
        assert_array_equal(test_dum, self.dummy)
        assert_equal(len(dum.dtype.names), 6)

    def test_structarray2d_drop(self):
        des = to_categorical(self.structdes, col='instrument', drop=True)
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 8)

    def test_structarray1d_drop(self):
        instr = self.structdes['instrument'].view(dtype=[('var1', 'f4')])
        dum = to_categorical(instr, drop=True)
        test_dum = np.column_stack(([dum[_] for _ in dum.dtype.names]))
        assert_array_equal(test_dum, self.dummy)
        assert_equal(len(dum.dtype.names), 5)

# def test_arraylike2d(self):
#        des = to_categorical(self.structdes.tolist(), col=2)
#        test_des = des[:,-5:]
#        assert_array_equal(test_des, self.dummy)
#        assert_equal(des.shape[1], 9)

#    def test_arraylike1d(self):
#        instr = self.structdes['instrument'].tolist()
#        dum = to_categorical(instr)
#        test_dum = dum[:,-5:]
#        assert_array_equal(test_dum, self.dummy)
#        assert_equal(dum.shape[1], 6)

#    def test_arraylike2d_drop(self):
#        des = to_categorical(self.structdes.tolist(), col=2, drop=True)
#        test_des = des[:,-5:]
#        assert_array_equal(test__des, self.dummy)
#        assert_equal(des.shape[1], 8)

#    def test_arraylike1d_drop(self):
#        instr = self.structdes['instrument'].tolist()
#        dum = to_categorical(instr, drop=True)
#        assert_array_equal(dum, self.dummy)
#        assert_equal(dum.shape[1], 5)


class TestCategoricalString(TestCategorical):
    # comment out until we have type coercion
    #    def test_array2d(self):
    #        des = np.column_stack((self.des, self.instr, self.des))
    #        des = to_categorical(des, col=2)
    #        assert_array_equal(des[:,-5:], self.dummy)
    #        assert_equal(des.shape[1],10)

    #    def test_array1d(self):
    #        des = to_categorical(self.instr)
    #        assert_array_equal(des[:,-5:], self.dummy)
    #        assert_equal(des.shape[1],6)

    #    def test_array2d_drop(self):
    #        des = np.column_stack((self.des, self.instr, self.des))
    #        des = to_categorical(des, col=2, drop=True)
    #        assert_array_equal(des[:,-5:], self.dummy)
    #        assert_equal(des.shape[1],9)

    def test_array1d_drop(self):
        des = to_categorical(self.string_var, drop=True)
        assert_array_equal(des, self.dummy)
        assert_equal(des.shape[1], 5)

    def test_recarray2d(self):
        des = to_categorical(self.recdes, col='str_instr')
        # better way to do this?
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 9)

    def test_recarray2dint(self):
        des = to_categorical(self.recdes, col=3)
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 9)

    def test_recarray1d(self):
        instr = self.structdes['str_instr'].view(np.recarray)
        dum = to_categorical(instr)
        test_dum = np.column_stack(([dum[_] for _ in dum.dtype.names[-5:]]))
        assert_array_equal(test_dum, self.dummy)
        assert_equal(len(dum.dtype.names), 6)

    def test_recarray1d_drop(self):
        instr = self.structdes['str_instr'].view(np.recarray)
        dum = to_categorical(instr, drop=True)
        test_dum = np.column_stack(([dum[_] for _ in dum.dtype.names]))
        assert_array_equal(test_dum, self.dummy)
        assert_equal(len(dum.dtype.names), 5)

    def test_recarray2d_drop(self):
        des = to_categorical(self.recdes, col='str_instr', drop=True)
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 8)

    def test_structarray2d(self):
        des = to_categorical(self.structdes, col='str_instr')
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 9)

    def test_structarray2dint(self):
        des = to_categorical(self.structdes, col=3)
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 9)

    def test_structarray1d(self):
        instr = self.structdes['str_instr'].view(dtype=[('var1', 'a10')])
        dum = to_categorical(instr)
        test_dum = np.column_stack(([dum[_] for _ in dum.dtype.names[-5:]]))
        assert_array_equal(test_dum, self.dummy)
        assert_equal(len(dum.dtype.names), 6)

    def test_structarray2d_drop(self):
        des = to_categorical(self.structdes, col='str_instr', drop=True)
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 8)

    def test_structarray1d_drop(self):
        instr = self.structdes['str_instr'].view(dtype=[('var1', 'a10')])
        dum = to_categorical(instr, drop=True)
        test_dum = np.column_stack(([dum[_] for _ in dum.dtype.names]))
        assert_array_equal(test_dum, self.dummy)
        assert_equal(len(dum.dtype.names), 5)

    def test_arraylike2d(self):
        pass

    def test_arraylike1d(self):
        pass

    def test_arraylike2d_drop(self):
        pass

    def test_arraylike1d_drop(self):
        pass
