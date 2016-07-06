from statsmodels.tools.data_interface import NumPyInterface
import pandas as pd


def test_list_numpy():
    data_list = [1, 2, 3]

    list_interface = NumPyInterface(external_type=list)
    list_to_numpy = list_interface.to_statsmodels(data_list)
    numpy_to_list = list_interface.from_statsmodels(list_to_numpy)

    assert data_list == numpy_to_list

def test_series_numpy():
    data_list = [1, 2, 3]

    data_series = pd.Series(data_list)
    series_interface = NumPyInterface(external_type=pd.Series)
    series_to_numpy = series_interface.to_statsmodels(data_series)
    numpy_to_series = series_interface.from_statsmodels(series_to_numpy)

    assert data_series.equals(numpy_to_series)

def test_data_frame_numpy():
    data_nested_list = [[1, 2, 3], [4, 5, 6]]

    data_frame = pd.DataFrame(data_nested_list)
    data_frame_interface = NumPyInterface(external_type=pd.DataFrame)
    data_frame_to_numpy = data_frame_interface.to_statsmodels(data_frame)
    numpy_to_data_frame = data_frame_interface.from_statsmodels(data_frame_to_numpy)

    assert data_frame.equals(numpy_to_data_frame)
