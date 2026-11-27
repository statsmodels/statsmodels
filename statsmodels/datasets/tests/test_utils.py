from statsmodels.compat.pandas import assert_frame_equal
from statsmodels.compat.python import PYTHON_IMPL_WASM

from pathlib import Path
from socket import timeout
from urllib.error import HTTPError, URLError

import numpy as np
from numpy.testing import assert_, assert_array_equal
import pandas as pd
import pytest

from statsmodels.datasets import check_internet, get_rdataset, utils, webuse

CUR_DIR = Path(__file__).resolve().parent

IGNORED_EXCEPTIONS = (HTTPError, URLError, UnicodeEncodeError, timeout)
if not PYTHON_IMPL_WASM:
    from ssl import SSLError

    IGNORED_EXCEPTIONS += (SSLError,)


@pytest.mark.smoke
def test_get_rdataset():
    test_url = (
        "https://raw.githubusercontent.com/vincentarelbundock/"
        "Rdatasets/master/csv/datasets/cars.csv"
    )
    internet_available = check_internet(test_url)
    if not internet_available:  # pragma: no cover
        pytest.skip("Unable to retrieve file - skipping test")
    try:
        duncan = get_rdataset("Duncan", "carData", cache=CUR_DIR)
    except IGNORED_EXCEPTIONS:
        pytest.skip("Failed with HTTPError or URLError, these are random")
    assert_(isinstance(duncan, utils.Dataset))
    duncan = get_rdataset("Duncan", "carData", cache=CUR_DIR)
    assert_(duncan.from_cache)


@pytest.mark.smoke
@pytest.mark.thread_unsafe(reason="Disk access")
def test_get_rdataset_write_read_cache():
    # test writing and reading cache
    try:
        guerry = get_rdataset("Guerry", "HistData", cache=CUR_DIR)
    except IGNORED_EXCEPTIONS:
        pytest.skip("Failed with HTTPError or URLError, these are random")

    assert_(guerry.from_cache is False)
    guerry2 = get_rdataset("Guerry", "HistData", cache=CUR_DIR)
    assert_(guerry2.from_cache is True)
    fn = (
        "raw.githubusercontent.com,vincentarelbundock,Rdatasets,master,csv,"
        "HistData,Guerry-v2.csv.zip"
    )
    Path(CUR_DIR).joinpath(fn).unlink()
    fn = (
        "raw.githubusercontent.com,vincentarelbundock,Rdatasets,master,doc,"
        "HistData,rst,Guerry-v2.rst.zip"
    )
    Path(CUR_DIR).joinpath(fn).unlink()


def test_webuse():
    # test copied and adjusted from iolib/tests/test_foreign
    from statsmodels.iolib.tests.results.macrodata import macrodata_result

    res2 = np.array([list(row) for row in macrodata_result])
    base_gh = (
        "https://github.com/statsmodels/statsmodels/raw/main/"
        "statsmodels/datasets/macrodata/"
    )
    internet_available = check_internet(base_gh)
    if not internet_available:  # pragma: no cover
        pytest.skip("Unable to retrieve file - skipping test")
    try:
        res1 = webuse("macrodata", baseurl=base_gh, as_df=False)
    except IGNORED_EXCEPTIONS:
        pytest.skip("Failed with HTTPError or URLError, these are random")
    assert_array_equal(res1, res2)


def test_webuse_pandas():
    # test copied and adjusted from iolib/tests/test_foreign
    from statsmodels.compat.pandas import assert_frame_equal

    from statsmodels.datasets import macrodata

    dta = macrodata.load_pandas().data
    base_gh = (
        "https://github.com/statsmodels/statsmodels/raw/main/"
        "statsmodels/datasets/macrodata/"
    )
    internet_available = check_internet(base_gh)
    if not internet_available:  # pragma: no cover
        pytest.skip("Unable to retrieve file - skipping test")
    try:
        res1 = webuse("macrodata", baseurl=base_gh)
    except IGNORED_EXCEPTIONS:
        pytest.skip("Failed with HTTP Error, these are random")
    res1 = res1.astype(float)
    assert_frame_equal(res1, dta.astype(float))


def test_clear_data_home(tmp_path):
    # clear_data_home is documented to delete the entire data home cache;
    # populate it with a file and confirm the whole directory disappears.
    data_home = utils.get_data_home(str(tmp_path / "sm_data"))
    assert Path(data_home).is_dir()
    (Path(data_home) / "cached_file.csv").write_text("a,b\n1,2\n")
    assert (Path(data_home) / "cached_file.csv").exists()

    utils.clear_data_home(data_home)

    assert not Path(data_home).exists()


def test_clear_data_home_uses_default_when_none(tmp_path, monkeypatch):
    # data_home=None (the default) falls back to get_data_home's own
    # default resolution (env var or ~/statsmodels_data)
    monkeypatch.setenv("STATSMODELS_DATA", str(tmp_path / "env_data_home"))

    data_home = utils.get_data_home(None)
    assert Path(data_home).is_dir()

    utils.clear_data_home(None)

    assert not Path(data_home).exists()


def _process_pandas_data():
    rng = np.random.default_rng(20261126)
    return pd.DataFrame(rng.standard_normal((10, 4)), columns=["a", "b", "c", "d"])


@pytest.mark.parametrize(
    "endog_idx",
    [1, np.int64(1), np.int32(1), np.array(1)],
    ids=["int", "int64", "int32", "0d-array"],
)
@pytest.mark.parametrize(
    "exog_idx, exog_names",
    [(None, ["a", "c", "d"]), ([3, 0], ["d", "a"])],
    ids=["exog-None", "exog-list"],
)
def test_process_pandas_int_like_endog_idx(endog_idx, exog_idx, exog_names):
    # A NumPy integer is a column position, as a Python int is
    data = _process_pandas_data()
    dataset = utils.process_pandas(data, endog_idx=endog_idx, exog_idx=exog_idx)
    assert dataset.endog_name == "b"
    pd.testing.assert_series_equal(dataset.endog, data["b"])
    assert dataset.exog_name == exog_names
    pd.testing.assert_frame_equal(dataset.exog, data[exog_names])


@pytest.mark.parametrize("endog_idx", [True, np.True_], ids=["True", "np.True_"])
def test_process_pandas_bool_endog_idx(endog_idx):
    # A bool is not a column position: True selected the column at position 1
    data = _process_pandas_data()
    with pytest.raises(KeyError):
        utils.process_pandas(data, endog_idx=endog_idx)


@pytest.mark.parametrize("index_idx", [None, 3])
@pytest.mark.parametrize("endog_idx", [0, ["a"]], ids=["int", "list"])
def test_process_pandas_int_exog_idx(endog_idx, index_idx):
    # An int exog_idx raised AttributeError: 'Series' object has no
    # attribute 'columns'. It selects one column, kept as a DataFrame.
    data = pd.DataFrame(np.arange(12.0).reshape(3, 4), columns=list("abcd"))
    ds = utils.process_pandas(
        data, endog_idx=endog_idx, exog_idx=1, index_idx=index_idx
    )
    expected = data[["b"]].copy()
    if index_idx is not None:
        expected.index = pd.Index(data["d"])
    assert_frame_equal(ds.exog, expected)
    assert ds.exog_name == ["b"]


@pytest.mark.parametrize("index_idx", [None, 2], ids=["index-None", "index-2"])
@pytest.mark.parametrize(
    "exog_idx, exog_names",
    [
        (None, ["a", "c", "d"]),
        ([3, 0], ["d", "a"]),
        (2, ["c"]),
        (np.array(2), ["c"]),
    ],
    ids=["exog-None", "exog-list", "exog-int", "exog-0d-array"],
)
def test_process_pandas_str_endog_idx(exog_idx, exog_names, index_idx):
    # A str selects one column as a Series, as its position does. It raised
    # AttributeError: 'Series' object has no attribute 'columns'. Both select
    # exog in the same way; a 0-d array exog_idx with an int endog_idx raised
    # an AttributeError on pandas < 3
    data = _process_pandas_data()
    dataset = utils.process_pandas(
        data, endog_idx="b", exog_idx=exog_idx, index_idx=index_idx
    )
    by_position = utils.process_pandas(
        data, endog_idx=1, exog_idx=exog_idx, index_idx=index_idx
    )
    assert dataset.endog_name == by_position.endog_name == "b"
    assert dataset.exog_name == by_position.exog_name == exog_names
    pd.testing.assert_series_equal(dataset.endog, by_position.endog)
    pd.testing.assert_frame_equal(dataset.exog, by_position.exog)
    if index_idx is None:
        pd.testing.assert_series_equal(dataset.endog, data["b"])
        pd.testing.assert_frame_equal(dataset.exog, data[exog_names])


@pytest.mark.parametrize(
    "endog_idx",
    [
        [1, 2],
        np.array([1, 2]),
        [np.int64(1), np.array(2)],
        range(1, 3),
        pd.Index([1, 2]),
    ],
    ids=["list", "array", "numpy-ints", "range", "index"],
)
@pytest.mark.parametrize("exog_idx", [None, [3, 0]], ids=["exog-None", "exog-list"])
def test_process_pandas_int_sequence_endog_idx(endog_idx, exog_idx):
    # Integers in a sequence are column positions, as an int and exog_idx
    # are. They were passed to .loc as labels, which raised a KeyError for
    # str column names. The labels of the same columns are the reference.
    data = _process_pandas_data()
    dataset = utils.process_pandas(data, endog_idx=endog_idx, exog_idx=exog_idx)
    by_label = utils.process_pandas(data, endog_idx=["b", "c"], exog_idx=exog_idx)
    assert dataset.endog_name == by_label.endog_name == ["b", "c"]
    pd.testing.assert_frame_equal(dataset.endog, data[["b", "c"]])
    exog_names = ["a", "d"] if exog_idx is None else ["d", "a"]
    assert dataset.exog_name == by_label.exog_name == exog_names
    pd.testing.assert_frame_equal(dataset.exog, data[exog_names])


@pytest.mark.parametrize("position", [0, 3, -1])
def test_process_pandas_one_position_list_matches_int(position):
    # [k] selects the same column as k, as a DataFrame
    data = _process_pandas_data()
    scalar = utils.process_pandas(data, endog_idx=position)
    listed = utils.process_pandas(data, endog_idx=[position])
    assert listed.endog_name == [scalar.endog_name]
    pd.testing.assert_frame_equal(listed.endog, scalar.endog.to_frame())
    pd.testing.assert_frame_equal(listed.exog, scalar.exog)


def test_process_pandas_int_sequence_endog_idx_int_columns():
    # With integer column names that differ from their positions, 0 was the
    # column at position 0 but [0] was the column named 0
    data = pd.DataFrame(np.arange(12.0).reshape(4, 3), columns=[2, 0, 1])
    scalar = utils.process_pandas(data, endog_idx=0)
    listed = utils.process_pandas(data, endog_idx=[0])
    assert scalar.endog_name == 2
    assert listed.endog_name == [2]
    pd.testing.assert_series_equal(listed.endog[2], scalar.endog)
    assert listed.exog_name == scalar.exog_name == [0, 1]


@pytest.mark.parametrize(
    "make_idx",
    [
        lambda: ["b", "c"],
        lambda: np.array(["b", "c"]),
        lambda: pd.Index(["b", "c"]),
        lambda: iter(["b", "c"]),
        lambda: [False, True, True, False],
    ],
    ids=["list", "array", "index", "iterator", "bool-mask"],
)
def test_process_pandas_label_sequence_endog_idx(make_idx):
    # A sequence that does not hold only integers, such as the labels that
    # star98 passes, selects by label, as before
    data = _process_pandas_data()
    dataset = utils.process_pandas(data, endog_idx=make_idx())
    assert dataset.endog_name == ["b", "c"]
    pd.testing.assert_frame_equal(dataset.endog, data[["b", "c"]])
    assert dataset.exog_name == ["a", "d"]


def test_process_pandas_int_sequence_endog_idx_out_of_range():
    # A position past the last column raises an IndexError, as an int does.
    # It raised a KeyError for a missing label.
    data = _process_pandas_data()
    with pytest.raises(IndexError):
        utils.process_pandas(data, endog_idx=4)
    with pytest.raises(IndexError):
        utils.process_pandas(data, endog_idx=[0, 4])
