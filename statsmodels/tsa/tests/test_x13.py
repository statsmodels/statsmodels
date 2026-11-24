from statsmodels.compat.pandas import MONTH_END

import contextlib
from pathlib import Path
import shutil
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

from statsmodels.datasets import co2, macrodata
from statsmodels.tools.sm_exceptions import X13Error, X13Warning
from statsmodels.tsa.x13 import (
    _find_x12,
    x13_arima_analysis,
    x13_arima_select_order,
)

x13path = _find_x12()

pytestmark = pytest.mark.skipif(x13path is False, reason="X13/X12 not available")

# X-13 relays its own notes on the data as an X13Warning. For the monthly CO2
# data it finds a seasonal peak in the spectrum of the regARIMA residuals, and
# with the history spec of the rawspec tests it also notes the start of the
# forecast history. They describe the data and the spec, not the code under
# test, so the tests that only need a run ignore exactly these messages, and
# any other X-13 warning still shows. The quarterly data give no warning.
SEASONAL_PEAK = (
    r"WARNING: At least one visually significant seasonal peak has been found"
    r"\s+in the estimated spectrum of the regARIMA residuals\."
)
HISTORY_NOTE = (
    r"\s+NOTE: The default starting date of the forecast history analysis has"
    r"\s+been\s+used since it is later than the default starting date"
    r"\s+determined by\s+the length of the maximum seasonal filter from the"
    r"\s+seasonal adjustment\."
)
# Warned before the X13Error for a rawspec that X-13 rejects
RAWSPEC_ERRORS = "User-provided rawspec file has errors"


@contextlib.contextmanager
def ignore_x13_warning(message):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=message + r"\Z", category=X13Warning)
        yield


dta = macrodata.load_pandas().data
index = pd.period_range(start="1959Q1", end="2009Q3", freq="Q")
dta.index = index
quarterly_data = dta.dropna()

dta = co2.load_pandas().data
dta["co2"] = dta.co2.interpolate()
monthly_data = dta.resample(MONTH_END)
# change in pandas 0.18 resample is deferred object
if not isinstance(monthly_data, (pd.DataFrame, pd.Series)):
    monthly_data = monthly_data.mean()

monthly_start_data = dta.resample("MS")
if not isinstance(monthly_start_data, (pd.DataFrame, pd.Series)):
    monthly_start_data = monthly_start_data.mean()

data = (
    monthly_data,
    monthly_start_data,
    monthly_data.co2,
    monthly_start_data.co2,
    quarterly_data.realgdp,
    quarterly_data[["realgdp"]],
)
ids = (
    "monthly",
    "monthly_start",
    "monthly_co2",
    "monthly_start_co2",
    "series",
    "dataframe",
)


@pytest.fixture(params=data, ids=ids)
def dataset(request):
    return request.param


@pytest.mark.parametrize("use_numpy", [True, False])
def test_x13_arima_select_order(dataset, use_numpy):
    if use_numpy:
        index = dataset.index
        dataset = np.squeeze(np.asarray(dataset))
        start = index[0]
        if isinstance(index, pd.DatetimeIndex):
            try:
                with pd.option_context("future.infer_freq_returns_offset", True):
                    freq = index.inferred_freq.freqstr
            except (AttributeError, ImportError):
                freq = index.inferred_freq
        elif isinstance(index, pd.PeriodIndex):
            start = start.to_timestamp()
            freq = index.freq
        else:
            raise NotImplementedError
        assert freq is not None
    else:
        freq = start = None
    with ignore_x13_warning(SEASONAL_PEAK):
        res = x13_arima_select_order(dataset, start=start, freq=freq)
    assert isinstance(res.order, tuple)
    assert isinstance(res.sorder, tuple)


@pytest.mark.thread_unsafe(reason="Uses matplotlib")
@pytest.mark.matplotlib
def test_x13_arima_plot(dataset, close_figures):
    with ignore_x13_warning(SEASONAL_PEAK):
        res = x13_arima_analysis(dataset)
    res.plot()


def test_x13_arima_plot_no_pandas(dataset, close_figures):
    with ignore_x13_warning(SEASONAL_PEAK):
        res = x13_arima_analysis(dataset)
    res.plot()


@pytest.mark.smoke
def test_log_diagnostics(dataset):
    with ignore_x13_warning(SEASONAL_PEAK):
        res = x13_arima_analysis(dataset, log_diagnostics=True)
    assert isinstance(res.x13_diagnostic, dict)
    assert next(iter(res.x13_diagnostic.keys())) == "F-D8"
    assert isinstance(next(iter(res.x13_diagnostic.values())), float)


@pytest.mark.smoke
def test_log_diagnostics_false(dataset):
    with ignore_x13_warning(SEASONAL_PEAK):
        res = x13_arima_analysis(dataset, log_diagnostics=False)
    assert isinstance(res.x13_diagnostic, dict)
    assert next(iter(res.x13_diagnostic.keys())) == "F-D8"
    assert next(iter(res.x13_diagnostic.values())) == "Log diagnostics not retrieved."


def test_x13_output_files_removed(dataset, tmp_path):
    # GH 8277, the files written by x12/x13 were left in the temporary directory
    with ignore_x13_warning(SEASONAL_PEAK):
        x13_arima_analysis(dataset, tempdir=str(tmp_path))
    assert list(tmp_path.iterdir()) == []
    # also if x12/x13 reports an error
    raw_spec_file = """
seri es {
    modelspan=(,)
}
x11 {
    seasonalma=(  msr)
"""
    with (
        pytest.raises(X13Error),
        pytest.warns(UserWarning, match=RAWSPEC_ERRORS),
    ):
        x13_arima_analysis(dataset, rawspec=raw_spec_file, tempdir=str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_x13_arima_rawspec_arg():
    with pytest.raises(ValueError):
        # error because both param and rawspec are specified
        x13_arima_analysis(dataset, outlier=True, rawspec="/fake/path.spc")

    with pytest.raises(ValueError):
        # error because of fake path
        x13_arima_analysis(dataset, rawspec="/fake/path.spc")


def test_x13_arima_rawspec_run(dataset):
    # example rawspec file string
    raw_spec_file = """
series {
    modelspan=(,)
    save=(B1)
    span=(,)
    type=(flow)
}
x11 {
    seasonalma=(  msr)
    appendfcst=yes
    mode=(mult)
    print=( seasadj seasonal adjustfac)
    save=(seasadj seasonal adjustfac)
    savelog=(  alldiagnostics)
}
arima {model=(0 1 0)(1 0 1)}
transform {
    function=log
}
regression {
    savelog=(  aictest)
 }
estimate {save=mdl}
slidingspans { }
history {
    estimates=(sadj seasonal fcst)
    fixmdl=yes
}
"""

    # pass rawspec as string
    with ignore_x13_warning(SEASONAL_PEAK + HISTORY_NOTE):
        x13_arima_analysis(dataset, rawspec=raw_spec_file)

    # pass rawspec as file path
    with tempfile.NamedTemporaryFile(suffix=".spc", delete_on_close=False) as ft:
        ft.write(raw_spec_file.encode("utf8"))
        ft.seek(0)
        ft.close()

        with ignore_x13_warning(SEASONAL_PEAK + HISTORY_NOTE):
            x13_arima_analysis(dataset, rawspec=ft.name)


def test_x13_arima_rawspec_no_save(dataset):
    # example rawspec file string
    raw_spec_file = """
series {
    modelspan=(,)
    save=(B1)
    span=(,)
    type=(flow)
}
x11 {
    seasonalma=(  msr)
    appendfcst=yes
    mode=(mult)
    print=( seasadj seasonal adjustfac)
    savelog=(  alldiagnostics)
}
arima {model=(0 1 0)(1 0 1)}
transform {
    function=log
}
regression {
    savelog=(  aictest)
 }
estimate {save=mdl}
slidingspans { }
history {
    estimates=(sadj seasonal fcst)
    fixmdl=yes
}
"""

    # pass rawspec as string
    with ignore_x13_warning(SEASONAL_PEAK + HISTORY_NOTE):
        x13_arima_analysis(dataset, rawspec=raw_spec_file)

    # pass rawspec as file path
    with tempfile.NamedTemporaryFile(suffix=".spc", delete_on_close=False) as ft:
        ft.write(raw_spec_file.encode("utf8"))
        ft.seek(0)
        ft.close()

        with ignore_x13_warning(SEASONAL_PEAK + HISTORY_NOTE):
            x13_arima_analysis(dataset, rawspec=ft.name)


def test_x13_arima_rawspec_no_x11(dataset):
    # example rawspec file string
    raw_spec_file = """
series {
    modelspan=(,)
    save=(B1)
    span=(,)
    type=(flow)
}
arima {model=(0 1 0)(1 0 1)}
transform {
    function=log
}
regression {
    savelog=(  aictest)
 }
estimate {save=mdl}
slidingspans { }
history {
    estimates=(sadj seasonal fcst)
    fixmdl=yes
}
"""

    # pass rawspec as string
    with ignore_x13_warning(SEASONAL_PEAK + HISTORY_NOTE):
        x13_arima_analysis(dataset, rawspec=raw_spec_file)

    # pass rawspec as file path
    with tempfile.NamedTemporaryFile(suffix=".spc", delete_on_close=False) as ft:
        ft.write(raw_spec_file.encode("utf8"))
        ft.seek(0)
        ft.close()

        with ignore_x13_warning(SEASONAL_PEAK + HISTORY_NOTE):
            x13_arima_analysis(dataset, rawspec=ft.name)
        try:
            Path(ft.name).unlink()
        except OSError:
            # While it best ot unlink, not strictly required
            pass


def test_x13_arima_invalid_rawspec(dataset):
    # bad rawspec file string ("series" misspelled, no closing "}" on "x11")
    raw_spec_file = """
seri es {
    modelspan=(,)
}
x11 {
    seasonalma=(  msr)
"""

    # pass rawspec as string
    with (
        pytest.raises(X13Error),
        pytest.warns(UserWarning, match=RAWSPEC_ERRORS),
    ):
        x13_arima_analysis(dataset, rawspec=raw_spec_file)

    # pass rawspec as file path
    with tempfile.NamedTemporaryFile(suffix=".spc", delete_on_close=False) as ft:
        ft.write(raw_spec_file.encode("utf8"))
        ft.seek(0)
        ft.close()

        with (
            pytest.raises(X13Error),
            pytest.warns(UserWarning, match=RAWSPEC_ERRORS),
        ):
            x13_arima_analysis(dataset, rawspec=ft.name)


def test_exog_column_called_name():
    gdp = quarterly_data.realgdp
    # pulses of known size; the regressors cover the year that X-13 forecasts
    nobs = gdp.shape[0] + 4
    exog = pd.DataFrame({"name": np.zeros(nobs), "x": np.zeros(nobs)})
    exog.iloc[60, 0] = 1.0
    exog.iloc[150:153, 1] = 1.0
    effects = {"name": 1000.0, "x": -800.0}
    endog = gdp + exog.to_numpy()[: gdp.shape[0]] @ np.array([1000.0, -800.0])

    res = x13_arima_analysis(endog, exog=exog, log=False, outlier=False)

    # X-13 prints the estimate of each user-defined regressor under its name
    lines = res.results[res.results.index("User-defined") :].splitlines()[1:3]
    estimates = {line.split()[0]: float(line.split()[1]) for line in lines}
    assert list(estimates) == ["name", "x"]
    for name, effect in effects.items():
        # within about 3 standard errors (about 100) of the effect
        np.testing.assert_allclose(estimates[name], effect, rtol=0.1)


def _copy_binary(directory):
    """Copy the binary that the default lookup finds into directory"""
    found = Path(shutil.which(str(x13path)))
    # shutil.which can give the name in another case than on disk on Windows
    name = next(
        path.name
        for path in found.parent.iterdir()
        if path.name.lower() == found.name.lower()
    )
    shutil.copy2(found.parent / name, directory / name)
    return directory / name


@pytest.mark.parametrize(
    "kind", ["upper_case", "upper_extension", "path_directory", "path_binary"]
)
def test_x12path_case_and_path(kind, tmp_path, monkeypatch):
    binary_dir = tmp_path / "binary"
    binary_dir.mkdir()
    binary = _copy_binary(binary_dir)
    # the environment variables do not lead to a binary
    monkeypatch.setenv("X13PATH", str(tmp_path))
    monkeypatch.setenv("X12PATH", str(tmp_path))
    if kind == "upper_case":
        x12path = str(binary.with_name(binary.name.upper()))
    elif kind == "upper_extension":
        # x13as_ascii.EXE, as shutil.which gives it on Windows
        x12path = str(binary.with_suffix(binary.suffix.upper()))
    elif kind == "path_directory":
        x12path = binary_dir
    else:
        x12path = binary
    expected = x13_arima_analysis(quarterly_data.realgdp, x12path=str(binary_dir))

    res = x13_arima_analysis(quarterly_data.realgdp, x12path=x12path)

    np.testing.assert_allclose(
        res.seasadj.to_numpy(), expected.seasadj.to_numpy(), rtol=1e-12
    )


@pytest.mark.parametrize(
    "endog",
    [quarterly_data.realgdp.rename(5), quarterly_data[["realgdp"]].set_axis([5], axis=1)],
    ids=["series", "dataframe"],
)
def test_endog_name_not_str(endog):
    res = x13_arima_analysis(endog)

    # X-13 prints the title and the name of the series
    lines = [line.strip() for line in res.results.splitlines()]
    assert "Series Title- 5" in lines
    assert "Series Name- 5" in lines
