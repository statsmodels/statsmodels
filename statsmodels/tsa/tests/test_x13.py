from statsmodels.compat.pandas import MONTH_END

import contextlib
import os
from pathlib import Path
import re
import shutil
import tempfile
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal
import pytest

from statsmodels.datasets import co2, macrodata
from statsmodels.tools.sm_exceptions import (
    IOWarning,
    X13Error,
    X13NotFoundError,
    X13Warning,
)
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


@pytest.mark.thread_unsafe(reason="Uses matplotlib")
@pytest.mark.matplotlib
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


def test_x13_arima_rawspec_arg(dataset):
    with pytest.raises(ValueError, match="other arguments not allowed"):
        # error because both diff and rawspec are specified
        x13_arima_analysis(dataset, diff=(1, 1), rawspec="/fake/path.spc")

    with pytest.raises(ValueError, match="not valid path or spec string"):
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


def test_object_index_of_dates_quarterly():
    # An object index of datetime.date gives the results of the PeriodIndex of
    # the same quarters, with the index of endog
    gdp = quarterly_data.realgdp
    dates = pd.Index(gdp.index.to_timestamp().date, dtype=object)
    expected = x13_arima_analysis(gdp)

    res = x13_arima_analysis(gdp.set_axis(dates))

    assert res.seasadj.index.equals(dates)
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


@pytest.mark.parametrize("name", ["GDP\\new", "GDP\\tab"], ids=["newline", "tab"])
def test_rawspec_name_with_backslash(name):
    res = x13_arima_analysis(
        quarterly_data.realgdp.rename(name), rawspec="series{}\nx11{mode=mult}"
    )

    # X-13 prints the title and the name of the series as they are
    lines = [line.strip() for line in res.results.splitlines()]
    assert f"Series Title- {name}" in lines
    assert f"Series Name- {name}" in lines


def test_rawspec_endog_not_pandas():
    gdp = quarterly_data.realgdp
    rawspec = "series{}\nx11{mode=mult}"
    expected = x13_arima_analysis(gdp, rawspec=rawspec)

    res = x13_arima_analysis(
        gdp.to_numpy(), start="1959-01-01", freq="QS", rawspec=rawspec
    )

    index = pd.date_range("1959-01-01", periods=gdp.shape[0], freq="QS")
    pd.testing.assert_index_equal(res.seasadj.index, index)
    np.testing.assert_allclose(
        res.seasadj.to_numpy(), expected.seasadj.to_numpy(), rtol=1e-12
    )


# The tests below compare the results with what X-13 itself prints in its main
# output (res.results): the settings it used, its tables and their summaries.

gdp = quarterly_data.realgdp


@pytest.fixture(scope="module")
def gdp_res():
    return x13_arima_analysis(gdp)


def _x13_binary():
    """The binary that the default lookup finds, as an absolute path"""
    found = Path(shutil.which(str(x13path)))
    # On Windows, which gives the extension in the case used by PATHEXT; the name
    # on disk is the one the lookup recognizes
    return next(
        path
        for path in found.parent.iterdir()
        if path.name.lower() == found.name.lower()
    )


def _regression_estimates(results):
    """Estimate and standard error of each row of the regression model table"""
    start = results.find("Regression Model")
    if start < 0:
        return {}
    estimates = {}
    # a title, a rule, two lines of headers and a rule precede the rows
    for line in results[start:].splitlines()[5:]:
        if line.strip().startswith("---"):
            break
        match = re.match(r"\s+(\S+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+-?[\d.]+\s*$", line)
        if match:
            estimates[match.group(1)] = (float(match.group(2)), float(match.group(3)))
    return estimates


@pytest.mark.parametrize(
    "lookup",
    ["directory", "binary", "directory_named_like_binary", "X13PATH", "X12PATH", "PATH"],
)
def test_binary_lookup(lookup, gdp_res, tmp_path, monkeypatch):
    binary = _x13_binary()
    empty = tmp_path / "empty"
    empty.mkdir()
    kwargs = {}
    if lookup == "directory":
        kwargs["x12path"] = str(binary.parent)
    elif lookup == "binary":
        kwargs["x12path"] = str(binary)
    elif lookup == "directory_named_like_binary":
        # the directory is searched, not its parent as for a path to the binary
        named = tmp_path / "x13as"
        named.mkdir()
        shutil.copy2(binary, named / binary.name)
        kwargs["x12path"] = str(named)
    elif lookup == "X13PATH":
        # searched before X12PATH
        monkeypatch.setenv("X13PATH", str(binary.parent))
        monkeypatch.setenv("X12PATH", str(empty))
    elif lookup == "X12PATH":
        # searched if X13PATH is not set
        monkeypatch.delenv("X13PATH", raising=False)
        monkeypatch.setenv("X12PATH", str(binary.parent))
    else:
        # searched if neither variable is set
        monkeypatch.delenv("X13PATH", raising=False)
        monkeypatch.delenv("X12PATH", raising=False)
        path = os.environ.get("PATH", "")
        monkeypatch.setenv("PATH", os.pathsep.join([str(binary.parent), path]))

    res = x13_arima_analysis(gdp, **kwargs)

    assert_allclose(res.seasadj.to_numpy(), gdp_res.seasadj.to_numpy(), rtol=1e-12)


@pytest.mark.parametrize("lookup", ["x12path", "X13PATH", "X12PATH"])
def test_binary_lookup_order(lookup, tmp_path, monkeypatch):
    binary_dir = str(_x13_binary().parent)
    kwargs = {}
    if lookup == "x12path":
        # the environment variables are not searched if x12path is given
        monkeypatch.setenv("X13PATH", binary_dir)
        monkeypatch.setenv("X12PATH", binary_dir)
        kwargs["x12path"] = str(tmp_path)
    elif lookup == "X13PATH":
        # X12PATH is not searched if X13PATH is set
        monkeypatch.setenv("X13PATH", str(tmp_path))
        monkeypatch.setenv("X12PATH", binary_dir)
    else:
        # the PATH is not searched if X12PATH is set
        monkeypatch.delenv("X13PATH", raising=False)
        monkeypatch.setenv("X12PATH", str(tmp_path))
        path = os.environ.get("PATH", "")
        monkeypatch.setenv("PATH", os.pathsep.join([binary_dir, path]))

    with pytest.raises(X13NotFoundError):
        x13_arima_analysis(gdp, **kwargs)


def test_x13_warning_relayed():
    with pytest.warns(X13Warning, match=SEASONAL_PEAK):
        x13_arima_analysis(monthly_data.co2)


@pytest.mark.parametrize(
    "endog, log",
    [
        (gdp, True),
        (gdp, False),
        (gdp, None),
        # positive, but X-13 prefers no transformation
        (quarterly_data.infl + 20, None),
        (monthly_data.co2, True),
    ],
    ids=["log", "no_log", "auto_log", "auto_no_log", "monthly_log"],
)
def test_decomposition_identity(endog, log):
    with ignore_x13_warning(SEASONAL_PEAK):
        res = x13_arima_analysis(endog, log=log)
    seasadj = res.seasadj.to_numpy()
    trend = res.trend.to_numpy()
    irregular = res.irregular.to_numpy()
    if log is None:
        # X-13 chooses the transformation and says which
        choice = re.search(r"AICC .* prefers (log|no) transformation", res.results)
        log = choice.group(1) == "log"

    if log:
        # multiplicative decomposition
        assert_allclose(irregular, seasadj / trend, rtol=1e-12)
        assert not np.allclose(irregular, seasadj - trend)
    else:
        # The tables are saved with 15 significant digits, and the additive
        # irregular is the difference of values of the size of the series
        atol = 1e-12 * np.abs(seasadj).max()
        assert_allclose(irregular, seasadj - trend, rtol=0, atol=atol)
        assert not np.allclose(irregular, seasadj / trend)


TABLE_TITLES = {
    "seasadj": "D 11  Final seasonally adjusted data",
    "trend": "D 12  Final trend cycle",
    "irregular": "D 13  Final irregular component",
}


def _printed_summary(results, title):
    """The statistics X-13 prints below a table, with their number of decimals"""
    summary = results[results.index("Table Total-", results.index(title)) :]
    stats = {}
    for label in ("Total", "Mean", "Std. Dev.", "Min", "Max"):
        text = re.search(re.escape(label) + r"\s*-\s*(-?[\d.]+)", summary).group(1)
        stats[label] = (float(text), len(text.partition(".")[2]))
    return stats


@pytest.mark.parametrize(
    "endog, log",
    [(gdp, True), (gdp, False), (monthly_data.co2, True)],
    ids=["log", "no_log", "monthly_log"],
)
def test_tables_match_printed_summary(endog, log):
    with ignore_x13_warning(SEASONAL_PEAK):
        res = x13_arima_analysis(endog, log=log)

    for name, title in TABLE_TITLES.items():
        table = getattr(res, name)
        assert table.name == name
        assert_index_equal(table.index, endog.index)
        # X-13 prints a multiplicative irregular in percent
        values = table.to_numpy() * (100.0 if log and name == "irregular" else 1.0)
        computed = {
            "Total": values.sum(),
            "Mean": values.mean(),
            "Std. Dev.": values.std(),
            "Min": values.min(),
            "Max": values.max(),
        }
        for label, (printed, decimals) in _printed_summary(res.results, title).items():
            # rounded to the printed decimals, a little more for a value at a half
            atol = 0.51 * 10.0**-decimals
            assert_allclose(computed[label], printed, rtol=0, atol=atol)


def _echoed(results, setting):
    return int(re.search(re.escape(setting) + r"\s*:?\s*(\d+)", results).group(1))


@pytest.mark.parametrize(
    "kwargs, settings",
    [
        (
            {"maxorder": (3, 2), "maxdiff": (1, 1)},
            {
                "Maximum order for regular ARMA parameters": 3,
                "Maximum order for seasonal ARMA parameters": 2,
                "Maximum order for regular differencing": 1,
                "Maximum order for seasonal differencing": 1,
            },
        ),
        (
            {"maxorder": (1, 1), "maxdiff": None, "diff": (2, 0)},
            {
                "Maximum order for regular ARMA parameters": 1,
                "Maximum order for seasonal ARMA parameters": 1,
                "Regular differencing set to": 2,
                "Seasonal differencing set to": 0,
            },
        ),
    ],
    ids=["maxdiff", "diff"],
)
def test_automdl_settings(kwargs, settings):
    res = x13_arima_select_order(gdp, **kwargs)

    for setting, value in settings.items():
        assert _echoed(res.results, setting) == value
    p, _, q = res.order
    seasonal_p, _, seasonal_q = res.sorder
    assert max(p, q) <= kwargs["maxorder"][0]
    assert max(seasonal_p, seasonal_q) <= kwargs["maxorder"][1]


@pytest.mark.parametrize(
    "endog, kwargs",
    [
        (monthly_data.co2, {}),
        (gdp, {}),
        (gdp, {"maxdiff": None, "diff": (2, 0)}),
        (gdp, {"log": False, "outlier": False}),
    ],
    ids=["monthly", "quarterly", "fixed_diff", "no_log"],
)
def test_select_order_matches_output(endog, kwargs):
    with ignore_x13_warning(SEASONAL_PEAK):
        res = x13_arima_select_order(endog, **kwargs)

    final = re.search(r"Final automatic model choice : (.*)", res.results).group(1)
    orders = [
        tuple(int(order) for order in group.split())
        for group in re.findall(r"\(([\d ]+)\)", final)
    ]
    assert res.order == orders[0]
    # the seasonal part is not shown for a nonseasonal model
    assert res.sorder == (orders[1] if len(orders) == 2 else (0, 0, 0))
    # a mean is included if the estimated regression model has a constant
    assert res.include_mean is ("Constant" in _regression_estimates(res.results))
    assert isinstance(res.stdout, bytes)


def test_select_order_airline_co2():
    with ignore_x13_warning(SEASONAL_PEAK):
        res = x13_arima_select_order(monthly_data.co2)

    assert res.order == (0, 1, 1)
    assert res.sorder == (0, 1, 1)
    assert res.include_mean is False


@pytest.mark.parametrize("forecast_periods, number", [(None, 4), (5, 5), (9, 9)])
def test_forecast_periods(forecast_periods, number):
    res = x13_arima_analysis(gdp, forecast_periods=forecast_periods)

    pattern = r"FORECASTING\s+Origin\s+(\S+)\s+Number\s+(\d+)"
    origin, produced = re.search(pattern, res.results).groups()
    # from the last observation, and one year if not given
    assert origin == "2009.3"
    assert int(produced) == number


def test_forecast_periods_zero():
    res = x13_arima_analysis(gdp, forecast_periods=0)

    assert "FORECASTING" not in res.results


@pytest.mark.parametrize("outlier", [True, False])
def test_outlier_detection(outlier):
    endog = gdp.copy()
    # an additive outlier of known size in 1989Q1
    endog.iloc[120] += 1500.0

    res = x13_arima_analysis(endog, outlier=outlier, log=False)

    estimates = _regression_estimates(res.results)
    outliers = {
        key: value for key, value in estimates.items() if key[:2] in ("AO", "LS", "TC")
    }
    if outlier:
        assert "AO1989.1" in outliers
        assert_allclose(outliers["AO1989.1"][0], 1500.0, rtol=0.1)
    else:
        assert outliers == {}


@pytest.mark.parametrize("trading", [True, False])
def test_trading_day_regressors(trading):
    res = x13_arima_analysis(gdp, trading=trading)

    estimates = _regression_estimates(res.results)
    # Sunday is derived from the other days
    assert {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat"}.issubset(estimates) is trading


def _exog_pulses(extra):
    nobs = gdp.shape[0] + extra
    index = pd.period_range("1959Q1", periods=nobs, freq="Q")
    a = np.zeros(nobs)
    a[60] = 1.0
    b = np.zeros(nobs)
    b[150:153] = 1.0
    return pd.DataFrame({"a": a, "b": b}, index=index)


@pytest.mark.parametrize("columns", [["a", "b"], ["b", "a"], "a"])
def test_exog_effects(columns):
    # the regressors cover the year that X-13 forecasts
    exog = _exog_pulses(4)
    names = [columns] if isinstance(columns, str) else columns
    effects = {"a": 1000.0, "b": -800.0}
    endog = gdp.copy()
    for name in names:
        endog += effects[name] * exog[name].to_numpy()[: gdp.shape[0]]

    res = x13_arima_analysis(endog, exog=exog[columns], log=False, outlier=False)

    # each estimate is within 3 standard errors (about 100) of its effect, while a
    # regressor read in the wrong order would have no or the other effect
    estimates = _regression_estimates(res.results)
    for name in names:
        assert_allclose(estimates[name][0], effects[name], rtol=0.1)


def test_exog_must_cover_forecasts():
    exog = _exog_pulses(0)["a"]
    with pytest.raises(X13Error, match="regression variables end date"):
        x13_arima_analysis(gdp, exog=exog)
    with pytest.raises(X13Error, match="regression variables end date"):
        x13_arima_analysis(gdp, exog=_exog_pulses(4)["a"], forecast_periods=6)

    # without forecasts, the regressors only need to cover the series
    res = x13_arima_analysis(gdp, exog=exog, forecast_periods=0)

    assert "a" in _regression_estimates(res.results)


def test_retspec(gdp_res):
    res = x13_arima_analysis(gdp, retspec=True)

    assert res.spec == x13_arima_analysis(gdp, speconly=True)
    assert_allclose(res.seasadj.to_numpy(), gdp_res.seasadj.to_numpy(), rtol=1e-12)
    assert not hasattr(gdp_res, "spec")


@pytest.mark.parametrize("form", ["list", "ndarray", "dataframe"])
def test_endog_forms(form, gdp_res):
    if form == "dataframe":
        endog = gdp.to_frame()
        res = x13_arima_analysis(endog)
        assert_frame_equal(res.observed, endog)
        index = gdp.index
    else:
        endog = gdp.tolist() if form == "list" else gdp.to_numpy()
        res = x13_arima_analysis(endog, start="1959-01-01", freq="QS")
        index = pd.date_range("1959-01-01", periods=gdp.shape[0], freq="QS")
        assert_index_equal(res.observed.index, index)
        assert_equal(res.observed.to_numpy(), gdp.to_numpy())

    for name in ("seasadj", "trend", "irregular"):
        table = getattr(res, name)
        assert_index_equal(table.index, index)
        expected = getattr(gdp_res, name).to_numpy()
        assert_allclose(table.to_numpy(), expected, rtol=1e-12)


def test_log_diagnostics_match_printed_tables():
    # X-13 also prints the diagnostics that it saves to the log in tables F 2
    # and F 3 of the main output
    rawspec = "series{}\nx11{ print=(f2 f3) savelog=(fd8 m7 q) }"
    res = x13_arima_analysis(gdp, rawspec=rawspec, log_diagnostics=True)

    patterns = {
        "F-D8": r"F-test for stable seasonality from Table D 8\.\s*:\s*([\d.]+)",
        "M07": r"M7\s*=\s*([\d.]+)",
        "Q": r"\*\*\* [A-Z ]+ \*\*\* at the level\s+([\d.]+)",
    }
    assert list(res.x13_diagnostic) == list(patterns)
    for key, pattern in patterns.items():
        text = re.search(pattern, res.results).group(1)
        # the log and the tables may round to different decimals
        atol = 0.51 * 10.0 ** -len(text.partition(".")[2])
        assert_allclose(res.x13_diagnostic[key], float(text), rtol=0, atol=atol)


@pytest.mark.thread_unsafe(reason="Uses matplotlib")
@pytest.mark.matplotlib
def test_plot_contents(gdp_res, close_figures):
    fig = gdp_res.plot()

    labels = ["Observed", "Seas. Adjusted", "Trend", "Irregular"]
    assert [ax.get_ylabel() for ax in fig.axes] == labels
    tables = [gdp_res.observed, gdp_res.seasadj, gdp_res.trend, gdp_res.irregular]
    for ax, table in zip(fig.axes, tables, strict=True):
        (line,) = ax.get_lines()
        assert_allclose(line.get_ydata(), table.to_numpy(), rtol=1e-12)


def test_output_file_removed_by_other_process(gdp_res, tmp_path, monkeypatch):
    unlink = Path.unlink

    def unlink_then_fail(self, *args, **kwargs):
        # the file is gone, but deleting it fails, as if another process had
        # removed it first
        unlink(self, *args, **kwargs)
        if self.suffix == ".out":
            raise FileNotFoundError(f"{self} was removed")

    monkeypatch.setattr(Path, "unlink", unlink_then_fail)

    # tempdir can be a Path
    with warnings.catch_warnings():
        warnings.simplefilter("error", IOWarning)
        res = x13_arima_analysis(gdp, tempdir=tmp_path)

    assert list(tmp_path.iterdir()) == []
    assert_allclose(res.seasadj.to_numpy(), gdp_res.seasadj.to_numpy(), rtol=1e-12)


@pytest.mark.parametrize("form", ["series", "dataframe"])
def test_endog_without_dates(form):
    gdp = quarterly_data.realgdp
    endog = gdp.reset_index(drop=True)
    if form == "dataframe":
        endog = endog.to_frame()
    expected = x13_arima_analysis(gdp.to_numpy(), start="1959-01-01", freq="QS")

    res = x13_arima_analysis(endog, start="1959-01-01", freq="QS")

    # the same results as for the values without an index, with the dates
    # built from start and freq
    index = pd.date_range("1959-01-01", periods=gdp.shape[0], freq="QS")
    pd.testing.assert_index_equal(res.observed.index, index)
    np.testing.assert_allclose(
        np.squeeze(res.observed.to_numpy()), gdp.to_numpy(), rtol=1e-12
    )
    for name in ("seasadj", "trend", "irregular"):
        table = getattr(res, name)
        pd.testing.assert_index_equal(table.index, index)
        np.testing.assert_allclose(
            table.to_numpy(), getattr(expected, name).to_numpy(), rtol=1e-12
        )
