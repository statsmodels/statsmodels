"""Regression tests for x13 file handling that do not need the X-13 binary.

``test_x13`` skips its whole module when x12a/x13as is absent. These checks only
exercise the spec-building path (``speconly=True`` returns before the binary is
called), the clean-up of the temporary files with ``run_spec`` replaced by a
stub, and the errors raised when no binary is found, so they live here and run
everywhere.
"""

from statsmodels.compat.pandas import MONTH_END, QUARTER_END, YEAR_END
from statsmodels.compat.python import PYTHON_IMPL_WASM

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pandas as pd
import pytest

import statsmodels
from statsmodels.tools.sm_exceptions import IOWarning, X13Error, X13NotFoundError
from statsmodels.tsa import x13

# Non-ASCII text that is not valid in an ASCII locale. The capital A with acute
# has the UTF-8 bytes c3 81, and 0x81 is undefined in cp1252, so decoding with
# the Windows locale codec fails or yields mojibake instead of this text.
NON_ASCII_TITLE = "café naïve Á"
SPEC_TEMPLATE = 'series{title="%s"}\n'

# Run in a fresh interpreter whose locale encoding is not UTF-8. The code is
# ASCII only and the result is written to a file as JSON with escaped
# characters, so nothing in the child depends on the locale except the read
# under test. The result is not written to stdout, because an editable install
# rebuilds statsmodels when it is imported and prints the progress of the
# build, with paths that need not be ASCII.
CHILD_CODE = """
import json
import sys
import threading

import pandas as pd

from statsmodels.tsa import x13

endog = pd.Series(
    range(24), index=pd.period_range("2000-01", periods=24, freq="M")
)
spec = x13.x13_arima_analysis(endog, rawspec=sys.argv[1], speconly=True)
with open(sys.argv[2], "w", encoding="ascii") as fh:
    fh.write(json.dumps(spec))
"""


def _monthly_endog():
    return pd.Series(range(24), index=pd.period_range("2000-01", periods=24, freq="M"))


@pytest.fixture
def no_x12(monkeypatch):
    monkeypatch.setattr(x13, "_check_x12", lambda *args, **kwargs: "x12dummy")


@pytest.mark.skipif(PYTHON_IMPL_WASM, reason="Emscripten does not support processes")
def test_rawspec_file_utf8_in_non_utf8_locale(tmp_path):
    # Python 3.10-3.14 decode text files with the locale encoding unless told
    # otherwise: cp1252 on Windows, ASCII for LC_ALL=C on Linux and macOS when
    # UTF-8 mode and locale coercion are off. A bare ``open`` in the rawspec
    # branch then fails or garbles the text. The locale cannot be changed in
    # the running process, so the call is made in a subprocess.
    specfile = tmp_path / "custom.spc"
    specfile.write_bytes((SPEC_TEMPLATE % NON_ASCII_TITLE).encode("utf-8"))

    env = os.environ.copy()
    env.update({"PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0", "LC_ALL": "C"})
    env.pop("PYTHONIOENCODING", None)
    root = str(Path(statsmodels.__file__).parent.parent)
    env["PYTHONPATH"] = os.pathsep.join(
        [root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )

    # Relative names keep the arguments ASCII whatever the temp directory is.
    result = tmp_path / "spec.json"
    res = subprocess.run(
        [sys.executable, "-c", CHILD_CODE, specfile.name, result.name],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    spec = json.loads(result.read_text(encoding="ascii"))
    assert f'title="{NON_ASCII_TITLE}"' in spec


def test_rawspec_invalid_path(tmp_path):
    missing = tmp_path / "missing.spc"

    with pytest.raises(ValueError, match="not valid path or spec string"):
        x13.x13_arima_analysis(_monthly_endog(), rawspec=str(missing), speconly=True)


def test_rawspec_path_object(tmp_path):
    specfile = tmp_path / "custom.spc"
    specfile.write_bytes((SPEC_TEMPLATE % NON_ASCII_TITLE).encode("utf-8"))

    spec = x13.x13_arima_analysis(_monthly_endog(), rawspec=specfile, speconly=True)
    expected = x13.x13_arima_analysis(
        _monthly_endog(), rawspec=str(specfile), speconly=True
    )

    assert f'title="{NON_ASCII_TITLE}"' in spec
    assert spec == expected

    with pytest.raises(ValueError, match="not valid path or spec string"):
        x13.x13_arima_analysis(
            _monthly_endog(), rawspec=tmp_path / "missing.spc", speconly=True
        )


def test_rawspec_file_utf8_bom(tmp_path):
    specfile = tmp_path / "bom.spc"
    specfile.write_bytes(b"\xef\xbb\xbf" + (SPEC_TEMPLATE % "bom").encode("utf-8"))

    spec = x13.x13_arima_analysis(
        _monthly_endog(), rawspec=str(specfile), speconly=True
    )

    assert "\ufeff" not in spec
    assert 'title="bom"' in spec


def test_rawspec_file_not_utf8(tmp_path):
    specfile = tmp_path / "latin1.spc"
    specfile.write_bytes((SPEC_TEMPLATE % NON_ASCII_TITLE).encode("latin-1"))

    with pytest.raises(ValueError, match="UTF-8") as exc_info:
        x13.x13_arima_analysis(_monthly_endog(), rawspec=str(specfile), speconly=True)

    assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)
    assert str(specfile) in str(exc_info.value)


# The files that x12/x13 writes next to the output name. The first four are
# read by x13_arima_analysis, the others are written for some specs only.
OUTPUT_EXTENSIONS = ("err", "out", "log", "d11", "d12", "d13", "mdl", "udg", "html")


class _FinishedProcess:
    def __init__(self):
        self.stdout = io.BytesIO(b"")

    def wait(self):
        return 0


@pytest.fixture
def stub_run_spec(monkeypatch, no_x12):
    """Replace run_spec by a stub that writes the files x12/x13 writes."""

    def install(error=""):
        # two header lines, then the date and the value
        table = "date\tvalue\n----\t-----\n" + "".join(
            f"{2000 + i // 12}{i % 12 + 1:02d}\t{i}.0\n" for i in range(24)
        )

        def run_spec(x12path, specpath, outname=None, **kwargs):
            for extension in OUTPUT_EXTENSIONS:
                content = table if extension.startswith("d1") else ""
                if extension == "err":
                    content = error
                Path(f"{outname}.{extension}").write_text(content, encoding="utf-8")
            return _FinishedProcess()

        monkeypatch.setattr(x13, "run_spec", run_spec)

    return install


@pytest.mark.parametrize("dirname", ["plain", "we[ird] dir"])
def test_output_files_removed(stub_run_spec, tmp_path, dirname):
    # GH 8277, the files written by x12/x13 were left in the temporary directory
    stub_run_spec()
    tempdir = tmp_path / dirname
    tempdir.mkdir()
    (tempdir / "keep.out").write_text("not written by x13")

    res = x13.x13_arima_analysis(
        _monthly_endog(), tempdir=str(tempdir), log_diagnostics=False
    )

    assert res.seasadj.shape == (24,)
    assert [path.name for path in tempdir.iterdir()] == ["keep.out"]


def test_output_files_removed_on_error(stub_run_spec, tmp_path):
    stub_run_spec(error="spc:\n ERROR: stub error")

    with pytest.raises(X13Error, match="stub error"):
        x13.x13_arima_analysis(
            _monthly_endog(), tempdir=str(tmp_path), log_diagnostics=False
        )

    assert list(tmp_path.iterdir()) == []


def test_output_file_not_deleted_warns(stub_run_spec, tmp_path, monkeypatch):
    stub_run_spec()
    unlink = Path.unlink

    def failing_unlink(self, *args, **kwargs):
        if self.suffix == ".d11":
            raise PermissionError(f"cannot delete {self}")
        return unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)

    with pytest.warns(IOWarning, match=r"Failed to delete resource .*\.d11"):
        x13.x13_arima_analysis(
            _monthly_endog(), tempdir=str(tmp_path), log_diagnostics=False
        )

    assert [path.suffix for path in tmp_path.iterdir()] == [".d11"]


@pytest.mark.parametrize(
    "exog, names, data",
    [
        (
            pd.DataFrame([[0.0, 10.0], [1.0, 12.0], [2.0, 11.0]]),
            "x1 x2",
            "0.0 10.0 1.0 12.0 2.0 11.0",
        ),
        # a constant column is named const
        (
            pd.DataFrame({3: [0.0, 1.0, 2.0], 5: [1.0, 1.0, 1.0]}),
            "x1 const",
            "0.0 1.0 1.0 1.0 2.0 1.0",
        ),
        # names that are not all str
        (
            pd.DataFrame({"a": [0.0, 1.0, 2.0], 1: [10.0, 12.0, 11.0]}),
            "x1 x2",
            "0.0 10.0 1.0 12.0 2.0 11.0",
        ),
    ],
    ids=["int_columns", "constant", "mixed_columns"],
)
# DataFrame.var with positional arguments warns on pandas 3
@pytest.mark.filterwarnings("error")
def test_spec_exog_columns_not_str(exog, names, data):
    spec = x13.x13_arima_analysis(_monthly_endog(), exog=exog, speconly=True)

    tokens = spec.split()
    start = tokens.index("regression{")
    expected = f"regression{{ user = ({names}) data = ({data}) }}"
    assert tokens[start : tokens.index("}", start) + 1] == expected.split()


@pytest.mark.parametrize(
    "exog, names, data",
    [
        (
            pd.DataFrame({"name": [0.0, 1.0, 2.0], "x": [10.0, 11.0, 12.0]}),
            "name x",
            "0.0 10.0 1.0 11.0 2.0 12.0",
        ),
        (
            pd.DataFrame({"x": [10.0, 11.0, 12.0], "name": [0.0, 1.0, 2.0]}),
            "x name",
            "10.0 0.0 11.0 1.0 12.0 2.0",
        ),
        (pd.Series([0.0, 1.0, 2.0], name="name"), "name", "0.0 1.0 2.0"),
        (pd.Series([0.0, 1.0, 2.0], name="shock"), "shock", "0.0 1.0 2.0"),
    ],
    ids=["frame_name_first", "frame_name_last", "series_name", "series"],
)
def test_spec_exog_column_called_name(exog, names, data):
    spec = x13.x13_arima_analysis(_monthly_endog(), exog=exog, speconly=True)

    # the user regressors are named by the columns, and their values are
    # given observation by observation
    tokens = spec.split()
    start = tokens.index("regression{")
    expected = f"regression{{ user = ({names}) data = ({data}) }}"
    assert tokens[start : tokens.index("}", start) + 1] == expected.split()


# A stand-in for the X-13 binary. The lookup runs it without arguments, and
# x13_arima_analysis with the spec and the name of the output files, for which
# it writes the files that are read back. It prints its own path, which is
# returned as stdout, so a test can tell which binary was used.
FAKE_X13 = r"""#!/bin/sh
if [ "$#" -eq 2 ]; then
    : > "$2.err"
    echo "Final automatic model choice : (0 1 1)(0 1 1)" > "$2.out"
    for table in d11 d12 d13; do
        printf 'date\tvalue\n----\t-----\n' > "$2.$table"
        i=0
        while [ "$i" -lt 24 ]; do
            printf '%s\t%s.0\n' "$i" "$i" >> "$2.$table"
            i=$((i + 1))
        done
    done
fi
echo "$0"
"""

fake_binaries = pytest.mark.skipif(
    sys.platform == "win32" or PYTHON_IMPL_WASM,
    reason="The stand-in binaries are shell scripts",
)


def _fake_binaries(directory, *names):
    directory.mkdir(exist_ok=True)
    for name in names:
        binary = directory / name
        binary.write_text(FAKE_X13, encoding="ascii", newline="\n")
        binary.chmod(0o755)
    return directory


@fake_binaries
@pytest.mark.parametrize("ext", ["", ".exe"], ids=["posix_names", "windows_names"])
@pytest.mark.parametrize("prefer_x13", [True, False])
@pytest.mark.parametrize(
    "func", [x13.x13_arima_analysis, x13.x13_arima_select_order]
)
def test_prefer_x13_environment(tmp_path, monkeypatch, func, prefer_x13, ext):
    # Each variable names a directory with one binary; prefer_x13 decides which
    # variable is used
    x13_dir = _fake_binaries(tmp_path / "x13", f"x13as{ext}")
    x12_dir = _fake_binaries(tmp_path / "x12", f"x12a{ext}")
    monkeypatch.setenv("X13PATH", str(x13_dir))
    monkeypatch.setenv("X12PATH", str(x12_dir))

    res = func(_monthly_endog(), prefer_x13=prefer_x13)

    expected = x13_dir / f"x13as{ext}" if prefer_x13 else x12_dir / f"x12a{ext}"
    assert res.stdout.decode().strip() == str(expected)


@fake_binaries
@pytest.mark.parametrize("ext", ["", ".exe"], ids=["posix_names", "windows_names"])
@pytest.mark.parametrize("x13_name", ["x13as", "x13as_ascii", "x13as_html"])
@pytest.mark.parametrize("variable", ["X12PATH", "X13PATH"])
def test_prefer_x13_false_binary_order(tmp_path, monkeypatch, variable, x13_name, ext):
    # With prefer_x13=False, x12a is used if the directory also has an x13as
    # binary, also if only X13PATH is set
    directory = _fake_binaries(tmp_path, f"{x13_name}{ext}", f"x12a{ext}")
    monkeypatch.delenv("X13PATH", raising=False)
    monkeypatch.delenv("X12PATH", raising=False)
    monkeypatch.setenv(variable, str(directory))

    res = x13.x13_arima_analysis(_monthly_endog(), prefer_x13=False)

    assert res.stdout.decode().strip() == str(directory / f"x12a{ext}")


@fake_binaries
@pytest.mark.parametrize("prefer_x13", [True, False])
@pytest.mark.parametrize("binary", [False, True], ids=["directory", "binary"])
def test_prefer_x13_not_used_with_x12path(tmp_path, monkeypatch, binary, prefer_x13):
    # prefer_x13 does nothing if x12path is given, as documented, and the
    # environment variables are not used
    directory = _fake_binaries(tmp_path / "bin", "x13as", "x12a")
    other = _fake_binaries(tmp_path / "other", "x13as", "x12a")
    monkeypatch.setenv("X13PATH", str(other))
    monkeypatch.setenv("X12PATH", str(other))
    x12path = directory / "x13as" if binary else directory

    res = x13.x13_arima_analysis(
        _monthly_endog(), x12path=x12path, prefer_x13=prefer_x13
    )

    assert res.stdout.decode().strip() == str(directory / "x13as")


@fake_binaries
def test_prefer_x13_false_falls_back(tmp_path, monkeypatch):
    # Without a binary in X12PATH, prefer_x13=False runs the binary in X13PATH,
    # which it ran before prefer_x13 was honoured
    x13_dir = _fake_binaries(tmp_path / "x13", "x13as")
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("X13PATH", str(x13_dir))
    monkeypatch.setenv("X12PATH", str(empty))

    res = x13.x13_arima_analysis(_monthly_endog(), prefer_x13=False)

    assert res.stdout.decode().strip() == str(x13_dir / "x13as")
    # but not if x12path is given
    with pytest.raises(x13.X13NotFoundError, match="x12a and x13as not found"):
        x13.x13_arima_analysis(_monthly_endog(), x12path=empty, prefer_x13=False)


@pytest.mark.parametrize("lookup", ["environment", "x12path"])
def test_speconly_without_binary(tmp_path, monkeypatch, lookup):
    # speconly returns the spec without looking up the binary, so neither the
    # environment variables nor x12path need to lead to one. The spec is
    # written out by hand from the X-13ARIMA-SEATS syntax; X-13 reads specs in
    # free format, so the comparison ignores the whitespace between tokens.
    monkeypatch.setenv("X13PATH", str(tmp_path))
    monkeypatch.setenv("X12PATH", str(tmp_path))
    kwargs = {"x12path": tmp_path / "x13as"} if lookup == "x12path" else {}
    endog = pd.Series(
        [1.5, 2.25, 3.0, 3.75, 4.5, 5.25, 6.0, 6.75],
        index=pd.period_range("1959Q3", periods=8, freq="Q"),
        name="gdp",
    )

    spec = x13.x13_arima_analysis(endog, speconly=True, **kwargs)

    expected = """
    Series {
        data=(1.5 2.25 3.0 3.75 4.5 5.25 6.0 6.75)
        appendbcst=no
        appendfcst=no
        period=4
        start=1959.3
        title="gdp"
        name="gdp"
    }
    transform{function=auto}
    outlier{}
    automdl{
        maxorder = (2 1)
        maxdiff = (2 1)
    }
    x11{ save=(d11 d12 d13)
         savelog=(fd8 m7 q)}
    """
    assert spec.split() == expected.split()
    # the lookup still happens when the analysis is run
    with pytest.raises(x13.X13NotFoundError, match="x12a and x13as not found"):
        x13.x13_arima_analysis(endog, **kwargs)


QUARTER_STARTS = list(pd.date_range("2000-01-01", periods=8, freq="QS"))
QUARTER_ENDS = [pd.Timestamp(1959, 7, 1) + pd.offsets.QuarterEnd(i) for i in range(1, 9)]
MONTH_ENDS = [pd.Timestamp(2000, 5, 1) + pd.offsets.MonthEnd(i) for i in range(1, 25)]


@pytest.mark.parametrize(
    "dates, freq, start",
    [
        ([ts.date() for ts in QUARTER_STARTS], "Q", "2000.1"),
        ([ts.date() for ts in QUARTER_ENDS], "Q", "1959.3"),
        ([ts.to_pydatetime() for ts in QUARTER_ENDS], "Q", "1959.3"),
        ([ts.date() for ts in MONTH_ENDS], "M", "2000.5"),
    ],
    ids=[
        "date_quarter_starts",
        "date_quarter_ends",
        "datetime_quarter_ends",
        "date_month_ends",
    ],
)
def test_object_index_of_dates(dates, freq, start):
    # An object index of datetime.date or datetime.datetime gives the period
    # and the start of the series, the same spec as a PeriodIndex of the dates
    index = pd.Index(dates, dtype=object)
    assert index.dtype == object
    endog = pd.Series(range(len(dates)), index=index, dtype=float, name="y")
    periods = endog.set_axis(pd.PeriodIndex(dates, freq=freq))

    spec = x13.x13_arima_analysis(endog, speconly=True)

    period = {"Q": 4, "M": 12}[freq]
    assert f"period={period}" in spec.split()
    assert f"start={start}" in spec.split()
    assert spec == x13.x13_arima_analysis(periods, speconly=True)


@pytest.mark.parametrize(
    "offset, freq",
    [
        (pd.offsets.BMonthEnd(), "M"),
        (pd.offsets.BMonthBegin(), "M"),
        (pd.offsets.BQuarterEnd(startingMonth=12), "Q"),
        (pd.offsets.BQuarterBegin(startingMonth=1), "Q"),
    ],
    ids=["BME", "BMS", "BQE", "BQS"],
)
def test_business_month_quarter_freq(offset, freq):
    # Business month and quarter frequencies raised "Only monthly and
    # quarterly periods are supported". The spec is that of the same values
    # with a monthly or quarterly PeriodIndex
    values = [float(v) for v in range(1, 25)]
    index = pd.date_range("2000-01-01", periods=24, freq=offset)
    endog = pd.Series(values, index=index, name="y")
    periods = pd.period_range(index[0], periods=24, freq=freq)
    expected = x13.x13_arima_analysis(
        pd.Series(values, index=periods, name="y"), speconly=True
    )
    spec = x13.x13_arima_analysis(endog, speconly=True)
    assert spec.split() == expected.split()
    assert f"period={12 if freq == 'M' else 4}" in spec.split()


@pytest.mark.parametrize(
    "offset",
    [
        pd.offsets.Week(weekday=6),
        pd.offsets.Day(),
        pd.offsets.BDay(),
        pd.offsets.BYearEnd(),
    ],
    ids=["W", "D", "B", "BYE"],
)
def test_freq_not_monthly_or_quarterly(offset):
    index = pd.date_range("2000-01-01", periods=24, freq=offset)
    endog = pd.Series([float(v) for v in range(24)], index=index)
    with pytest.raises(ValueError, match="Only monthly and quarterly periods"):
        x13.x13_arima_analysis(endog, speconly=True)


@pytest.mark.parametrize(
    "make_endog, name",
    [
        (lambda: _monthly_endog().rename(5), "5"),
        (lambda: _monthly_endog().rename(2.5), "2.5"),
        (lambda: pd.DataFrame({7: _monthly_endog()}), "7"),
        (lambda: _monthly_endog().rename("gdp"), "gdp"),
        (lambda: _monthly_endog().rename(None), "Unnamed Series"),
    ],
    ids=["int", "float", "dataframe_int_column", "str", "none"],
)
def test_spec_endog_name(no_x12, make_endog, name):
    spec = x13.x13_arima_analysis(make_endog(), speconly=True)

    data = " ".join(str(value) for value in range(24))
    expected = f"""
    Series {{
        data=({data})
        appendbcst=no
        appendfcst=no
        period=12
        start=2000.1
        title="{name}"
        name="{name}"
    }}
    """
    assert spec[: spec.index("}") + 1].split() == expected.split()


@pytest.mark.parametrize(
    "name",
    ["GDP\\new", "GDP\\data", "GDP\\tab", "GDP\\\\x", "GDP\\g<0>"],
    ids=["newline_escape", "bad_escape", "tab_escape", "double", "group"],
)
def test_rawspec_name_with_backslash(no_x12, name):
    endog = _monthly_endog().rename(name)

    spec = x13.x13_arima_analysis(endog, rawspec="series{}", speconly=True)

    # the name is copied as it is, as in the spec without rawspec
    data = " ".join(str(value) for value in range(24))
    expected = f"""
    Series {{
        data=({data})
        appendbcst=no
        appendfcst=no
        period=12
        start=2000.1
        title="{name}"
        name="{name}"
    }}
    x11 {{save=(d11 d12 d13)}}
    """
    assert spec.split() == expected.split()
    plain = x13.x13_arima_analysis(endog, speconly=True)
    assert f'title="{name}"' in plain.split()


@pytest.mark.parametrize(
    "dates",
    [
        # month starts without June 2000
        pd.date_range("2000-01-01", periods=25, freq="MS").delete(5),
        # the 15th of each month
        [pd.Timestamp(2000 + i // 12, i % 12 + 1, 15) for i in range(24)],
        # quarter starts without the third quarter of 2001
        pd.date_range("2000-01-01", periods=25, freq="QS").delete(6),
    ],
    ids=["month_missing", "mid_month", "quarter_missing"],
)
@pytest.mark.parametrize("func", [x13.x13_arima_analysis, x13.x13_arima_select_order])
def test_datetime_index_without_freq(no_x12, func, dates):
    # pandas cannot infer a frequency from these dates, so neither the period
    # nor the start of the series is known
    index = pd.DatetimeIndex(dates)
    assert index.freq is None
    endog = pd.Series(range(len(index)), index=index, dtype=float, name="y")

    with pytest.raises(ValueError, match="frequency of the index of endog could not"):
        func(endog)


MONTH_STARTS = [pd.Timestamp(2000 + i // 12, i % 12 + 1, 1) for i in range(24)]
QUARTER_ENDS = [pd.Timestamp(1959, 7, 1) + pd.offsets.QuarterEnd(i) for i in range(1, 9)]


@pytest.mark.parametrize(
    "dates, period, start",
    [(MONTH_STARTS, 12, "2000.1"), (QUARTER_ENDS, 4, "1959.3")],
    ids=["month_starts", "quarter_ends"],
)
def test_datetime_index_frequency_inferred(no_x12, dates, period, start):
    # Without a freq of its own, the period of the index is inferred from the
    # dates, and the start is the year and month or quarter of the first date
    index = pd.DatetimeIndex(dates)
    assert index.freq is None
    endog = pd.Series(range(len(index)), index=index, dtype=float, name="y")

    spec = x13.x13_arima_analysis(endog, speconly=True)

    assert f"period={period}" in spec.split()
    assert f"start={start}" in spec.split()


def _loud_binary(path, stream):
    # FAKE_X13, which first prints 1 MiB to stdout or stderr when it is run
    # without arguments, as the lookup does (X-13 prints a short message),
    # more than an operating system pipe holds
    redirect = " >&2" if stream == "stderr" else ""
    noise = (
        'if [ "$#" -eq 0 ]; then\n'
        f"    {{ dd if=/dev/zero bs=1024 count=1024 2>/dev/null; }}{redirect}\n"
        "fi\n"
    )
    script = FAKE_X13.replace("#!/bin/sh\n", "#!/bin/sh\n" + noise, 1)
    path.write_text(script, encoding="ascii", newline="\n")
    path.chmod(0o755)
    return path


@fake_binaries
@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_lookup_binary_with_long_output(tmp_path, stream):
    # The lookup does not read what the binary prints. Sent to a pipe, the
    # output would fill it, and the binary would wait for it to be read and
    # the lookup for the binary to exit, forever. The analysis runs in a
    # thread, so that the test fails instead of hanging.
    binary = _loud_binary(tmp_path / "x13as", stream)
    outcome = {}

    def analysis():
        try:
            outcome["res"] = x13.x13_arima_analysis(_monthly_endog(), x12path=tmp_path)
        except Exception as exc:  # raised again below
            outcome["exc"] = exc

    thread = threading.Thread(target=analysis, daemon=True)
    thread.start()
    thread.join(60)

    assert not thread.is_alive(), "the lookup of the binary did not return"
    if "exc" in outcome:
        raise outcome["exc"]
    assert outcome["res"].stdout.decode().strip() == str(binary)


def _failing_binary(path):
    # A binary that exits with an error status, which X-13 does not do when it
    # is run without arguments
    path.write_text("#!/bin/sh\nexit 1\n", encoding="ascii", newline="\n")
    path.chmod(0o755)


@fake_binaries
def test_lookup_skips_failing_binary(tmp_path):
    # A binary that exits with an error status is not used: the next name is
    # tried, and without a binary that works, none is found
    directory = _fake_binaries(tmp_path, "x12a")
    _failing_binary(directory / "x13as")

    res = x13.x13_arima_analysis(_monthly_endog(), x12path=directory)

    assert res.stdout.decode().strip() == str(directory / "x12a")
    _failing_binary(directory / "x12a")
    with pytest.raises(x13.X13NotFoundError, match="x12a and x13as not found"):
        x13.x13_arima_analysis(_monthly_endog(), x12path=directory)


@fake_binaries
@pytest.mark.parametrize("ext", ["", ".exe"], ids=["posix_names", "windows_names"])
@pytest.mark.parametrize("name", ["x13as", "x12a", "x13as_ascii", "x13as_html"])
def test_x12path_names_binary(tmp_path, monkeypatch, name, ext):
    # x12path that names a binary runs that binary, also if its directory has
    # binaries that come before it in the search order
    names = [binary + ext for binary in ("x13as", "x12a", "x13as_ascii", "x13as_html")]
    directory = _fake_binaries(tmp_path / "bin", *names)
    other = _fake_binaries(tmp_path / "other", *names)
    monkeypatch.setenv("X13PATH", str(other))
    monkeypatch.setenv("X12PATH", str(other))
    binary = directory / f"{name}{ext}"

    res = x13.x13_arima_analysis(_monthly_endog(), x12path=binary)

    assert res.stdout.decode().strip() == str(binary)


@fake_binaries
def test_x12path_named_binary_missing(tmp_path):
    # If the named binary is not there, the other binaries in its directory are
    # tried in the usual order, as before
    directory = _fake_binaries(tmp_path, "x12a", "x13as_html")

    res = x13.x13_arima_analysis(_monthly_endog(), x12path=directory / "x13as_ascii")

    assert res.stdout.decode().strip() == str(directory / "x12a")


@pytest.mark.parametrize(
    "endog", [list(range(24)), tuple(range(24))], ids=["list", "tuple"]
)
def test_rawspec_endog_not_pandas(no_x12, endog):
    rawspec = "series{}\nx11{mode=add}"

    spec = x13.x13_arima_analysis(
        endog, start="1990-04-01", freq="QS", rawspec=rawspec, speconly=True
    )

    # start and freq give the dates of the series spec, which is spliced into
    # the series spec of rawspec; save is added to the x11 spec
    data = " ".join(str(value) for value in range(24))
    expected = f"""
    Series {{
        data=({data})
        appendbcst=no
        appendfcst=no
        period=4
        start=1990.2
        title="Unnamed Series"
        name="Unnamed Series"
    }}
    x11{{
        save=(d11 d12 d13)
        mode=add}}
    """
    assert spec.split() == expected.split()


def test_rawspec_start_freq_with_dates_ignored(no_x12):
    endog = _monthly_endog()

    spec = x13.x13_arima_analysis(
        endog, start="1990-04-01", freq="QS", rawspec="series{}", speconly=True
    )

    # the dates of the index are used, as without rawspec
    assert spec == x13.x13_arima_analysis(endog, rawspec="series{}", speconly=True)
    assert "start=2000.1" in spec.split()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"diff": (1, 1)},
        {"exog": pd.Series(range(36), name="x")},
        {"exog": pd.DataFrame({"x": range(36), "y": range(36, 72)})},
    ],
    ids=["diff", "exog_series", "exog_dataframe"],
)
def test_rawspec_diff_exog_raise(no_x12, kwargs):
    with pytest.raises(ValueError, match="other arguments not allowed for diff and exog"):
        x13.x13_arima_analysis(
            _monthly_endog(), rawspec="series{}", speconly=True, **kwargs
        )


# The spec that x13_arima_analysis writes, compared with the spec written out
# by hand from the X-13ARIMA-SEATS syntax. X-13 reads specs in free format, so
# the comparison ignores the amount and kind of whitespace between the tokens,
# but not whether there is any.
def _normalize(spec):
    return " ".join(spec.split())


QUARTERLY_VALUES = [1.5, 2.25, 3.0, 3.75, 4.5, 5.25, 6.0, 6.75]
QUARTERLY_DATA = "1.5 2.25 3.0 3.75 4.5 5.25 6.0 6.75"


def _quarterly_endog(name="gdp"):
    index = pd.period_range("1959Q3", periods=8, freq="Q")
    return pd.Series(QUARTERLY_VALUES, index=index, name=name)


def _series_spec(period=4, start="1959.3", name="gdp", title=None, data=None):
    title = name if title is None else title
    data = QUARTERLY_DATA if data is None else data
    return f"""
    Series {{
        data=({data})
        appendbcst=no
        appendfcst=no
        period={period}
        start={start}
        title="{title}"
        name="{name}"
    }}
    """


DEFAULT_OPTIONS = """
transform{function=auto}
outlier{}
automdl{
    maxorder = (2 1)
    maxdiff = (2 1)
}
x11{ save=(d11 d12 d13)
     savelog=(fd8 m7 q)}
"""


def test_spec_default(no_x12):
    spec = x13.x13_arima_analysis(_quarterly_endog(), speconly=True)

    assert _normalize(spec) == _normalize(_series_spec() + DEFAULT_OPTIONS)


AUTO = "transform{function=auto} outlier{}"
AUTOMDL = "automdl{ maxorder = (2 1) maxdiff = (2 1) }"
X11 = "x11{ save=(d11 d12 d13) savelog=(fd8 m7 q)}"


@pytest.mark.parametrize(
    "kwargs, options",
    [
        ({"log": True}, f"transform{{function=log}} outlier{{}} {AUTOMDL} {X11}"),
        ({"log": False}, f"transform{{function=none}} outlier{{}} {AUTOMDL} {X11}"),
        ({"log": None}, f"{AUTO} {AUTOMDL} {X11}"),
        ({"outlier": False}, f"transform{{function=auto}} {AUTOMDL} {X11}"),
        (
            {"maxorder": (3, 2), "maxdiff": (1, 0)},
            f"{AUTO} automdl{{ maxorder = (3 2) maxdiff = (1 0) }} {X11}",
        ),
        (
            {"maxdiff": None, "diff": (1, 0)},
            f"{AUTO} automdl{{ maxorder = (2 1) diff = (1 0) }} {X11}",
        ),
        # maxdiff takes precedence over diff
        ({"diff": (1, 0)}, f"{AUTO} {AUTOMDL} {X11}"),
        (
            {"trading": True},
            f"{AUTO} {AUTOMDL} regression{{ variables = (td) }} {X11}",
        ),
        ({"forecast_periods": 6}, f"{AUTO} {AUTOMDL} forecast{{ maxlead = (6) }} {X11}"),
        ({"forecast_periods": 0}, f"{AUTO} {AUTOMDL} forecast{{ maxlead = (0) }} {X11}"),
    ],
    ids=[
        "log",
        "no_log",
        "auto_log",
        "no_outlier",
        "maxorder_maxdiff",
        "diff",
        "diff_ignored",
        "trading",
        "forecast",
        "no_forecast",
    ],
)
def test_spec_options(no_x12, kwargs, options):
    spec = x13.x13_arima_analysis(_quarterly_endog(), speconly=True, **kwargs)

    assert _normalize(spec) == _normalize(_series_spec() + options)


def _exog_frame():
    index = _quarterly_endog().index
    return pd.DataFrame(
        {"a": np.arange(8.0), "b": np.arange(10.0, 18.0)}, index=index
    )


# X-13 reads the user regressors observation by observation, with the values
# of all variables for one date before those of the next date
FRAME_DATA = "0.0 10.0 1.0 11.0 2.0 12.0 3.0 13.0 4.0 14.0 5.0 15.0 6.0 16.0 7.0 17.0"
REORDERED_DATA = (
    "10.0 0.0 11.0 1.0 12.0 2.0 13.0 3.0 14.0 4.0 15.0 5.0 16.0 6.0 17.0 7.0"
)
SERIES_DATA = "0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0"


@pytest.mark.parametrize(
    "make_exog, names, data",
    [
        (_exog_frame, "a b", FRAME_DATA),
        (lambda: _exog_frame()[["b", "a"]], "b a", REORDERED_DATA),
        (lambda: _exog_frame()["a"].rename("shock"), "shock", SERIES_DATA),
        # names that are not str are replaced by generated names
        (lambda: _exog_frame()["a"].rename(None), "x1", SERIES_DATA),
        (lambda: _exog_frame()["a"].rename(3), "x1", SERIES_DATA),
    ],
    ids=["frame", "frame_reordered", "series", "unnamed", "int_name"],
)
@pytest.mark.parametrize("trading", [False, True])
def test_spec_exog(no_x12, make_exog, names, data, trading):
    spec = x13.x13_arima_analysis(
        _quarterly_endog(), exog=make_exog(), trading=trading, speconly=True
    )

    variables = "variables = (td)" if trading else ""
    regression = f"regression{{ {variables} user = ({names}) data = ({data}) }}"
    options = f"{AUTO} {AUTOMDL} {regression} {X11}"
    assert _normalize(spec) == _normalize(_series_spec() + options)


@pytest.mark.parametrize("exog", [np.arange(8.0), list(range(8))])
def test_spec_exog_not_pandas(no_x12, exog):
    with pytest.raises(ValueError, match="exog is not a Series or DataFrame"):
        x13.x13_arima_analysis(_quarterly_endog(), exog=exog, speconly=True)


ENDOG_FORMS = {
    "period_quarterly": (_quarterly_endog, {}, 4, "1959.3", "gdp"),
    "datetime_quarter_start": (
        lambda: _quarterly_endog().set_axis(
            pd.date_range("1959-07-01", periods=8, freq="QS")
        ),
        {},
        4,
        "1959.3",
        "gdp",
    ),
    "datetime_quarter_end": (
        lambda: _quarterly_endog().set_axis(
            pd.date_range("1959-09-30", periods=8, freq=QUARTER_END)
        ),
        {},
        4,
        "1959.3",
        "gdp",
    ),
    "period_monthly": (
        lambda: _quarterly_endog().set_axis(
            pd.period_range("1959-07", periods=8, freq="M")
        ),
        {},
        12,
        "1959.7",
        "gdp",
    ),
    "datetime_month_start": (
        lambda: _quarterly_endog().set_axis(
            pd.date_range("1959-07-01", periods=8, freq="MS")
        ),
        {},
        12,
        "1959.7",
        "gdp",
    ),
    "datetime_month_end": (
        lambda: _quarterly_endog().set_axis(
            pd.date_range("1959-07-31", periods=8, freq=MONTH_END)
        ),
        {},
        12,
        "1959.7",
        "gdp",
    ),
    # no freq on the index: it is inferred from the dates
    "datetime_no_freq": (
        lambda: _quarterly_endog().set_axis(
            pd.DatetimeIndex(
                ["2000-07-31", "2000-08-31", "2000-09-30", "2000-10-31"]
                + ["2000-11-30", "2000-12-31", "2001-01-31", "2001-02-28"]
            )
        ),
        {},
        12,
        "2000.7",
        "gdp",
    ),
    "dataframe": (lambda: _quarterly_endog().to_frame(), {}, 4, "1959.3", "gdp"),
    "dataframe_int_column": (
        lambda: _quarterly_endog().to_frame().set_axis([0], axis=1),
        {},
        4,
        "1959.3",
        "Unnamed Series",
    ),
    "list": (
        lambda: list(QUARTERLY_VALUES),
        {"start": "1959-07-01", "freq": "QS"},
        4,
        "1959.3",
        "Unnamed Series",
    ),
    "tuple": (
        lambda: tuple(QUARTERLY_VALUES),
        {"start": "1959-07-01", "freq": "MS"},
        12,
        "1959.7",
        "Unnamed Series",
    ),
    "ndarray": (
        lambda: np.array(QUARTERLY_VALUES),
        {"start": pd.Timestamp("1959-09-30"), "freq": pd.offsets.QuarterEnd()},
        4,
        "1959.3",
        "Unnamed Series",
    ),
}


@pytest.mark.parametrize("form", list(ENDOG_FORMS))
def test_spec_endog_forms(no_x12, form):
    make_endog, kwargs, period, start, name = ENDOG_FORMS[form]

    spec = x13.x13_arima_analysis(make_endog(), speconly=True, **kwargs)

    expected = _series_spec(period=period, start=start, name=name)
    assert _normalize(spec) == _normalize(expected + DEFAULT_OPTIONS)


def test_spec_endog_int_and_long_name(no_x12):
    # X-13 limits the series name to 64 and the title to 79 characters
    name = "".join(chr(ord("a") + i % 26) for i in range(100))
    endog = pd.Series(
        np.arange(1, 9), index=_quarterly_endog().index, name=name
    )

    spec = x13.x13_arima_analysis(endog, speconly=True)

    expected = _series_spec(
        name=name[:64], title=name[:79], data="1 2 3 4 5 6 7 8"
    )
    assert _normalize(spec) == _normalize(expected + DEFAULT_OPTIONS)


@pytest.mark.parametrize(
    "index",
    [
        pd.date_range("2000-01-02", periods=8, freq="W"),
        pd.period_range("2000-01-02", periods=8, freq="W"),
        pd.date_range("2000-01-01", periods=8, freq="D"),
        pd.date_range("2000-12-31", periods=8, freq=YEAR_END),
        pd.period_range("2000", periods=8, freq="Y"),
    ],
    ids=["weekly", "weekly_period", "daily", "annual", "annual_period"],
)
def test_endog_frequency_not_supported(no_x12, index):
    endog = pd.Series(QUARTERLY_VALUES, index=index)

    with pytest.raises(ValueError, match="Only monthly and quarterly periods"):
        x13.x13_arima_analysis(endog, speconly=True)


def test_endog_dataframe_two_columns(no_x12):
    endog = _exog_frame()

    with pytest.raises(ValueError, match="more than one column"):
        x13.x13_arima_analysis(endog, speconly=True)


@pytest.mark.parametrize(
    "kwargs", [{}, {"start": "1959-07-01"}, {"freq": "QS"}], ids=["none", "start", "freq"]
)
def test_endog_not_pandas_needs_start_and_freq(no_x12, kwargs):
    with pytest.raises(ValueError, match="start and freq cannot be none"):
        x13.x13_arima_analysis(QUARTERLY_VALUES, speconly=True, **kwargs)


def test_rawspec_other_arguments_raise(no_x12):
    # diff raises with a spec (start and freq give the dates of array_like
    # input, see the tests of the rawspec argument checks)
    with pytest.raises(ValueError, match="other arguments not allowed"):
        x13.x13_arima_analysis(
            _quarterly_endog(), rawspec="series{}", speconly=True, diff=(1, 1)
        )


@pytest.mark.parametrize(
    "rawspec, expected",
    [
        # the output tables are added to the list that is saved
        (
            "series{\n  modelspan=(,)\n}\nx11{ save=(d10) mode=mult }",
            "modelspan=(,) } x11{ save=( d11 d12 d13 d10) mode=mult }",
        ),
        # a save argument is added to an x11 spec without one; any case
        (
            "SERIES { }\ntransform{function=log}\nX11 {mode=add}",
            "} transform{function=log} X11 { save=(d11 d12 d13) mode=add}",
        ),
        # an x11 spec is added if there is none
        (
            "series {span=(1960.1,)}\narima{model=(0 1 1)}",
            "span=(1960.1,)} arima{model=(0 1 1)} x11 {save=(d11 d12 d13)}",
        ),
    ],
    ids=["x11_save", "x11_no_save", "no_x11"],
)
def test_rawspec_spliced(no_x12, rawspec, expected):
    spec = x13.x13_arima_analysis(_quarterly_endog(), rawspec=rawspec, speconly=True)

    # the series data and attributes come first in the series spec of rawspec
    series = _normalize(_series_spec())
    assert _normalize(spec) == series[: -len(" }")] + " " + expected


@pytest.mark.parametrize(
    "x12path", ["", "x13as", "x13as.exe"], ids=["directory", "binary", "binary_exe"]
)
@pytest.mark.parametrize(
    "func", [x13.x13_arima_analysis, x13.x13_arima_select_order]
)
def test_x13_not_found_x12path(tmp_path, x12path, func):
    # a directory without the binary, or a binary in it that does not exist;
    # the environment variables are not used when x12path is given
    with pytest.raises(X13NotFoundError, match="x12a and x13as not found"):
        func(_quarterly_endog(), x12path=str(tmp_path / x12path))


@pytest.mark.parametrize("variable", ["X13PATH", "X12PATH"])
def test_x13_not_found_environment(tmp_path, monkeypatch, variable):
    monkeypatch.delenv("X13PATH", raising=False)
    monkeypatch.delenv("X12PATH", raising=False)
    monkeypatch.setenv(variable, str(tmp_path))

    with pytest.raises(X13NotFoundError, match="x12a and x13as not found"):
        x13.x13_arima_analysis(_quarterly_endog())


NO_DATES = {
    "range_index": pd.Series,
    "int_index": lambda values: pd.Series(values, index=range(10, 34)),
    "float_index": lambda values: pd.Series(values, index=np.arange(24.0)),
    "str_index": lambda values: pd.Series(values, index=[f"t{i}" for i in range(24)]),
    "dataframe": lambda values: pd.DataFrame({0: values}),
}


@pytest.mark.parametrize("form", list(NO_DATES))
@pytest.mark.parametrize(
    "start, freq, period, first",
    [("1990-01-01", "QS", 4, "1990.1"), ("1990-05-01", "MS", 12, "1990.5")],
    ids=["quarterly", "monthly"],
)
def test_spec_endog_without_dates(no_x12, form, start, freq, period, first):
    values = [float(i) for i in range(24)]
    endog = NO_DATES[form](values)

    spec = x13.x13_arima_analysis(endog, start=start, freq=freq, speconly=True)

    # the same spec as for the values without an index
    expected = x13.x13_arima_analysis(
        np.array(values), start=start, freq=freq, speconly=True
    )
    assert spec == expected
    data = " ".join(str(value) for value in values)
    series = f"""
    Series {{
        data=({data})
        appendbcst=no
        appendfcst=no
        period={period}
        start={first}
        title="Unnamed Series"
        name="Unnamed Series"
    }}
    """
    assert spec[: spec.index("}") + 1].split() == series.split()


def test_spec_endog_without_dates_named(no_x12):
    values = [float(i) for i in range(24)]
    endog = pd.Series(values, name="gdp")

    spec = x13.x13_arima_analysis(endog, start="1990-01-01", freq="QS", speconly=True)

    expected = x13.x13_arima_analysis(
        np.array(values), start="1990-01-01", freq="QS", speconly=True
    )
    assert spec == expected.replace('"Unnamed Series"', '"gdp"')


@pytest.mark.parametrize(
    "kwargs", [{}, {"start": "1990-01-01"}, {"freq": "QS"}], ids=["none", "start", "freq"]
)
def test_endog_without_dates_needs_start_and_freq(no_x12, kwargs):
    with pytest.raises(ValueError, match="start and freq cannot be none"):
        x13.x13_arima_analysis(pd.Series(range(24)), speconly=True, **kwargs)


_DATES = pd.date_range("2000-01-01", periods=24, freq="MS")
WITH_DATES = {
    "datetime_index": _DATES,
    "period_index": _DATES.to_period("M"),
    "object_timestamps": pd.Index(list(_DATES), dtype=object),
    "object_dates": pd.Index([date.date() for date in _DATES], dtype=object),
    "categorical": pd.CategoricalIndex(_DATES),
}


@pytest.mark.parametrize("index", list(WITH_DATES))
def test_spec_endog_dates_used_with_start_and_freq(no_x12, index):
    endog = pd.Series([float(i) for i in range(24)], index=WITH_DATES[index])

    spec = x13.x13_arima_analysis(endog, start="1990-01-01", freq="QS", speconly=True)

    # start and freq are not used if the index has the dates
    assert spec == x13.x13_arima_analysis(endog, speconly=True)
    tokens = spec[: spec.index("}") + 1].split()
    assert "period=12" in tokens
    assert "start=2000.1" in tokens
