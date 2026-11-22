"""Regression tests for x13 file handling that do not need the X-13 binary.

``test_x13`` skips its whole module when x12a/x13as is absent. These checks only
exercise the spec-building path (``speconly=True`` returns before the binary is
called) and the clean-up of the temporary files with ``run_spec`` replaced by a
stub, so they live here and run everywhere.
"""

from statsmodels.compat.python import PYTHON_IMPL_WASM

import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

import statsmodels
from statsmodels.tools.sm_exceptions import IOWarning, X13Error
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
