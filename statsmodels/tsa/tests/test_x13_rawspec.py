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

x13._check_x12 = lambda x12path=None: "x12dummy"
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
    monkeypatch.setattr(x13, "_check_x12", lambda x12path=None: "x12dummy")


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


def test_rawspec_invalid_path(no_x12, tmp_path):
    missing = tmp_path / "missing.spc"

    with pytest.raises(ValueError, match="not valid path or spec string"):
        x13.x13_arima_analysis(_monthly_endog(), rawspec=str(missing), speconly=True)


def test_rawspec_path_object(no_x12, tmp_path):
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


def test_rawspec_file_utf8_bom(no_x12, tmp_path):
    specfile = tmp_path / "bom.spc"
    specfile.write_bytes(b"\xef\xbb\xbf" + (SPEC_TEMPLATE % "bom").encode("utf-8"))

    spec = x13.x13_arima_analysis(
        _monthly_endog(), rawspec=str(specfile), speconly=True
    )

    assert "\ufeff" not in spec
    assert 'title="bom"' in spec


def test_rawspec_file_not_utf8(no_x12, tmp_path):
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
