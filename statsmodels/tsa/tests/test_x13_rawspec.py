"""Regression tests for x13 rawspec file handling that do not need the X-13 binary.

``test_x13`` skips its whole module when x12a/x13as is absent. These checks only
exercise the spec-building path (``speconly=True`` returns before the binary is
called), so they live here and run everywhere.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

import statsmodels
from statsmodels.tsa import x13

# Non-ASCII text that is not valid in an ASCII locale. The capital A with acute
# has the UTF-8 bytes c3 81, and 0x81 is undefined in cp1252, so decoding with
# the Windows locale codec fails or yields mojibake instead of this text.
NON_ASCII_TITLE = "café naïve Á"
SPEC_TEMPLATE = 'series{title="%s"}\n'

# Run in a fresh interpreter whose locale encoding is not UTF-8. The code is
# ASCII only and the result is written as JSON with escaped characters, so
# nothing in the child depends on the locale except the read under test.
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
sys.stdout.write(json.dumps(spec))
"""


def _monthly_endog():
    return pd.Series(range(24), index=pd.period_range("2000-01", periods=24, freq="M"))


@pytest.fixture
def no_x12(monkeypatch):
    monkeypatch.setattr(x13, "_check_x12", lambda x12path=None: "x12dummy")


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

    # A relative name keeps the argument ASCII whatever the temp directory is.
    res = subprocess.run(
        [sys.executable, "-c", CHILD_CODE, specfile.name],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="ascii",
        timeout=120,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    spec = json.loads(res.stdout)
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
