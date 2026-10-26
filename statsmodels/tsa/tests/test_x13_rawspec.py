"""Regression tests for x13 rawspec file handling that do not need the X-13 binary.

``test_x13`` skips its whole module when x12a/x13as is absent. These checks only
exercise the spec-building path (``speconly=True`` returns before the binary is
called), so they live here and run everywhere.
"""

import pandas as pd

from statsmodels.tsa import x13


def _monthly_endog():
    return pd.Series(range(24), index=pd.period_range("2000-01", periods=24, freq="M"))


def test_rawspec_file_read_through_utf8_reader(monkeypatch, tmp_path):
    monkeypatch.setattr(x13, "_check_x12", lambda x12path=None: "x12dummy")

    specfile = tmp_path / "custom.spc"
    specfile.write_text('series{title="on disk"}\n', encoding="utf-8")

    sentinel = 'series{title="café naïve"}\n'
    calls = []

    def spy(fname):
        calls.append(str(fname))
        return sentinel

    monkeypatch.setattr(x13, "_open_and_read", spy)

    spec = x13.x13_arima_analysis(
        _monthly_endog(), rawspec=str(specfile), speconly=True
    )

    assert str(specfile) in calls
    assert "café naïve" in spec
