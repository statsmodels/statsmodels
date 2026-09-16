"""Tests for statsmodels.iolib.foreign"""
import numpy as np

from statsmodels.iolib.foreign import savetxt


def test_savetxt_plain_array(tmp_path):
    # savetxt is exported from statsmodels.iolib.api but had no test
    # coverage at all.
    path = tmp_path / "out.csv"
    x = np.array([[1.0, 2.5], [3.0, 4.5]])
    savetxt(str(path), x, fmt="%.2f", delimiter=",")

    content = path.read_text(encoding="utf-8")
    assert content == "1.00,2.50\n3.00,4.50\n"


def test_savetxt_with_names(tmp_path):
    path = tmp_path / "out.csv"
    x = np.array([[1.0, 2.5], [3.0, 4.5]])
    savetxt(str(path), x, names=["a", "b"], fmt="%.1f", delimiter=",")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "a,b"
    assert lines[1] == "1.0,2.5"
    assert lines[2] == "3.0,4.5"


def test_savetxt_structured_array_uses_field_names(tmp_path):
    path = tmp_path / "out.csv"
    x = np.array(
        [(1.0, 2), (3.0, 4)], dtype=[("weight", float), ("count", int)]
    )
    savetxt(str(path), x, fmt=["%.1f", "%d"], delimiter=",")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "weight,count"
    assert lines[1] == "1.0,2"
    assert lines[2] == "3.0,4"
