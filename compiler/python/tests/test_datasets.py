import numpy as np
import pandas as pd
import pytest

from causal_compiler.datasets import IHDP_COLUMNS, load_ihdp, resample_rows


def test_resample_rows_shape_and_determinism():
    frame = pd.DataFrame({"a": np.arange(10), "b": np.arange(10) * 2.0})
    first = resample_rows(frame, 1000, seed=3)
    second = resample_rows(frame, 1000, seed=3)
    assert len(first) == 1000
    pd.testing.assert_frame_equal(first, second)
    assert set(first["a"]) <= set(frame["a"])


def test_load_ihdp(tmp_path):
    try:
        frame = load_ihdp(1, cache_dir=tmp_path)
    except OSError:
        pytest.skip("network unavailable")
    assert list(frame.columns) == IHDP_COLUMNS
    assert frame.shape == (747, 30)
    assert set(frame["A"].unique()) <= {0.0, 1.0}
