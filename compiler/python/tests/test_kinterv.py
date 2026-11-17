import numpy as np
import pytest

iree_rt = pytest.importorskip("iree.runtime")
pytest.importorskip("iree.compiler")

from causal_compiler import kinterv


def test_fused_matches_statsmodels():
    try:
        results = kinterv.fit_logit(3000)
        frame = kinterv.make_data(500, seed=1)
    except OSError:
        pytest.skip("IHDP download unavailable")
    w, c = kinterv.unpack(results)
    x = np.ascontiguousarray(frame[kinterv.COVARIATES].to_numpy())
    doses = np.array([0.0, 1.5, 4.0])
    device, fused = kinterv.load(kinterv.compile_fused(500, w.shape[0], 3, "generic"))
    args = [iree_rt.asdevicearray(device, a) for a in (x, w, c, doses)]
    got = kinterv.to_numpy(fused(*args))
    ref = np.column_stack(
        [results.predict(frame.assign(T=t)).to_numpy() for t in doses])
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)
